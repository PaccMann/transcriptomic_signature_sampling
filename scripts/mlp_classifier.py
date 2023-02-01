import argparse
import json
import logging
import os
import warnings

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
from fdsa.models.set_matching.dnn import DNNSetMatching
from fdsa.utils.helper import setup_logger
from imblearn.metrics import classification_report_imbalanced
from signature_sampling.torch_data import TCGADataset
from sklearn.metrics import balanced_accuracy_score
from torch.utils.data import DataLoader

warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser()

parser.add_argument(
    "main_dir", type=str, help="Path to the main directory containing the datasets."
)

parser.add_argument(
    "training_params", type=str, help="Path to the training parameters json file."
)

parser.add_argument(
    "results_path", type=str, help="Path to save the results, logs and best model."
)

parser.add_argument("model_name", type=str, help="Name of model.")
parser.add_argument("seed", type=int, help="Seed for the model.")
parser.add_argument(
    "--repeat_id", type=str, help="Repetition number of experiment.", default=None
)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def main(
    main_dir: str,
    training_params: str,
    results_path: str,
    model_name: str,
    seed: int,
    repeat_id: str,
):

    torch.manual_seed(seed)
    np.random.seed(seed)

    if repeat_id:
        results_dir = os.path.join(results_path, f"run_{repeat_id}")
    else:
        results_dir = os.path.join(results_path)

    os.makedirs(os.path.join(results_dir, "weights"), exist_ok=True)
    os.makedirs(os.path.join(results_dir, "results"), exist_ok=True)

    # setup logging
    logger = setup_logger(
        "MLP-CRC", os.path.join(results_dir, "mlp_crc.log"), logging.DEBUG
    )

    with open(training_params, "r") as readjson:
        train_params = json.load(readjson)

    train_params["seed"] = seed
    train_params["main_dir"] = main_dir

    with open(os.path.join(results_dir, "params.json"), "w") as f:
        json.dump(train_params, f)

    lr_ = train_params.get("lr", 1e-3)
    l2_weight = train_params.get("l2_weight", 0.0001)

    model = DNNSetMatching(train_params).to(device)

    loss_fn = nn.CrossEntropyLoss(reduction="none")

    optimizer = optim.Adam(model.parameters(), lr=lr_, weight_decay=l2_weight)

    epochs = train_params.get("epochs", 100)

    train_loss = []
    val_loss = []
    min_loss = np.inf

    xtrain_path = os.path.join(main_dir, train_params["xtrain"])
    ytrain_path = os.path.join(main_dir, train_params["ytrain"])

    xval_path = os.path.join(main_dir, train_params["xval"])
    yval_path = os.path.join(main_dir, train_params["yval"])

    xtest_path = os.path.join(main_dir, train_params["xtest"])
    ytest_path = os.path.join(main_dir, train_params["ytest"])

    real_weight = train_params.get("real_weight", 1.0)
    synthetic_weight = train_params.get("synthetic_weight", 1.0)

    train_dataset = TCGADataset(
        xtrain_path,
        ytrain_path,
        sample_weights={"real": real_weight, "synthetic": synthetic_weight},
    )
    val_dataset = TCGADataset(
        xval_path,
        yval_path,
        sample_weights={"real": real_weight, "synthetic": synthetic_weight},
    )
    test_dataset = TCGADataset(xtest_path, ytest_path)

    bs = train_params.get("batch_size", 128)
    train_loader = DataLoader(train_dataset, batch_size=bs, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=bs)
    val_loader = DataLoader(val_dataset, batch_size=bs)

    for e in range(epochs):
        logger.info("=== Epoch [{}/{}]".format(e + 1, epochs))

        for x_train, y_train, weights_train in train_loader:

            y_train_pred = model(x_train.to(device))
            loss_train = loss_fn(y_train_pred, y_train.to(device)) * weights_train.to(
                device
            )
            loss_train = loss_train.mean()

            optimizer.zero_grad()
            loss_train.backward()
            optimizer.step()

        loss_train_numpy = loss_train.detach().cpu().numpy()
        y_train_pred_numpy = np.argmax(y_train_pred.detach().cpu().numpy(), axis=1)
        y_train_numpy = y_train.detach().cpu().numpy()

        train_loss.append(loss_train_numpy)
        logger.info("train loss = {}".format(loss_train_numpy))

        logger.info(
            "balanced acuracy = {}".format(
                balanced_accuracy_score(y_train_numpy, y_train_pred_numpy)
            )
        )

        model.eval()
        avg_valid_loss = 0
        val_metric = {}
        val_preds = []
        val_true = []
        for idx, (x_val, y_val, weights_val) in enumerate(val_loader):

            y_val_pred = model(x_val.to(device))
            loss_val = loss_fn(y_val_pred, y_val.to(device)) * weights_val.to(device)
            loss_val = loss_val.mean()

            y_val_pred_numpy = np.argmax(y_val_pred.detach().cpu().numpy(), axis=1)
            y_val_numpy = y_val.detach().cpu().numpy()
            val_preds.append(y_val_pred_numpy)
            val_true.append(y_val_numpy)

            avg_valid_loss = (
                avg_valid_loss * idx + loss_val.detach().cpu().numpy()
            ) / (idx + 1)

        val_metric["val_balanced_accuracy"] = balanced_accuracy_score(
            y_val_numpy,
            y_val_pred_numpy,
        )

        # TODO: save val_metric
        with open(os.path.join(results_dir, "results", "val_metric.json"), "w") as f:
            json.dump(val_metric, f)

        val_loss.append(avg_valid_loss)
        logger.info("avg validation loss = {}".format(avg_valid_loss))

        if avg_valid_loss < min_loss:
            steps = 0
            min_loss = avg_valid_loss
            best_model = model
            logger.info(
                "Current best model with validation loss {} in epoch {}".format(
                    avg_valid_loss, e + 1
                )
            )
            torch.save(
                {
                    "epoch": e,
                    "model_state_dict": best_model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "train_loss": loss_train,
                    "valid_loss": avg_valid_loss
                    # 'learning_rate': lr_expscheduler.get_last_lr()[0]
                },
                os.path.join(results_dir, "weights", model_name),
            )
        # Early Stopping
        if avg_valid_loss > min_loss:
            steps += 1
        if steps == train_params.get("early_stopping_wait", 10):
            # print(f"Early stopping at {e} epochs")
            logger.info(f"Early stopping at {e} epochs")
            break

    torch.save(train_loss, os.path.join(results_dir, "results", "train_loss"))
    torch.save(val_loss, os.path.join(results_dir, "results", "avg_valid_loss"))

    clrs = sns.color_palette("Set2", 2)
    fig, ax = plt.subplots(constrained_layout=True)
    fig.suptitle(f"Train and Validation Loss of {model_name}")
    ax.set_xlabel("Epochs")
    ax.set_ylabel("Loss")
    ax.plot(range(len(train_loss)), train_loss, color=clrs[0], label="Training Loss")
    ax.plot(range(len(val_loss)), val_loss, color=clrs[1], label="Validation Loss")
    ax.legend()
    fig.savefig(os.path.join(results_dir, "results", f"{model_name}_loss.png"), dpi=300)

    # test

    test_metric = {}
    avg_test_loss = 0
    test_preds = []
    test_true = []
    y_test_latent = []
    best_model.eval()

    for idx, (x_test, y_test, weights_test) in enumerate(test_loader):

        y_test_pred = best_model(x_test)
        loss_test = loss_fn(y_test_pred, y_test.to(device)) * weights_test.to(device)
        loss_test = loss_test.mean()

        y_test_latent.append(y_test_pred)

        y_test_pred_numpy = np.argmax(y_test_pred.detach().cpu().numpy(), axis=1)
        y_test_numpy = y_test.detach().cpu().numpy()
        test_preds.append(y_test_pred_numpy)
        test_true.append(y_test_numpy)

        avg_test_loss = (avg_test_loss * idx + loss_test.detach().cpu().numpy()) / (
            idx + 1
        )

    torch.save(y_test_latent, os.path.join(results_dir, "results", "test_latent"))
    test_preds = np.concatenate(test_preds)
    test_true = np.concatenate(test_true)

    clf_report = classification_report_imbalanced(
        test_true,
        test_preds,
        target_names=["CMS1", "CMS2", "CMS3", "CMS4"],
        output_dict=True,
        zero_division=0,
    )

    pd.DataFrame.from_dict(clf_report).T.to_csv(
        os.path.join(results_dir, "results", "clf_report.csv")
    )

    test_metric["test_balanced_accuracy"] = balanced_accuracy_score(
        y_test_numpy,
        y_test_pred_numpy,
    )
    test_metric["loss"] = avg_test_loss

    with open(os.path.join(results_dir, "results", "test_metric.json"), "w") as f:
        json.dump(test_metric, f)

    logger.info("avg test loss = {}".format(avg_test_loss))


if __name__ == "__main__":
    args = parser.parse_args()
    main(
        args.main_dir,
        args.training_params,
        args.results_path,
        args.model_name,
        args.seed,
        args.repeat_id,
    )
