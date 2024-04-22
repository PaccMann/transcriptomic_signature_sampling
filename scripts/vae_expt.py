import argparse
import json
import logging
import os
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn.functional as F
import torch.optim as optim
from fdsa.utils.helper import setup_logger
from sklearn import metrics
from sklearn.calibration import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from torch.utils.data import DataLoader
from umap import UMAP

from signature_sampling.torch_data import TCGADataset
from signature_sampling.utils import purity_score
from signature_sampling.vae import VAE

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
parser.add_argument("seed", type=int, help="Initialisation seed.")

parser.add_argument(
    "--repeat_id", type=str, help="Repetition number of Expt.", default=None
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
        "VAE-CRC", os.path.join(results_dir, "vae_crc.log"), logging.DEBUG
    )

    with open(training_params, "r") as readjson:
        train_params = json.load(readjson)

    train_params["seed"] = seed

    with open(os.path.join(results_dir, "params.json"), "w") as f:
        json.dump(train_params, f)

    xtrain_path = os.path.join(main_dir, train_params["xtrain"])
    ytrain_path = os.path.join(main_dir, train_params["ytrain"])
    xtest_path = os.path.join(main_dir, train_params["xtest"])
    ytest_path = os.path.join(main_dir, train_params["ytest"])
    xval_path = os.path.join(main_dir, train_params["xval"])
    yval_path = os.path.join(main_dir, train_params["yval"])

    xtrain_df = pd.read_csv(xtrain_path,index_col=0)
    ytrain_df = pd.read_csv(ytrain_path,index_col=0)
    xtest_df = pd.read_csv(xtest_path,index_col=0)
    ytest_df = pd.read_csv(ytest_path,index_col=0)
    xval_df = pd.read_csv(xval_path,index_col=0)
    yval_df = pd.read_csv(yval_path,index_col=0)

    train_dataset = TCGADataset(xtrain_df, ytrain_df, LabelEncoder)
    test_dataset = TCGADataset(xtest_df, ytest_df, train_dataset.label_embedder)
    val_dataset = TCGADataset(xval_df, yval_df, train_dataset.label_embedder)

    bs = train_params.get("batch_size", 32)
    train_loader = DataLoader(
        train_dataset, batch_size=bs, drop_last=False, shuffle=True
    )
    test_loader = DataLoader(test_dataset, batch_size=bs)
    val_loader = DataLoader(val_dataset, batch_size=bs)

    lr_ = train_params.get("lr", 1e-3)
    l2_weight = train_params.get("l2_weight", 0.0001)
    ae_type = train_params.get("ae_type", "vae")
    model = VAE(train_params).to(device)

    optimizer = optim.Adam(model.parameters(), lr=lr_, weight_decay=l2_weight)

    epochs = train_params.get("epochs", 100)

    train_loss = []
    val_loss = []
    min_loss = np.inf

    for e in range(epochs):
        logger.info("=== Epoch [{}/{}]".format(e + 1, epochs))
        latent_train = []
        train_labels = []
        latent_val = []
        val_labels = []

        for x, y in train_loader:
            
            x_recon, z_train, q_z, p_z = model(x.to(device))
            loss_train = model.loss(x_recon, x, q_z, p_z)
           
            optimizer.zero_grad()
            loss_train.backward()
            optimizer.step()

            z_train_numpy = z_train.detach().cpu().numpy()
            y_train_numpy = y.detach().cpu().numpy()

            silhoutte_train = metrics.silhouette_score(z_train_numpy, y_train_numpy)
            latent_train.append(z_train_numpy)
            train_labels.append(y_train_numpy)

        train_loss.append(loss_train.detach().cpu().numpy())
        logger.info("train loss = {}".format(loss_train.detach().cpu().numpy()))
        logger.info("latent silhoutte score = {}".format(silhoutte_train))

        model.eval()
        avg_valid_loss = 0
        avg_silhoutte_val = 0
        for idx, (x_val, y_val) in enumerate(val_loader):
            if ae_type == "vae":
                x_recon, z_val, q_z, p_z = model(x_val.to(device))
                loss_valid = model.loss(x_recon, x_val, q_z, p_z)

            elif ae_type == "ae":
                x_recon, z_val = model(x_val.to(device))
                loss_valid = model.loss(x_recon, x_val)

            z_val_numpy = z_val.detach().cpu().numpy()
            y_val_numpy = y_val.detach().cpu().numpy()

            avg_silhoutte_val = (
                avg_silhoutte_val * idx
                + metrics.silhouette_score(z_val_numpy, y_val_numpy)
            ) / (idx + 1)

            avg_valid_loss = (
                avg_valid_loss * idx + loss_valid.detach().cpu().numpy()
            ) / (idx + 1)
            latent_val.append(z_val_numpy)
            val_labels.append(y_val_numpy)

        val_loss.append(avg_valid_loss)
        logger.info("avg validation loss = {}".format(avg_valid_loss))
        logger.info("mean latent silhoutte score = {}".format(avg_silhoutte_val))

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
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "train_loss": loss_train,
                    "valid_loss": avg_valid_loss,
                    # 'learning_rate': lr_expscheduler.get_last_lr()[0]
                },
                os.path.join(results_dir, "weights", model_name),
            )
            torch.save(
                latent_train, os.path.join(results_dir, "results", "best_latent_train")
            )
            torch.save(
                train_labels, os.path.join(results_dir, "results", "best_train_labels")
            )
            torch.save(
                latent_val, os.path.join(results_dir, "results", "best_latent_val")
            )
            torch.save(
                val_labels, os.path.join(results_dir, "results", "best_val_labels")
            )
            # save val and train latent here
        # Early Stopping
        if avg_valid_loss > min_loss:
            steps += 1
        if steps == train_params.get("early_stopping_wait", 8):
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

    loss_dict = defaultdict(list)
    test_latent = []
    test_labels = []
    test_recon = []
    test_input = []
    best_model.eval()
    for x, y in test_loader:
        x_recon, z, q_z, p_z = best_model(x.to(device))
        loss = best_model.loss(x_recon, x, q_z, p_z)

        test_input.append(x.detach().cpu())
        test_latent.append(z.detach().cpu())
        test_labels.append(y.detach().cpu())
        test_recon.append(x_recon.detach().cpu())

        loss_dict["recon loss"].append(float(F.mse_loss(x_recon, x).data))

        loss_dict["KL Loss"].append(
            float(torch.distributions.kl.kl_divergence(q_z, p_z).sum(-1).mean().data)
        )
        loss_dict["ELBO"].append(float(loss.data))

    test_input = torch.cat(test_input)
    test_recon = torch.cat(test_recon)
    test_latent = torch.cat(test_latent)
    test_labels = torch.cat(test_labels)

    logger.info("Test Info: {}".format({k: np.mean(v) for k, v in loss_dict.items()}))
    test_input_silhoutte = metrics.silhouette_score(test_input, test_labels)
    test_recon_silhoutte = metrics.silhouette_score(test_recon, test_labels)
    test_latent_silhoutte = metrics.silhouette_score(test_latent, test_labels)
    loss_dict.update(
        {
            "test_input_silhoutte": float(test_input_silhoutte),
            "test_recon_silhoutte": float(test_recon_silhoutte),
            "test_latent_silhoutte": float(test_latent_silhoutte),
        }
    )

    with open(os.path.join(results_dir, "results", "test_metrics.json"), "w") as f:
        json.dump(loss_dict, f)

    torch.save(test_input, os.path.join(results_dir, "results", "test_input"))
    torch.save(test_recon, os.path.join(results_dir, "results", "test_recon"))
    torch.save(test_latent, os.path.join(results_dir, "results", "test_latent"))

    umap_obj = UMAP(random_state=42)
    umap_input = umap_obj.fit(test_input.numpy())
    input_umap = umap_input.transform(test_input.numpy())
    recon_umap = umap_input.transform(test_recon.numpy())

    umap_latent = UMAP(random_state=42)
    latent_umap = umap_latent.fit_transform(test_latent.numpy())

    plotdf = pd.DataFrame(
        {
            "Input_UMAP1": input_umap[:, 0],
            "Input_UMAP2": input_umap[:, 1],
            "Reconstructed_UMAP1": recon_umap[:, 0],
            "Reconstructed_UMAP2": recon_umap[:, 1],
            "Latent_UMAP1": latent_umap[:, 0],
            "Latent_UMAP2": latent_umap[:, 1],
            "cms": test_dataset.label_embedder.inverse_transform(
                test_labels.cpu().numpy()
            ),
        }
    )

    fig2, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
    num_label_type = len(plotdf["cms"].unique())
    current_palette = (
        sns.color_palette("colorblind")
        + sns.color_palette("dark")
        + sns.color_palette("deep")
    )
    current_palette = current_palette[0 : num_label_type + 1]
    current_palette.pop(3)
    sns.scatterplot(
        data=plotdf,
        x="Input_UMAP1",
        y="Input_UMAP2",
        hue="cms",
        ax=ax1,
        palette=current_palette,
        hue_order=["CMS1", "CMS2", "CMS3", "CMS4"],
    )
    sns.scatterplot(
        data=plotdf,
        x="Reconstructed_UMAP1",
        y="Reconstructed_UMAP2",
        hue="cms",
        ax=ax2,
        palette=current_palette,
        hue_order=["CMS1", "CMS2", "CMS3", "CMS4"],
    )
    sns.scatterplot(
        data=plotdf,
        x="Latent_UMAP1",
        y="Latent_UMAP2",
        hue="cms",
        ax=ax3,
        palette=current_palette,
        hue_order=["CMS1", "CMS2", "CMS3", "CMS4"],
    )
    ax1.get_legend().remove()
    ax2.get_legend().remove()
    handles, labels = ax3.get_legend_handles_labels()
    ax3.get_legend().remove()
    fig2.legend(handles, labels, loc="lower center", ncol=4)
    fig2.subplots_adjust(bottom=0.2)
    ax1.set_title(f"Input Data({train_params['input_size']}D)")
    ax2.set_title(f"Reconstructed Data({train_params['input_size']}D)")
    ax3.set_title(f"Latent Code({train_params['latent_size']}D)")
    fig2.suptitle(
        f"UMAP Visualisation of Input and Output Data from {model_name}"
    )
    fig2.subplots_adjust(wspace=0.3, hspace=0.4)
    fig2.savefig(
        os.path.join(results_dir, "results", f"{model_name}_dataviz.png"),
        dpi=300,
        bbox_inches="tight",
    )


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
