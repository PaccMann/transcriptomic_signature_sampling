from pathlib import Path

import numpy as np
import pandas as pd
import torch.nn as nn
from fdsa.models.set_matching.dnn import DNNSetMatching
from sklearn.model_selection import PredefinedSplit
from skorch import NeuralNetClassifier
from skorch.callbacks import EarlyStopping
from skorch.helper import predefined_split


def skorch_mlp_wrapper(params, valid_split):
    net = NeuralNetClassifier(
        DNNSetMatching,
        module__params=params,
        max_epochs=params["epochs"],
        criterion=nn.CrossEntropyLoss(),
        lr=params["lr"],
        # Shuffle training data on each epoch
        iterator_train__shuffle=True,
        train_split=predefined_split(valid_split),
        callbacks=[EarlyStopping(load_best=True)],
    )
    return net
