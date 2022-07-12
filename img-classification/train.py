# Import Relevant Libraries
from src import utils, engine, data_setup
import os
from pathlib import Path
import sys
import argparse

import mlflow
from torch import nn
from torchvision import transforms

# Set the random seeds
utils.set_seeds()

# Create a parser
parser = argparse.ArgumentParser(description="Get some hyperparameters to train model.")

# Get an arg for num_epochs
parser.add_argument(
    "--num_epochs", default=10, type=int, help="the number of epochs to train for"
)

# Get an arg for batch_size
parser.add_argument(
    "--batch_size", default=32, type=int, help="number of samples per batch"
)

# Get an arg for hidden_units
parser.add_argument(
    "--hidden_units",
    default=10,
    type=int,
    help="number of hidden units in hidden layers",
)

# Get an arg for learning_rate
parser.add_argument(
    "--learning_rate", default=0.001, type=float, help="learning rate to use for model",
)

# Create an arg for training directory
parser.add_argument(
    "--train_dir",
    default="data/pizza_steak_sushi/train",
    type=str,
    help="directory file path to training data in standard image classification format",
)

# Create an arg for test directory
parser.add_argument(
    "--test_dir",
    default="data/pizza_steak_sushi/test",
    type=str,
    help="directory file path to testing data in standard image classification format",
)

# Get our arguments from the parser
args = parser.parse_args()

# Setup hyperparameters
NUM_EPOCHS = args.num_epochs
BATCH_SIZE = args.batch_size
HIDDEN_UNITS = args.hidden_units
LEARNING_RATE = args.learning_rate

data_dir = os.path.join(sys.path[0], "..", "data", "pizza_steak_sushi")

with mlflow.start_run() as run:
    mlflow.log_param("dataset", data_dir)
    mlflow.log_param("model name", model_name)
    mlflow.log_param("number of classes", num_classes)
    mlflow.log_param("Batch size", batch_size)
    mlflow.log_param("epochs", num_epochs)
    mlflow.log_param("feature extracted", feature_extract)
    mlflow.log_param("pre-trained", pre_trained)
    # Setup the loss function
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Train and evaluate
    model, results = train(
        model=model,
        train_dataloader=train_dataloader,
        test_dataloader=test_dataloader,
        optimizer=optimizer,
        loss_fn=loss_fn,
        epochs=num_epochs,
        device=device,
    )
    mlflow.pytorch.log_model(model, "models")
    mlflow.pytorch.save_model(model, os.path.join(save_model, today))

#     mlflow.log_metric('history',hist)
