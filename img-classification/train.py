# Import Relevant Libraries
from src import utils, engine, data_setup
from datetime import date
import os
import sys

import mlflow
import torch
from torch import nn
import torchvision

# Importing Hydra
from omegaconf import DictConfig, OmegaConf
import hydra

# Get Today's Date
today = str(date.today())

# Set the random seeds
utils.set_seeds()


@hydra.main(version_base=None, config_path="conf/", config_name="config")
def train_model(cfg: DictConfig) -> None:
    # Create training and testing DataLoaders as well as get a list of class names
    train_dataloader, test_dataloader, class_names = data_setup.create_dataloaders(
        train_dir=cfg["paths"]["train_dir"],
        test_dir=cfg["paths"]["test_dir"],
        transform=torchvision.models.EfficientNet_B0_Weights.DEFAULT.transforms(),  # perform same data transforms on our own data as the pretrained model
        batch_size=32,
        num_workers=0,
    )

    # Define Model
    weights = (
        torchvision.models.EfficientNet_B0_Weights.DEFAULT
    )  # .DEFAULT = best available weights
    model = torchvision.models.efficientnet_b0(weights=weights).to(
        cfg["params"]["device"]
    )
    # Freeze all base layers in the "features" section of the model (the feature extractor) by setting requires_grad=False
    for param in model.features.parameters():
        param.requires_grad = False

    # Get the length of class_names (one output unit for each class)
    output_shape = len(class_names)

    # Recreate the classifier layer and seed it to the target device
    model.classifier = torch.nn.Sequential(
        torch.nn.Dropout(p=0.2, inplace=True),
        torch.nn.Linear(
            in_features=1280,
            out_features=output_shape,  # same number of output units as our number of classes
            bias=True,
        ),
    ).to(cfg["params"]["device"])

    with mlflow.start_run() as run:
        mlflow.log_param("dataset", cfg['paths']['data_dir'])
        mlflow.log_param("model name", cfg["params"]["model_name"])
        mlflow.log_param("number of classes", cfg["params"]["num_classes"])
        mlflow.log_param("Batch size", cfg["params"]["batch_size"])
        mlflow.log_param("epochs", cfg["params"]["num_epochs"])
        mlflow.log_param("feature extracted", cfg["params"]["feature_extract"])
        mlflow.log_param("pre-trained", cfg["params"]["pre_trained"])
        # Setup the loss function
        loss_fn = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(
            model.parameters(), lr=cfg["params"]["learning_rate"]
        )

        # Train and evaluate
        model, results = engine.train(
            model=model,
            train_dataloader=train_dataloader,
            test_dataloader=test_dataloader,
            optimizer=optimizer,
            loss_fn=loss_fn,
            epochs=cfg["params"]["num_epochs"],
            device=cfg["params"]["device"],
        )
        mlflow.pytorch.log_model(model, "models")
        mlflow.pytorch.save_model(
            model, os.path.join(cfg["paths"]["save_model"], today)
        )


#     mlflow.log_metric('history',hist)


if __name__ == "__main__":
    train_model()
