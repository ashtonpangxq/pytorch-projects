# Import Relevant Libraries
from src import utils, engine, data_setup
import os
from pathlib import Path
import sys

import mlflow
from torch import nn

# Set the random seeds
utils.set_seeds()

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
