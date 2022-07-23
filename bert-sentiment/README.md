# Image Classification with PyTorch
This repository contains implementation of Sentiment Analysis model with [PyTorch](https://pytorch.org/) and Experiment Tracking with [MLFlow](https://mlflow.org/).

Dataset is trained with `BERT` Model.

## Set Up Instructions
```sh
conda create -n bert-sentiment python=3.8
conda activate bert-sentiment
pip install -r requirements.txt
```

## Config File
The config file is a json file containing all the parameters and paths required for training. List of parameters are listed below:
* **num_epochs**: Number of Training Epochs to train model.
* **batch_size**: The batch size used for training and may be adjusted accordingly.
* **hidden_units**: The number of hidden units for the Model.
* **learning_rate**: The learning rate for Model Optimizer. 
* **device**: device to use for training. (cuda or cpu)

```
params:
  num_epochs: 10
  batch_size: 32
  learning_rate: 0.001
  device: "cuda"
```

## Training steps 

- edit the `config_files/training.json` 
- run `python main.py --config training`
- after finshing the training it will create a confusion matrix 
- the notebook will also save trained model with `pytorch` and `mlflow`  
    
## MLflow 

`MLflow` helps in tracking experiments, packaging code into reproducible runs, and sharing and deploying models.
I have used `MLflow` to track my experiments and save parameters used for a particular training. We tracked 7 parameters in this case which can be seen later.

- Install MLflow from PyPI via ```pip install mlflow```
- The MLflow Tracking UI will show runs logged in `./mlruns` at [http://localhost:5000](http://localhost:5000). Start it with: `mlflow ui`

## Run as Docker Container 

- `sudo docker build -t classification:0.1 .`
- `sudo docker run -it --rm -p 5000:5000 -v <dataset path>:/code/data/ classification:0.1`
- `cd code`
- `python main.py --config training`
- `mlflow server --host=0.0.0.0`
- `http://localhost:5000/`