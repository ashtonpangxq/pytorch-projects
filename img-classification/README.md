# Image Classification with PyTorch
This repository contains implementation of Image Classification model with [PyTorch](https://pytorch.org/) and Experiment Tracking with [MLFlow](https://mlflow.org/).

Datasets can be trained with `VGG, SqueezeNet, DenseNet, ResNet, AlexNet, Inception` for this particular project.

## Set Up Instructions
```sh
conda create -n img-classification python=3.8
conda activate img-classification
pip install -r requirements.txt
```

## Config File
The config file is a .yaml file containing all the parameters and paths required for training. List of parameters are listed below:
* **model_name**: Name of Model
* **num_epochs**: Number of Training Epochs to train model.
* **batch_size**: The batch size used for training and may be adjusted accordingly.
* **hidden_units**: The number of hidden units for the Model.
* **learning_rate**: The learning rate for Model Optimizer. 
* **num_classes**: Number of classes in the dataset.
* **feature_extract** is a boolean that defines if we are finetuning or feature extracting.
    * If feature_extract = False, the model is finetuned and all model parameters are updated. 
    * If feature_extract = True, only the last layer parameters are updated, the others remain fixed.
* **pre_trained**: If using pre-trained parameters for training.
* **device**: device to use for training. (cuda or cpu)

```
params:
  model_name: "EfficientNetB2"
  num_epochs: 10
  batch_size: 32
  hidden_units: 10
  learning_rate: 0.001
  num_classes: 3
  feature_extract: True
  pre_trained: True
  device: "cuda"
  classes:
    - pizza
    - steak
    - sushi
paths:
  save_model: "exp/data/models/"
  save_confusion_mat: "exp/data/efficientnetb2_mat.csv"
  data_dir: "data/"
  train_dir: "data/pizza_steak_sushi/train"
  test_dir: "data/pizza_steak_sushi/test"
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