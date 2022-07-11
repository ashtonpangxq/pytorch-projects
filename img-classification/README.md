# Image Classification with PyTorch
This repository contains implementation of Image Classification model with [PyTorch](https://pytorch.org/) and Experiment Tracking with [MLFlow](https://mlflow.org/).

Datasets can be trained with `VGG, SqueezeNet, DenseNet, ResNet, AlexNet, Inception` for this particular project.

## Config File
The config file is a json file containing all the parameters and paths required for training. List of parameters are listed below:
* **model_name** is the name of the model you wish to use and must be selected from this list: [resnet, alexnet, vgg, squeezenet, densenet, inception].
* **num_classes** is the number of classes in the dataset.
* **batch_size** is the batch size used for training and may be adjusted accordingly.
* **num_epochs** is the number of training epochs we want to run for training our model.
* **feature_extract** is a boolean that defines if we are finetuning or feature extracting.
    * If feature_extract = False, the model is finetuned and all model parameters are updated. 
    * If feature_extract = True, only the last layer parameters are updated, the others remain fixed.

```
{
  "model_name":"<squeezenet/resnet/vgg16/densenet/alexnet/inception>", 
  "num_classes": "<number of classes>",
  "batch_size": "32",
  "num_epochs": "<number of epoches>",
  "feature_extract": "False",
  "pre_trained": "True",
  "save_model": "<path to save model>",
  "save_confusion_mat": "/exp/data/densenet_17_mat.csv",
  "data_dir":"<path of the dataset>",
  "classes": [<list of class names>]
}
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