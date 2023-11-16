# Distributed Training for PyTorch Models with Ray
![image](torchXray.png)
### Project Overview
This project demonstrates the use of the Ray framework to facilitate distributed training of PyTorch models.
The goal is to provide an overview of one approach for tackling large-scale machine learning training by leveraging Ray's distributed computing capabilities.

### Prerequisits
The following requirements should be met before running a distributed training job:
* Ray Cluster is setup: checkout the [official documentation](https://docs.ray.io/en/latest/cluster/getting-started.html) for more info on setting up a cluster
* Access to a multi-GPU environment (with CUDA devices)

### Installation
Before running the project ensure to run:
```
pip install -r requirements.txt
```

### Dataset
We'll be using the built-in MNIST dataset from the torchvision library

### Model
A simple Image classifier with convolutional layers

### Structure
* Components: contains required components for running ray distributed training job


### Configuration
Modify ./components/train_config.json to set the following 
Modify config.json in order to set the following variables:
* Train Loop Config: configuration parameters for running training loop
* Scaling Config: configuration parameters for scaling
* Run Config: configuration parameters for running ray distributed training
* Torch Config: torch backend specific configs

### Usage
To kick off a distributed training job run:
```python3 run_job.py --model ```