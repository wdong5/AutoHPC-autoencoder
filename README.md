# AutoHPC-autoencoder

### Table of Contents
<!--1. [Introduction](#introduction)
1. [Requirements and Dependencies](#requirements-and-dependencies)
1. [Installation](#installation)
1. [Training Dataset](#training-dataset)
1. [Citation](#citation)
1. [Testing Pre-trained Models](#testing-pre-trained-models)
1. [Downloading Results](#downloading-results)
1. [Slow-motion Generation](#slow-motion-generation)
1. [Training New Models](#training-new-models)
1. [Google Colab Demo](#google-colab-demo)-->

### Introduction
<!--We propose the **Smart**-**P**ower **G**rid **sim**ulation (**Smart-PGsim**): Using Neural Network to AccelerateAC-OPF Power Grid Simulation.
Smart-PGsim generates **multitask-learning (MTL)** neural network (NN) models to predict the initial values of variables critical to the problem convergence. MTL models allow information sharing when predicting multiple dependent variables while including customized layers to predict individual variables. We show that, to achieve the required accuracy, it is paramount to embed **domain-specific constraints** derived from the specific power-grid components in the MTL model.  Smart-PGsim then employs the predicted initial values as a high-quality initial condition for the power-grid numerical solver (warm start), resulting in both higher performance compared to state-of-the-art solutions.-->

### Requirements and Dependencies
- Ubuntu (We test with Ubuntu = 16.04.5 LTS)
- Python (We test with Python = 3.7 in Anaconda3 = 4.1.1)
- Cuda & Cudnn (We test with Cuda = 10.0 and Cudnn = 7.0)
- GCC (Compiling PyTorch 1.0.0 extension files (.c/.cu) requires gcc = 4.9.1 and nvcc = 10.0 compilers)
- NVIDIA GPU (We use TESLA V100(Volta) GPUs, but we support compute_50/52/60/61 devices.))

1) install anaconda：
    $ bash Anaconda3-2020.11-Linux-x86_64.sh
2) create conda env:
    $ conda create -n autoHPC python=3.7
3) install tensorflow-gpu
    $ conda install -c anaconda tensorflow-gpu==2.3.0
4) set up environment for autokeras:
    $ pip install git+https://github.com/keras-team/keras-tuner.git
    $ pip install autokeras
5) install library:
    $ pip install matplotlib
    $ pip install sklearn
    $ pip install pandas  
6) set up environment for bayesian optimization
    $ conda install -c conda-forge bayesian-optimization

### Installation
Download repository:
    $ git clone https://github.com/wdong5/AutoHPC-autoencoder.git
    $ cd  AutoHPC-autoencoder
   
