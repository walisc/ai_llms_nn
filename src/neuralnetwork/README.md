Deep Neural Network

This pack contains the implementaion of a neural network using various approaches, namely 


1. Vanilla Python (standard library only)
2. Python + NumPy
3. PyTorch
4. TensorFlow

Setup

Before running you will need to run `poerty install`

Please note: if you will be using CUDA (Nvida GPU) for training with PyTorch or Tensorflow, the current configuration in using CUDA 12.6. If you want to use a newer version you are welcome update this source, and install as appropitate. e.g for CUDA 13.0

Pytourch

poetry source add --priority=explicit pytorch-gpu-src-cul130 https://download.pytorch.org/whl/cu130

poerty add --source pytorch-gpu-src-cul130 <TODO>

TensorFlow

<TODO>

Running

PLease note: 
More effience way to load data