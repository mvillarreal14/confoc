# ConFoc: Content-Focus Protection Against Trojan Attacks on Neural Networks
ConFoc is a model-hardening technique to protect Deep Neural Networks (DNNs)against trojan or backdoor attacks. Its results were presented in the paper  "ConFoc: Content-Focus Protection Against Trojan Attacks on Neural Networks". This repo contains:
* The utilized version of fast.ai deep learning library.
* The source code to generate styled images and train/retrain models. 
* Examples in notebooks on these tasks, which show how to run ConFoc. 
* The GTRSB and VGGFACE datasets used in our experiments. Notice that we include two examples for each dataset of the resulting styled images in the folder data. These examples are sub-folders with the suffices "_00" and "_01".

  
## To install
Follow steps in fastai folder as indicated by their authors. You might have to unzip the file fastai.zip within the directory fastai.

## A list of technologies used within the project
* Python version: 3.6.1
* PyTorch version: 0.3.1
* CUDA version   : 9.0.176
* CuDNN version: 7005

## To run experiments
1. Prepare your data as needed. Please review the examples to get a clear idea on how to complete this step.
2. Use the provided notebook to train your model on your data and test it accordingly.
3. Source code allows users to run experiments from the command line.

## Paper Citations
To be added. Expected to be published in the Transactions of Dependable and Secure Computing (TDSC) journal.