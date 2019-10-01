# Summary of the code for processing HSI data with capsule network

The code converges well but has some problem when computing the probabilistic map.

## What does each file do

- `caps_model.py` includes the model we use.
- `caps_train.py` loads the data and defines how the model should be trained.
- `HSI_Data_Preparation.py` reads the original .mat data and processes it to patches with specific size. 
- `nidaye.py prepare_data.py seeData.py` are some assistive scripts to help understanding the dataset, codes and bugs.
- `utils.py` contains some utilities. the patch size is defined here.

---

## How did I write the project

The project is based on the github repository [CNN_HSIC_MRF](https://github.com/xiangyongcao/CNN_HSIC_MRF). I modified 
the `cnn_train.py` to `caps_train.py`. So they're almost the same. I wrote the caps_model.py to store the caps model. I
modified the [notebook file](https://github.com/ageron/handson-ml/blob/master/extra_capsnets-cn.ipynb) to create the 
CapsNet model. The `HSI_Data_Preparation.py` and `utils.py` are directly copied from the repository. 

--- 

## How does the project work

The `caps_train.py` is the main file of this project. It calls `the HSI_Data_Preparation.py` to prepare the data and 
store the patches into tow dictionaries and feed them to the CapsNet model defined in `caps_model.py.` Then the model 
will be trained with specific hyper parameters. After training, the model is saved to the directory you specified. The 
`caps_train.py` uses the trained model to calculate the prob map. The prob map will be stored to a specific directory 
(Currently it will be saved to the file `./prob_map.mat`. You may sooner be able to specify the directory you want to 
save the prob map. But I haven't written this function because it's not very urgent.) for later process. (For example, 
feed it into a Markov Random Field to smooth the segmented graph.)

---

## Structure of the capsule network

I defined three CapsNet models in the `caps_model.py`. 

The first one is similar to the one Hiton used in his paper 
[Dynamic Routing Between Capsules](https://arxiv.org/abs/1710.09829). One convolution layer, one primary layer and one
output layer. The capsule layer uses 512 3*3 filters to extract feature maps. The primary layer accept the output of 
the convolution layer and feed it to 288 capsules with dimension of [8]. The output layer contains 16 capsules with 
dimension of [num_classes]. The primary layer and the output layer deployed the dynamic routing.

The second one has two more convolution layers. The third one has one more capsule layer than the second one. 

--- 

## How to run the project

### Environment

- `python` I use 3.5, but 3.6 or 3.7 should be OK. 3.4 or lower python 3 is not recommended because i don't know 
whether they will cause errors or not. Python 2 isn't compatible.

- `tensorflow` 1.13.1 is recommended but other versions should be OK.

- `numpy` I use 1.16.4, other versions should be OK.

- `pygco` Use the latest version is OK. The team(or person) stopped upgrade the library. I have some problem 
installing it into Windows. So I recommend you to install it into a Linux-based operating system. 

- `scikit-learn` 0.21.2 is recommended but other versions should be OK.

- `scikit-image` 0.15.0 is recommended but other versions should be OK.

- `pandas` 0.24.2 is recommended but other versions should be OK.

- `pillow` 6.1.0 is recommended but other versions should be OK.

### Run

1. Clone or download the repository from [github](https://github.com/osmium18452/CAPSNET_HSI)
`git clone git@github.com:osmium18452/CAPSNET_HSI.git`

2. Run `python caps_train.py` to run with default settings. Run `python caps_train --help` for more options.

---

## Current issues

1. The model won't converge anymore when the cross entropy loss decrease to 2.6 or when the margin loss decrease to 
0.08(Though it looks small but the possibility it calculates for each class is very close. See `100data.txt` for the 
result it calculated.).

2. Sometimes it the model can't be trained on a GPU but sometimes it can. On CPU this situation won't happen. But it 
doesn't matter because it seems training on GPU (RTX 2080Ti) is not faster than on CPU (Dual Xeon E5-2678 with 64 GiB).

---

## Presumed reason

No obvious reason has been found. Personally I think the CapsNet model is incompatible with the multi-channel dataset. 
I find the CapsNet I wrote converges quickly when training with single-channel dataset, such as MNIST, SmallNORB. But 
when training on multi-channel graph dataset, it will stop converging around a specific point. The model I wrote may 
have some defects. But it converged well on MNIST. Some time earlier i build another CapsNet model with the 
[CapsLayer](https://github.com/naturomics/CapsLayer) on github. But it stopped converging when the cross entropy loss 
decreased to 2.5, just the same as the one in this repository.
