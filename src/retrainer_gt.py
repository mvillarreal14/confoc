#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 23 00:17:45 2019

@author: mvillarreal
"""
# Add the proper imports
import warnings  
from fastai.sgdr import *
from fastai.plots import *
from fastai.model import *
from fastai.dataset import *
from fastai.imports import *
from fastai.transforms import *
from fastai.conv_learner import *
warnings.simplefilter('ignore')


def main():
    
    # Get arguments: device, initial and final bases
    dev    = int(sys.argv[1])  # gpu
    model  = str(sys.argv[2])  # model
    suffix = str(sys.argv[3])  # model
    path   = str(sys.argv[4])  # path_id
    trn    = str(sys.argv[5])  # trn folder (e.g. comb, comb_00, comb_01, ... comb_08)

    if path == 'gt':
        path = Path('../data/gtsrb/')
    else:
        path = Path(path)
    
    # Check for CUDA. Must return True if there is an working NVidia 
    # GPU set up.
    if torch.cuda.is_available() == False:
        print('Error. CUDA is not available')
        sys.exit(0)
        
    # Check CuDNN, a NVidia package that provides special accelerated 
    # functions for deep learning.
    if torch.backends.cudnn.enabled == False:
        print('Error. CUDA is not available')
        sys.exit(0)
        
    # Set device
    torch.cuda.set_device(dev)
    
    # Enable benchmark mode in cudnn. This way, cudnn will look for the 
    # optimal set of algorithms for that particular configuration.
    torch.backends.cudnn.benchmark=True
    
    # Data augmenatation setting
    arch=resnet34
    sz=96
    wd=5e-4
    bs = 256
    aug_tfms = [RandomRotate(20), RandomLighting(0.8, 0.8)]
    tfms = tfms_from_model(arch, sz, aug_tfms=aug_tfms, max_zoom=1.2)
            
    # Get data and leaner
    trn_ds_name  = trn
    val_ds_name  = 'valid_01'
    test_ds_name = 'test_10each'
    precompute = False                                                        
    data = ImageClassifierData.from_paths(path, tfms=tfms, bs=bs, trn_name=trn_ds_name, 
                                      val_name=val_ds_name, test_name=test_ds_name)
    learner = ConvLearner.pretrained(arch, data, precompute=precompute)

    # Load model
    learner.load(model)
    
    # Train
    learner.fit(1e-1, n_cycle=5, cycle_len=1, cycle_mult=2, wds=wd)
    learner.unfreeze()
    #cycles = [5, 6, 6, 7, 7, 8, 8, 8]
    cycles = [5, 6, 6, 7]
    for i, c in enumerate(cycles):
        print(f'model version: 0{i}')
        _id = '_0' + str(i)
        m_name = model + suffix + _id
        learner.fit(1e-1, n_cycle=c, cycle_len=1, cycle_mult=2, wds=wd)
        learner.save(m_name)

main()
