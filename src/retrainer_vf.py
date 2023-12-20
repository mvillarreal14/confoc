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

def vggface():
    m = torchvision.models.vgg16()
    m.classifier._modules['6'] = torch.nn.Linear(4096, 2622)
    return m

def main():
    
     # Get arguments: device, initial and final bases
    dev    = int(sys.argv[1])  # gpu
    model  = str(sys.argv[2])  # model
    suffix = str(sys.argv[3])  # model
    path   = str(sys.argv[4])  # path_id
    trn    = str(sys.argv[5])  # trn folder (e.g. comb, comb_00, comb_01, ... comb_08)
    bsize  = int(sys.argv[6])  # batch size
    
    if path == 'vf':
        path = Path('../data/vggface/')
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
    sz=224
    wd=5e-4
    bs=bsize
    aug_tfms = [RandomCrop(224), RandomFlip()]
    #aug_tfms = [RandomCrop(224), RandomFlip(), RandomRotate(20)]
    stats_bgr_with_norm    = A([93.5940/255, 104.7624/255, 129.1863/255], [1.0, 1.0, 1.0])
    stats_rgb_with_norm    = A([129.1863/255, 104.7624/255, 93.5940/255], [1.0, 1.0, 1.0])
    stats_bgr_without_norm = A([93.5940, 104.7624, 129.1863], [1.0, 1.0, 1.0])
    tfms = tfms_from_stats(stats_bgr_without_norm, sz, aug_tfms=aug_tfms, max_zoom=1.2)
    
    # Definition of classes
    with open(path/'names.txt', "r") as file:
        classes = [line.rstrip().lower() for line in file]
        
    # Get data and leaner
    print('Creating learner... ', end='')
    is_rgb=False 
    do_norm=False
    trn_ds_name  = trn
    val_ds_name  = 'valid_01'
    test_ds_name = 'test_crop_224'                                                        
    data = ImageClassifierData.from_paths(path, tfms=tfms, bs=bs, trn_name=trn_ds_name, 
                                      val_name=val_ds_name, test_name=test_ds_name,
                                      classes=classes, is_rgb=is_rgb, do_norm=do_norm)
    learner = Learner.from_model_data(vggface(), data, metrics=[accuracy], crit=nn.CrossEntropyLoss())
    
    # Load model
    learner.load(model)
    print('Done!!!')
    
    # Train
    print('Initializing training:')
    cycles = [5, 5, 5, 6, 6, 6, 7, 7]
    for i, c in enumerate(cycles):
        print(f'  - model version: 0{i}')
        _id = '_0' + str(i)
        m_name = model + suffix + _id
        learner.fit(1e-3, n_cycle=6, cycle_len=1, cycle_mult=2, wds=wd)
        learner.save(m_name)
    
main()
