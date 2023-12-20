#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This program generates styled images provided a list of images and style bases.

@author: mvillarreal
"""

# Imports 
from igen import *
import sys

def main():

    # Get arguments: device, initial and final bases
    data = str(sys.argv[1])  # Either gt or vf
    mode = int(sys.argv[2])  # Mode (1- one folder or 2-structure)
    src  = str(sys.argv[3])  # Source folder (e.g. valid, heal, etc.)
    dev  = int(sys.argv[4])  # GPU
    rbeg = int(sys.argv[5])  # Initial directory or image (if mdde == 1) in folder 
    rend = int(sys.argv[6])  # Final directory or image (if model == 1) in folder

    # Check whether to generate content or styled version
    if len(sys.argv) == 7:
        is_content = True
    elif len(sys.argv) == 9:
        bbeg = int(sys.argv[7])  # Initial base
        bend = int(sys.argv[8])  # Final base
        is_content = False
    else:
        print('Error 1: Incorrect number of arguments.')
        sys.exit()
    
    # Validating mode
    is_one_folder = True
    if mode == 1:
        is_ond_folder = True
    elif mode == 2:
        is_one_folder = False
    else:
        print('Error 2: mode {mode} invalid.')
        sys.exit()

    # Processing source path
    #if path[-1] == '/': path = path[:-1]
    #dir_src = path.split('/')[-1]
    #path = Path(path)
    if data == 'gt':
    	path = Path('../data/gtsrb/')
    else:
        path = Path('../data/vggface/')
    dir_src  = src
    path_src = path/dir_src  
    
    # Select device and enable benchmark setting
    torch.cuda.set_device(dev)
    torch.backends.cudnn.benchmark=True

    # Define model
    m_vgg = to_gpu(vgg16(True)).eval()
    set_trainable(m_vgg, False)

    # Define image transformations using the model
    sz = 288 if data == 'gt' else 224
    trn_tfms,val_tfms = tfms_from_model(vgg16, sz)

    # Get desired output layers
    layers = [i-1 for i,o in enumerate(children(m_vgg)) if isinstance(o,nn.MaxPool2d)]

    # Index of feature layer for content
    layer_idx = 2

    # Required for Step function
    max_iter = 1000
    show_iter = 100

    # Required Optimizer (type string so far)
    opt_alg = None

    # Require printing
    verbose = False

    # Create generator object
    sig = StyledImageGenerator(None, None, m_vgg, val_tfms, layers, layer_idx , 
                               max_iter, show_iter, opt_alg,verbose=verbose)
    
    # Set base path and get list of bases 
    if is_content: 
        bases = [0]
    else:
        path_bases = path/'bases/'
        bases = os.listdir(path_bases )
        bases = sorted([b for b in bases if '.jpg' in b or '.png' in b])
        bases = bases[(bbeg-1):bend]    

    # Get range of directories when needed
    if is_one_folder: 
        dirs = [0]
    else:
        dirs = os.listdir(path_src)
        if data == 'gt':
            dirs = sorted ([d for d in dirs if d != '.DS_Store'], key=int)
        else:
            dirs = sorted ([d for d in dirs if d != '.DS_Store'])
        dirs = dirs[(rbeg-1):rend]
 
    # Do the process for each base
    for b in bases:
                
        # Get dest path by adding the based id to src folder and set set style to generator when needed
        if is_content:
            dir_dst = dir_src + '_00'
        else:
            sig.set_s(path_bases/b)
            suffix = b[5:7]
            dir_dst = dir_src + '_' + suffix
        path_dst = path/dir_dst
        os.makedirs(path_dst, exist_ok=True)
    
        # To track progress
        item1 = 'Content images' if is_content else 'Styled images with base ' + b 
        item2 = 'images' if is_one_folder  else 'directories'
        print(f'{item1} ({item2} {rbeg}-{rend})')
                
        # Crete image contents
        for i, d in enumerate(dirs):
    
            # Update path_src and path_dst when needed and get filenames
            path_src_final = path_src
            path_dst_final = path_dst
            if is_one_folder:
                filenames = sorted(os.listdir(path_src_final))
                filenames = [f for f in filenames if ('.png' in f or '.jpg' in f)]
                filenames = filenames[(rbeg-1):rend]
            else:
                path_src_final = path_src/d
                path_dst_final = path_dst/d
                filenames = os.listdir(path_src_final)
                filenames = [f for f in filenames if ('.png' in f or '.jpg' in f)]
                os.makedirs(path_dst_final, exist_ok=True)
                
            # Iterate over source files
            for j, f in enumerate(filenames):

                # Get new name filename and check if it already exists 
                parts = f.split('.')
                new_f = f[:-4]+'_00'+f[-4:] if is_content else f[:-4]+'_'+suffix+f[-4:]
                #new_f = parts[0]+'_00.'+parts[1] if is_content else parts[0]+'_'+ suffix+'.'+parts[1]
                exists = os.path.isfile(path_dst_final/new_f)

	        # Get styled version if does not exist
                if not exists:

                    # Get x and set it in generator
                    x = path_src_final/f
                    sig.set_x(x)
                                        
                    # Set loss scale and get image
                    if is_content:
                        sig.set_content_loss_scale(1e3)
                        new_x  = sig.get_content()
                    else:
                        sig.set_content_loss_scale(1e6)
                        #sig.set_loss_scale_automatically()
                        new_x = sig.get_styled_image()
        
                    # Save image
                    print(new_f)
                    plt.imsave(str(path_dst_final/new_f), new_x)    

main()


