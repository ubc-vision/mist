# Partly from https://raw.githubusercontent.com/vcg-uvic/lf-net-release/master/common/argparse_utils.py

import json
import argparse
import sys

def str2bool(v):
    return v.lower() in ('true', '1', 'yes', 'y', 't')

def get_mist_config():
    parser = argparse.ArgumentParser()
    ## --- General Settings
    # Json Path
    parser.add_argument('--path_json', type=str, default='')
    # Use Seed or Not
    parser.add_argument('--set_seed', type=str2bool, default=True)
    # Run name and comment
    parser.add_argument('--name', type=str, default='test')
    parser.add_argument('--comment', type=str, default='')
    # Dataset base path
    parser.add_argument('--dataset_dir', type=str, default='/home/yjin/datasets')
    # Model base path
    parser.add_argument('--model_dir', type=str, default='./pretrained_models')
    # Choose dataset
    parser.add_argument('--dataset', type=str, default='mnist_hard')
    # Number of Epoch
    parser.add_argument('--epochs', type=int, default=2000)
    # Batch Size
    parser.add_argument('--batch_size', type=int, default=32)
    # Save Weight
    parser.add_argument('--save_weights',type=str2bool, default=False)
    # Resume
    parser.add_argument('--resume',type=str2bool, default=False)
    # Validation
    parser.add_argument('--run_val',type=str2bool, default=True)
    parser.add_argument('--val_path', type=str, default='./val_results')
    # Test
    parser.add_argument('--test_path', type=str, default='./test_results')
    # Data argumentation (only works on PASCAL dataset)
    parser.add_argument('--rand_horiz_flip', type=str2bool, default=False)
    parser.add_argument('--rand_maskout', type=str2bool, default=False)
    # Pretrained model path
    parser.add_argument('--pretrained_resnet',type=str2bool,default= True)
    # Tensorboard 
    parser.add_argument('--summary_period', type=float, default=1)
    parser.add_argument('--save_period', type=float, default=5)
    parser.add_argument('--valid_period', type=float, default=5)
    parser.add_argument('--cooldown_period', type=float, default=-1)
    parser.add_argument('--log_dir', type=str, default='logs/')

    ## --- Training Settings
    # Detector Learning Rate
    parser.add_argument('--lr_detector', type=float, default=1e-4)
    # Key Points Learning Rate and Iteration
    parser.add_argument('--k_iter', type=int, default=2)
    parser.add_argument('--lr_k', type=float, default=1e4)    
    # Classifier Learning Rate
    parser.add_argument('--lr_task', type=float, default=1e-4)
    # Reconstruction method (gaussian, single_point)
    parser.add_argument('--heatmap_reconstruct',type=str,default='single_point')
    ## Task network loss function
    parser.add_argument('--loss_type', type=str, default='MSE') 

    ## --- Network Settings 
    # Num of classes
    parser.add_argument('--num_classes', type=int, default=10)
    # Image size
    parser.add_argument('--image_size', type=int, default=80)

    ## --- Detector Settings 
    # Detector Backbone VGG16 or ResNet
    parser.add_argument('--detector_backbone',type=str,default='CustomResNet')
    # Softmax kernal size to heatmap ratio 
    parser.add_argument('--sm_kernal_size_ratio',  type=float, default=0.2)
    # NMS kernal size to heatmap ratio 
    parser.add_argument('--nms_kernal_size_ratio',  type=float, default=0.05)
    # Number of Key Points
    parser.add_argument('--k', type=int, default=9)
    # Spatial Softmax
    parser.add_argument('--spatial_softmax', type=str2bool, default=True) 
    parser.add_argument('--softmax_strength', type=float, default=10)
    # Sub pixel accuracy
    parser.add_argument('--sub_pixel_kp', type=str2bool, default=False)
    ## Bbox size
    parser.add_argument('--anchor_size', type=float, default=0.25)

    ## --- Classifier Settings 
    # Patch extraction
    parser.add_argument('--patch_from_featuremap', type=str2bool, default=False)
    # Patch Size
    parser.add_argument('--patch_size', type=int, default=32)        

    config = parser.parse_args()

    # overwrite with configs in json
    if config.path_json != '':
        with open(config.path_json) as f:
            params = json.load(f)
        for key,value in params.items():
            setattr(config,key,value) 

    return config

def print_config(config):
    print('---------------------- CONFIG ----------------------')
    print()
    args = list(vars(config))
    args.sort()
    for arg in args:
        print(arg.rjust(25,' ') + '  ' + str(getattr(config, arg)))
    print()
    print('----------------------------------------------------')

def config_to_string(config):
    string = '\n\n'
    string += 'python ' + ' '.join(sys.argv)
    string += '\n\n'
    # string += '---------------------- CONFIG ----------------------\n'
    args = list(vars(config))
    args.sort()
    for arg in args:
        string += arg.rjust(25,' ') + '  ' + str(getattr(config, arg)) + '\n\n'
    # string += '----------------------------------------------------\n'
    return string
