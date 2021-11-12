import argparse
from random import choices

def get_arguments():
    parser = argparse.ArgumentParser(description = 'Training Arguments')
    parser.add_argument('--gpu_id', default='0', type=int, help='id(s) for CUDA_VISIBLE_DEVICES')
    parser.add_argument('--dataset', default='ADE20K', type=str, choices=['ADE20K'])
    parser.add_argument('--arch', default = 'ResNet50', type=str, choices = ['ResNet50','ResNet101'])
    parser.add_argument('--optimizer', default = 'SGD', type=str, choices = ['SGD','Nesterov','Adam','AdamW'])
    parser.add_argument('--lr', default = 0.01, type=float)
    parser.add_argument('--wd', '--weight_decay','--wdecay', default=0.0001, type=float)
    parser.add_argument('--momentum', default = 0.01, type=float)
    parser.add_argument('--power',default=0.9, type=float)
    parser.add_argument('--epochs', default=500, type=int, help='number of total epochs')
    parser.add_argument('--batch_size', default=16, type=int)
    parser.add_argument('--num_workers',default=16, type=int)
    parser.add_argument('--dilation',default=True, type=bool)
    parser.add_argument('--pretrained',default=True, type=bool)

    parser.add_argument('--resume',default=False, type=bool)
    
    parser.add_argument('--name_opt',default='',type=str)
    
    args = parser.parse_args()
    return args

