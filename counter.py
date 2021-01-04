import argparse
import os
from importlib import import_module
import sys

def count_parameters(net):
    params = sum([param.nelement() for param in net.parameters() if param.requires_grad])
    print("Params: %f M" % (params/1000000))

def parse_args():
    parser = argparse.ArgumentParser(description='Pytorch-models Args')
    parser.add_argument('--dataset', type=str, choices=['cifar10','cifar100','imagenet'], required=True, help='choose a dataset to learn')
    parser.add_argument('--model', type=str, required=True, help='choose a network')
    args = parser.parse_args()
    return args

if __name__ == '__main__':

    args = parse_args()
    model_path = 'models.'+ args.dataset
    net = getattr(import_module(model_path), args.model)
    count_parameters(net())




