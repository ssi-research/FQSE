import argparse
import torch
import train_env.asteroid_librimix.asteroid_librimix_trainer as asteroid_librimix_trainer
from utils import get_device
DEVICE = get_device()


def argument_handler():
    parser = argparse.ArgumentParser()
    #####################################################################
    # General Config
    #####################################################################
    parser.add_argument('--yml_path', '-y', type=str, required=True, help='YML configuration file')
    parser.add_argument('--use_cpu', action="store_true", help='Use cpu')

    args = parser.parse_args()
    return args


def train():

    # ------------------------------------
    # Read args
    # ------------------------------------
    args = argument_handler()
    device = "cpu" if args.use_cpu or not torch.cuda.is_available() else 'cuda'

    # ------------------------------------
    # Run training
    # ------------------------------------
    asteroid_librimix_trainer.train(args.yml_path, device)

    print("Training is done!")

if __name__ == '__main__':
    train()















