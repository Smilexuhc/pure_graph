import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='flickr')
    parser.add_argument('--train_sample', type=int, default=1)
    parser.add_argument('--eval_sample', type=int, default=0)
    parser.add_argument('--loss_norm', type=int, default=1)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--sampler', type=str, default='rw')

    args = parser.parse_args()
    print('Model setting:', args)
    return args
