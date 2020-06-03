import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='flickr')
    parser.add_argument('--train_sample', type=int, default=1,choices=[0,1])
    parser.add_argument('--eval_sample', type=int, default=0,choices=[0,1])
    parser.add_argument('--loss_norm', type=int, default=1,choices=[0,1])
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--sampler', type=str, default='rw',choices=['rw','ns','rn'])
    parser.add_argument('--gcn_type',type=str, default='sage',choices=['sage','gat'])

    args = parser.parse_args()

    return args
