import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='flickr')
    parser.add_argument('--train_sample', type=int, default=1, choices=[0, 1])
    parser.add_argument('--eval_sample', type=int, default=0, choices=[0, 1])
    parser.add_argument('--loss_norm', type=int, default=1, choices=[0, 1])
    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--batch_size', type=int, default=6000)
    parser.add_argument('--sampler', type=str, default='rw', choices=['rw', 'ns', 'rn', 'edge', 'node', 'cluster'])
    parser.add_argument('--gcn_type', type=str, default='sage', choices=['sage', 'gat'])
    parser.add_argument('--use_gpu', type=int, default=1, choices=[0, 1])
    parser.add_argument('--save_log', type=int, default=1, choices=[0, 1])
    parser.add_argument('--save_summary', type=int, default=1, choices=[0, 1])
    parser.add_argument('--log_interval', type=int, default=10)

    args = parser.parse_args()

    return args


def get_log_name(args, prefix='test', use_args=None):
    if use_args is None:
        use_args = ['dataset', 'batch_size', 'train_sample', 'eval_sample', 'loss_norm', 'sampler', 'gcn_type']
    args = vars(args)
    log_name = prefix + '-' + '-'.join([arg + '=' + str(args[arg]) for arg in use_args])
    return log_name
