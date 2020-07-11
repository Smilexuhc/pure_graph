import argparse
import yaml
from easydict import EasyDict


def parse_config(config_path):
    with open(config_path) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    return config


def parse_args(config_path):
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='flickr')
    parser.add_argument('--train_sample', type=int, default=1, choices=[0, 1])
    parser.add_argument('--eval_sample', type=int, default=0, choices=[0, 1])
    parser.add_argument('--loss_norm', type=int, default=1, choices=[0, 1])
    parser.add_argument('--epochs', type=int)
    parser.add_argument('--batch_size', type=int)
    parser.add_argument('--sampler', type=str, default='rw',
                        choices=['rw-my', 'rw', 'ns', 'node-my', 'edge', 'node', 'cluster', 'gec'])
    parser.add_argument('--gcn_type', type=str, default='sage', choices=['sage', 'gat'])
    parser.add_argument('--use_gpu', type=int, default=1, choices=[0, 1])
    parser.add_argument('--save_log', type=int, default=1, choices=[0, 1])
    parser.add_argument('--save_summary', type=int, default=1, choices=[0, 1])
    parser.add_argument('--log_interval', type=int, default=10)
    parser.add_argument('--self_loop', type=int, default=1, choices=[0, 1])
    parser.add_argument('--num_heads', type=int, default=1)
    parser.add_argument('--num_parts', type=int, default=100)
    parser.add_argument('--hidden_units', type=int, default=256)
    parser.add_argument('--num_clusters', type=int, default=50)
    parser.add_argument('--cluster_type', type=str)
    parser.add_argument('--walk_length', type=int, default=0)

    args = parser.parse_args()
    if args.train_sample == 0:
        args.loss_norm = 0
    if args.sampler == 'cluster':
        args.loss_norm = 0

    args = vars(args)

    config = parse_config(config_path)
    config = config[args['dataset']]
    config = EasyDict(config)

    if not config:
        raise ValueError('Please check default hparams file!')

    for k, v in args.items():
        if v is not None:
            config[k] = v

    return config


def get_log_name(args, prefix='test', use_args=None):
    if use_args is None:
        use_args = ['dataset', 'batch_size', 'train_sample', 'eval_sample', 'loss_norm', 'sampler', 'gcn_type']
    args = vars(args)
    log_name = prefix + '-' + '-'.join([arg + '=' + str(args[arg]) for arg in use_args])
    return log_name
