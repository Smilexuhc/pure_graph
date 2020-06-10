import os.path as osp
from torch_geometric.datasets import Flickr,Reddit,PPI,Yelp
from torch_geometric.data import GraphSAINTRandomWalkSampler, \
    NeighborSampler, GraphSAINTNodeSampler, GraphSAINTEdgeSampler
from sampler import GraphSAINTNodeSampler, GraphSAINTEdgeSampler, MySAINTSampler
import torch.nn as nn
from metric_and_loss import NormCrossEntropyLoss,NormBCEWithLogitsLoss


def load_dataset(dataset='flickr'):
    """

    Args:
        dataset: str, name of dataset, assuming the raw dataset path is ./data/your_dataset/raw.
                 torch_geometric.dataset will automatically preprocess the raw files and save preprocess dataset into
                 ./data/your_dataset/preprocess

    Returns:
        dataset
    """
    path = osp.join(osp.dirname(osp.realpath(__file__)), 'data', dataset)
    if dataset == 'flickr':
        dataset = Flickr(path)

    elif dataset == 'reddit':
        dataset = Reddit(path)

    elif dataset == 'ppi':
        dataset= PPI(path)

    elif dataset == 'yelp':
        dataset = Yelp(path)

    else:
        raise KeyError('Dataset name error')

    return dataset

def build_loss_op(args):
    if args.dataset in ['flickr','reddit']:
        if args.loss_norm == 1:
            return NormCrossEntropyLoss()
        else:
            return nn.CrossEntropyLoss(reduction='none')
    else:
        if args.loss_norm == 1:
            return NormBCEWithLogitsLoss()
        else:
            return nn.BCEWithLogitsLoss(reduction='none')




