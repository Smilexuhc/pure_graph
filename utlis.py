import os.path as osp
from torch_geometric.datasets import Flickr,Reddit,PPI,Yelp


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



