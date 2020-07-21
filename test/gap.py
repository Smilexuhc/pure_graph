import torch
from torch_sparse import SparseTensor
from torch_geometric.utils import to_scipy_sparse_matrix
from sampler.gapsampler import GAPSampler
from sampler.gecsampler import GECData

from utils.utils import load_dataset

dataset = load_dataset('ppi')
data = dataset[0]
node_emb = GECData(data, save_dir=dataset.processed_dir, logging=print).get_node_emb()
sampler = GAPSampler(data, node_emb, save_dir=dataset.processed_dir)
