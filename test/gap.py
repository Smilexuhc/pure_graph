import torch
from torch_sparse import SparseTensor
from torch_geometric.utils import to_scipy_sparse_matrix

torch.st

row, col = edge_index.cpu()
edge_attr = torch.ones(row.size(0))

torch.sparse.LongTensor((edge_attr, (row, col)), (N, N))

