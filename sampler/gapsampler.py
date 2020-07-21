import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import os.path as osp
from torch_geometric.utils import subgraph, degree, to_dense_adj
import numpy as np


def to_sparsetensor(edge_index, num_nodes):
    row, col = edge_index
    indices = torch.stack([row, col], dim=0)
    values = torch.ones(row.size(0))
    return torch.sparse.LongTensor(indices, values, torch.Size([num_nodes, num_nodes]))


class PartitionClassifier(nn.Module):
    def __init__(self, input_dim, num_parts):
        super(PartitionClassifier).__init__()
        self.input_dim = input_dim
        self.num_parts = num_parts
        self.linear_1 = nn.Linear(input_dim, 64)
        self.ac_1 = nn.ReLU()
        self.linear_2 = nn.Linear(input_dim, 32)
        self.ac_2 = nn.ReLU()
        self.output_fc = nn.Linear(32, num_parts)

    def forward(self, x):
        x = self.linear_1(x)
        x = self.ac_1(x)
        x = self.linear_2(x)
        x = self.ac_2(x)

        return F.log_softmax(x, dim=1)


class GAPSampler(object):
    def __init__(self, data, node_emb, num_parts=25, use_gpu=1, save_dir=None, logging=print):
        self._num_parts = num_parts
        self.logging = logging
        if use_gpu == 1:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            device = torch.device('cpu')
        self._input_dim = node_emb.size(1)
        self._N = data.num_nodes
        self._file_path = osp.join(save_dir or '', self.__filename__)

        if save_dir is not None and osp.exists(self._file_path):
            logging('Load saved gap results from {}.'.format(self._file_path))
        else:
            logging('Train gap graph partition model: ')
            self.__train__(data, node_emb, device, logging)

    @property
    def __filename__(self):
        return f'{self.__class__.__name__.lower()}_{self._num_parts}.npy'

    def __train_step__(self, model, A, optimizer, loader, loss_fn, device):
        model.train()
        total_loss = 0
        for x_batch, nids, d_batch in loader:
            optimizer.zero_grad()
            pred = model(x_batch.to(device))
            A_batch, _ = subgraph(nids, A, num_nodes=self._N)
            A_batch = to_dense_adj(A_batch)
            loss = loss_fn(pred, A_batch, d_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        return total_loss / len(loader)

    @torch.no_grad()
    def __save_results__(self, model, x, device, save_dir):
        # TODO: full data inference
        model.eval()
        out = model(x.to(device))
        pred = out.argmax(dim=-1)
        pred = pred.cpu().numpy()
        np.save(save_dir, pred)

    def __train__(self, data, x, device, logging):

        model = PartitionClassifier(self._input_dim, self._num_parts)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, )

        mean_nodes = data.num_nodes / self._num_parts
        loss_fn = GAPCutLoss(mean_nodes)
        out_degrees = degree(data.edge_index[0], num_nodes=self._N)
        in_degrees = degree(data.edge_index[1], num_nodes=self._N)
        degrees = (in_degrees + out_degrees) / 2
        all_nodes = torch.arange(self._N)
        x = torch.from_numpy(x)
        dataset = TensorDataset(x, all_nodes, degrees)

        data_loader = DataLoader(dataset, batch_size=1024, shuffle=True)
        for epoch in range(30):
            loss = self.__train_step__(model,
                                       data.edge_index,
                                       optimizer,
                                       data_loader,
                                       loss_fn,
                                       device)
            logging(f'Epoch-{epoch:02d}, Loss: {loss:.4f}')
        self.__save_results__(model, x, device, self._file_path)


class GAPCutLoss(nn.Module):
    def __init__(self, mean_nodes):
        super(GAPCutLoss).__init__()
        self.mean_nodes = mean_nodes

    def forward(self, Y, A, degree):
        """
        Upper means mat, lower means vec.
        Args:
            Y(Tensor):
            A:
            degree:

        Returns:

        """

        gamma = torch.mm(Y.t, degree)
        # todo sparse element wise
        error_cut = torch.mm(torch.div(Y, gamma), (1 - Y.T))
        error_cut = torch.mul(error_cut, A).sum()

        error_partition = torch.mm(torch.ones((1, degree.size(0))), Y)
        error_partition = error_partition.squeeze(dim=0)
        error_partition = (error_partition - self.mean_nodes).square().sum()

        return error_cut + error_partition

# class GAPCutLoss(torch.autograd.Function):
#     '''
#     Class for forward and backward pass for the loss function described in https://arxiv.org/abs/1903.00614
#     arguments:
#         Y_ij : Probability that a node i belongs to partition j
#         A : sparse adjecency matrix
#     Returns:
#         Loss : Y/Gamma * (1 - Y)^T dot A
#     '''
#
#     @staticmethod
#     def forward(ctx, Y, A):
#         ctx.save_for_backward(Y, A)
#         D = torch.sparse.sum(A, dim=1).to_dense()
#         Gamma = torch.mm(Y.t(), D.unsqueeze(1))
#         YbyGamma = torch.div(Y, Gamma.t())
#         # print(Gamma)
#         Y_t = (1 - Y).t()
#         loss = torch.tensor([0.], requires_grad=True).to('cuda')
#         idx = A._indices()
#         data = A._values()
#         for i in range(idx.shape[1]):
#             # print(YbyGamma[idx[0,i],:].dtype)
#             # print(Y_t[:,idx[1,i]].dtype)
#             # print(torch.dot(YbyGamma[idx[0, i], :], Y_t[:, idx[1, i]]) * data[i])
#             loss += torch.dot(YbyGamma[idx[0, i], :], Y_t[:, idx[1, i]]) * data[i]
#             # print(loss)
#         # loss = torch.sum(torch.mm(YbyGamma, Y_t) * A)
#         return loss
#
#     @staticmethod
#     def backward(ctx, grad_out):
#         Y, A, = ctx.saved_tensors
#         idx = A._indices()
#         data = A._values()
#         D = torch.sparse.sum(A, dim=1).to_dense()
#         Gamma = torch.mm(Y.t(), D.unsqueeze(1))
#         # print(Gamma.shape)
#         gradient = torch.zeros_like(Y)
#         # print(gradient.shape)
#         for i in range(gradient.shape[0]):
#             for j in range(gradient.shape[1]):
#                 alpha_ind = (idx[0, :] == i).nonzero()
#                 alpha = idx[1, alpha_ind]
#                 A_i_alpha = data[alpha_ind]
#                 temp = A_i_alpha / torch.pow(Gamma[j], 2) * (Gamma[j] * (1 - 2 * Y[alpha, j]) - D[i] * (
#                     Y[i, j] * (1 - Y[alpha, j]) + (1 - Y[i, j]) * (Y[alpha, j])))
#                 gradient[i, j] = torch.sum(temp)
#
#                 l_idx = list(idx.t())
#                 l2 = []
#                 l2_val = []
#                 # [l2.append(mem) for mem in l_idx if((mem[0] != i).item() and (mem[1] != i).item())]
#                 for ptr, mem in enumerate(l_idx):
#                     if ((mem[0] != i).item() and (mem[1] != i).item()):
#                         l2.append(mem)
#                         l2_val.append(data[ptr])
#                 extra_gradient = 0
#                 if l2:
#                     for val, mem in zip(l2_val, l2):
#                         extra_gradient += (-D[i] * torch.sum(
#                             Y[mem[0], j] * (1 - Y[mem[1], j]) / torch.pow(Gamma[j], 2))) * val
#
#                 gradient[i, j] += extra_gradient
#
#         # print(gradient)
#         return gradient, None
