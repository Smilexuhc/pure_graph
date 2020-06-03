import os.path as osp
from .parse_args import parse_args
import torch
import torch.nn.functional as F
from torch_geometric.datasets import Flickr
from torch_geometric.data import GraphSAINTRandomWalkSampler
from torch_geometric.nn import SAGEConv
from torch_geometric.utils import degree
import numpy as np


class Net(torch.nn.Module):
    def __init__(self, hidden_channels):
        super(Net, self).__init__()
        in_channels = dataset.num_node_features
        out_channels = dataset.num_classes
        self.conv1 = SAGEConv(in_channels, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, hidden_channels)
        self.conv3 = SAGEConv(hidden_channels, hidden_channels)
        self.lin = torch.nn.Linear(3 * hidden_channels, out_channels)

    def set_aggr(self, aggr):
        self.conv1.aggr = aggr
        self.conv2.aggr = aggr
        self.conv3.aggr = aggr

    def forward(self, x0, edge_index, edge_weight=None):
        x1 = F.relu(self.conv1(x0, edge_index, edge_weight))
        x1 = F.dropout(x1, p=0.2, training=self.training)
        x2 = F.relu(self.conv2(x1, edge_index, edge_weight))
        x2 = F.dropout(x2, p=0.2, training=self.training)
        x3 = F.relu(self.conv3(x2, edge_index, edge_weight))
        x3 = F.dropout(x3, p=0.2, training=self.training)
        x = torch.cat([x1, x2, x3], dim=-1)
        x = self.lin(x)
        return x.log_softmax(dim=-1)


def train_sample(norm_loss):
    model.train()
    model.set_aggr('add')

    total_loss = total_examples = 0
    for data in loader:
        data = data.to(device)
        optimizer.zero_grad()
        out = model(data.x, data.edge_index, data.edge_norm * data.edge_attr)
        if norm_loss ==1:
            loss = F.nll_loss(out, data.y, reduction='none')
            loss = (loss * data.node_norm)[data.train_mask].sum()
        else:
            loss = F.nll_loss(out, data.y, reduction='none')[data.train_mask].sum()
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * data.num_nodes
        total_examples += data.num_nodes
    return total_loss / total_examples


def train_full():
    model.train()
    model.set_aggr('mean')

    optimizer.zero_grad()
    out = model(data.x.to(device), data.edge_index.to(device))
    loss = F.nll_loss(out[data.train_mask], data.y.to(device)[data.train_mask])
    loss.backward()
    optimizer.step()
    return loss.item()


@torch.no_grad()
def eval_full():
    model.eval()
    model.set_aggr('mean')

    out = model(data.x.to(device), data.edge_index.to(device))
    pred = out.argmax(dim=-1)
    correct = pred.eq(data.y.to(device))

    accs = []
    for _, mask in data('train_mask', 'val_mask', 'test_mask'):
        accs.append(correct[mask].sum().item() / mask.sum().item())
    return accs


@torch.no_grad()
def eval_sample():
    model.eval()
    model.set_aggr('add')
    accs_all = []
    for data in loader:
        data = data.to(device)
        out = model(data.x, data.edge_index, data.edge_norm * data.edge_attr)
        pred = out.argmax(dim=-1)
        accs_batch = []

        correct = pred.eq(data.y.to(device))

        for _, mask in data('train_mask', 'val_mask', 'test_mask'):
            accs_batch.append(correct[mask].sum().item() / mask.sum().item())
        accs_all.append(accs_batch)
    accs = []
    for i in range(3):
        accs.append(np.mean(accs_all[:,i]))


if __name__ == '__main__':
    args = parse_args()

    if args.dataset == 'Flickr':
        path = osp.join(osp.dirname(osp.realpath(__file__)), 'data', 'Flickr')
        dataset = Flickr(path)
    else:
        raise KeyError('Dataset name error')

    data = dataset[0]
    row, col = data.edge_index
    data.edge_attr = 1. / degree(col, data.num_nodes)[col]  # Norm by in-degree.

    if args.sampler == 'rw':
        loader = GraphSAINTRandomWalkSampler(data, batch_size=args.batch_size, walk_length=2,
                                             num_steps=5, sample_coverage=1000,
                                             save_dir=dataset.processed_dir,
                                             num_workers=0)
    else:
        raise KeyError('Sampler type error')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = Net(hidden_channels=256).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)


    summary_all = []
    for epoch in range(1, 51):
        if args.train_sample == 1:
            loss = train_sample(norm_loss=args.norm_loss)
        else:
            loss = train_full()
        if args.eval_sample == 1:
            accs = eval_sample()
        else:
            accs = eval_full()
        print(f'Epoch: {epoch:02d}, Loss: {loss:.4f}, Train: {accs[0]:.4f}, '
              f'Val: {accs[1]:.4f}, Test: {accs[2]:.4f}')
        summary_all.append(accs[2])
    print('ALL epoch acc:',summary_all)



