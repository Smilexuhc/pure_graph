import os.path as osp
from parse_args import parse_args, get_log_name
import torch
import torch.nn.functional as F
from torch_geometric.datasets import Flickr, Reddit
from torch_geometric.data import GraphSAINTRandomWalkSampler, NeighborSampler
from torch_geometric.utils import degree
import numpy as np
from nets import SAGENet, GATNet
from logger import LightLogging
from sampler import GraphSAINTNodeSampler, GraphSAINTEdgeSampler, MySAINTSampler
from sklearn.metrics import f1_score, accuracy_score
import tensorboardX
from utlis import load_dataset
import pandas as pd

log_path = './logs'
summary_path = './summary'


def train_sample(norm_loss):
    model.train()
    model.set_aggr('add')

    total_loss = total_examples = 0
    for data in loader:
        data = data.to(device)
        optimizer.zero_grad()
        if norm_loss == 1:
            out = model(data.x, data.edge_index, data.edge_norm * data.edge_attr)

            loss = F.nll_loss(out, data.y, reduction='none')
            loss = (loss * data.node_norm)[data.train_mask].sum()
        else:
            out = model(data.x, data.edge_index)
            loss = F.nll_loss(out, data.y, reduction='none')[data.train_mask].mean()
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
    f1_scores = []
    for _, mask in data('train_mask', 'val_mask', 'test_mask'):
        accs.append(correct[mask].sum().item() / mask.sum().item())

    for _, mask in data('train_mask', 'val_mask', 'test_mask'):
        f1_scores.append(f1_score(data.y[mask].cpu().numpy(), pred[mask].cpu().numpy(), average='micro'))
    return accs, f1_scores


@torch.no_grad()
def eval_sample(norm_loss):
    model.eval()
    model.set_aggr('add')

    res_df_list = []
    for data in loader:

        data = data.to(device)

        if norm_loss == 1:
            out = model(data.x, data.edge_index, data.edge_norm * data.edge_attr)
        else:
            out = model(data.x, data.edge_index)
        pred = out.argmax(dim=-1)
        correct = pred.eq(data.y.to(device))

        res_batch = pd.DataFrame()
        res_batch['nid'] = data.indices
        res_batch['pred'] = pred
        res_df_list.append(res_batch)

    res_df = pd.concat(res_df_list, axis=0)
    res_df = res_df.groupby('nid')['pred'].apply(lambda x: np.argmax(np.bincount(x['pred']))).reset_index()
    res_df.columns = ['nid', 'pred']
    res_df = res_df.merge('node_df', on=['nid'], how='left')

    accs = res_df.groupby(['mask']).apply(lambda x:accuracy_score(x['y'], x['pred'])).reset_index()
    accs.columns = ['mask','acc']
    accs = accs.sort_values(by=['mask'],ascending=True)

    f1_scores = res_df.groupby(['mask']).apply(lambda x:f1_score(x['y'], x['pred'],average='micro')).reset_index()
    f1_scores.columns = ['mask', 'f1']
    f1_scores = f1_scores.sort_values(by=['mask'],ascending=True)

    accs = list(accs['acc'])
    f1_scores = list(f1_scores['f1'])
    return accs, f1_scores


if __name__ == '__main__':

    args = parse_args()
    log_name = get_log_name(args, prefix='test')

    if args.save_log == 1:
        logger = LightLogging(log_path=log_path, log_name=log_name)
    else:
        logger = LightLogging(log_name=log_name)
    logger.info('Model setting: {}'.format(args))

    dataset = load_dataset(args.dataset)
    logger.info('Dataset: {}'.format(args.dataset))

    data = dataset[0]
    row, col = data.edge_index
    data.edge_attr = 1. / degree(col, data.num_nodes)[col]  # Norm by in-degree.
    data.indices = torch.arange(0, data.num_nodes).int()

    # todo add it into dataset or rewrite it in easy way
    node_df = pd.DataFrame()
    node_df['nid'] = range(data.num_nodes)
    node_df['y'] = data.y.cpu().numpy()
    node_df['mask'] = -1
    node_df['mask'][data.train_mask] = 0
    node_df['mask'][data.val_mask] = 1
    node_df['mask'][data.val_mask] = 2

    if args.sampler == 'rw':
        logger.info('Use GraphSaint randomwalk sampler')
        loader = GraphSAINTRandomWalkSampler(data, batch_size=args.batch_size, walk_length=2,
                                             num_steps=5, sample_coverage=1000,
                                             save_dir=dataset.processed_dir,
                                             num_workers=0)
    elif args.sampler == 'rn':
        logger.info('Use random node sampler')
        loader = MySAINTSampler(data, sample_type='node', batch_size=args.batch_size)
    else:
        raise KeyError('Sampler type error')

    if args.use_gpu == 1:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device('cpu')

    Net = {'sage': SAGENet, 'gat': GATNet}.get(args.gcn_type)
    logger.info('GCN type: {}'.format(args.gcn_type))
    model = Net(in_channels=dataset.num_node_features,
                hidden_channels=256,
                out_channels=dataset.num_classes).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # todo replace by tensorboard
    summary_accs_train = []
    summary_accs_test = []
    summary_f1s_train = []
    summary_f1s_test = []
    for epoch in range(1, args.epochs + 1):
        if args.train_sample == 1:
            loss = train_sample(norm_loss=args.loss_norm)
        else:
            loss = train_full()
        if args.eval_sample == 1:
            accs, f1_scores = eval_sample(norm_loss=args.loss_norm)
        else:
            accs, f1_scores = eval_full()
        if epoch % args.log_interval == 0:
            logger.info(f'Epoch: {epoch:02d}, Loss: {loss:.4f};'
                        f'Train-acc: {accs[0]:.4f}, Train-f1: {f1_scores[0]:.4f}; '
                        # f'Val-acc: {accs[1]:.4f}, Val-f1: {f1_scores[1]:.4f};'
                        f'Test-acc: {accs[2]:.4f}, Test-f1: {f1_scores[2]:.4f};')
        summary_accs_train.append(accs[0])
        summary_f1s_train.append(f1_scores[0])
        summary_accs_test.append(accs[2])
        summary_f1s_test.append(f1_scores[2])

    summary_accs_train = np.array(summary_accs_train)
    summary_accs_test = np.array(summary_accs_test)
    summary_f1s_train = np.array(summary_f1s_train)
    summary_f1s_test = np.array(summary_f1s_test)
    logger.info('Experiment Results:')
    logger.info('Experiment setting: {}'.format(log_name))
    logger.info('Best acc: {}, epoch: {}, f1-micro: {}'.format(summary_accs_test.max(), summary_accs_test.argmax(),
                                                               summary_f1s_test[summary_accs_test.argmax()]))
    logger.info('Best f1-micro: {}, epoch: {}, acc: {}'.format(summary_f1s_test.max(), summary_f1s_test.argmax(),
                                                               summary_accs_test[summary_f1s_test.argmax()]))
    summary_path = summary_path + '/' + log_name + '.npz'
    np.savez(summary_path, train_acc=summary_accs_train, test_acc=summary_accs_test,
             train_f1=summary_f1s_train, test_f1=summary_f1s_test)
    logger.info('Save summary to file')
    logger.info('Save logs to file')
