from parse_args import parse_args, get_log_name
import torch
from torch_geometric.utils import degree, contains_self_loops, add_self_loops, remove_self_loops
import numpy as np
from nets import SAGENet, GATNet
from logger import LightLogging
from sklearn.metrics import accuracy_score, f1_score
import tensorboardX
from utlis import load_dataset, build_loss_op, build_sampler
import pandas as pd
from time import time

log_path = './logs'
summary_path = './summary'
torch.manual_seed(2020)


def train_sample(norm_loss, loss_op):
    model.train()
    if norm_loss:
        model.set_aggr('add')
    else:
        model.set_aggr('mean')
    sub_graph_nodes = []
    sub_graph_edges = []
    total_loss = total_examples = 0
    for data in loader:
        data = data.to(device)
        optimizer.zero_grad()
        sub_graph_nodes.append(data.num_nodes)
        sub_graph_edges.append(data.edge_index.shape[1])
        if norm_loss == 1:
            out = model(data.x, data.edge_index, data.edge_norm * data.edge_attr)
            loss = loss_op(out, data)
        else:
            out = model(data.x, data.edge_index)
            loss = loss_op(out, data.y)[data.train_mask].mean()
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * data.num_nodes
        total_examples += data.num_nodes

    return total_loss / total_examples, np.mean(sub_graph_nodes), np.mean(sub_graph_edges)


def train_full(loss_op):
    model.train()
    model.set_aggr('mean')

    optimizer.zero_grad()
    out = model(data.x.to(device), data.edge_index.to(device))

    loss = loss_op(out[data.train_mask], data.y.to(device)[data.train_mask]).mean()

    loss.backward()
    optimizer.step()
    return loss.item(), data.num_nodes, data.edge_index.shape[1]


@torch.no_grad()
def eval_full():
    model.eval()
    model.set_aggr('mean')

    out = model(data.x.to(device), data.edge_index.to(device))
    out = out.log_softmax(dim=-1)
    pred = out.argmax(dim=-1)
    correct = pred.eq(data.y.to(device))

    accs = []
    for _, mask in data('train_mask', 'val_mask', 'test_mask'):
        accs.append(correct[mask].sum().item() / mask.sum().item())

    return accs


@torch.no_grad()
def eval_full_multi():
    model.eval()
    model.set_aggr('mean')
    out = model(data.x.to(device), data.edge_index.to(device))
    out = (out > 0).float().cpu().numpy()
    accs = []
    for _, mask in data('train_mask', 'val_mask', 'test_mask'):
        score = f1_score(data.y[mask], out[mask], average='micro')
        accs.append(score)
    return accs


@torch.no_grad()
def eval_sample(norm_loss):
    model.eval()
    if norm_loss:
        model.set_aggr('add')
    else:
        model.set_aggr('mean')

    res_df_list = []
    for data in loader:

        data = data.to(device)

        if norm_loss == 1:
            # TODO: check edge attr
            out = model(data.x, data.edge_index, data.edge_norm * data.edge_attr)
        else:
            out = model(data.x, data.edge_index)
        out = out.log_softmax(dim=-1)
        pred = out.argmax(dim=-1)

        res_batch = pd.DataFrame()
        res_batch['nid'] = data.indices.cpu().numpy()
        res_batch['pred'] = pred.cpu().numpy()
        res_df_list.append(res_batch)

    res_df_duplicate = pd.concat(res_df_list)
    tmp = res_df_duplicate.groupby(['nid', 'pred']).size().unstack().fillna(0)
    res_df = pd.DataFrame()
    res_df['nid'] = tmp.index
    res_df['pred'] = tmp.values.argmax(axis=1)
    # res_df = res_df.groupby('nid')['pred'].apply(lambda x: np.argmax(np.bincount(x))).reset_index()  # 10s

    res_df.columns = ['nid', 'pred']
    res_df = res_df.merge(node_df, on=['nid'], how='left')

    accs = res_df.groupby(['mask']).apply(lambda x: accuracy_score(x['y'], x['pred'])).reset_index()
    accs.columns = ['mask', 'acc']
    accs = accs.sort_values(by=['mask'], ascending=True)
    accs = accs['acc'].values

    return accs


def eval_sample_multi(norm_loss):
    model.eval()
    if norm_loss:
        model.set_aggr('add')
    else:
        model.set_aggr('mean')

    res_df_list = []
    for data in loader:

        data = data.to(device)

        if norm_loss == 1:
            out = model(data.x, data.edge_index, data.edge_norm * data.edge_attr)
        else:
            out = model(data.x, data.edge_index)
        res_batch = (out > 0).float().cpu().numpy()
        res_batch = pd.DataFrame(res_batch)
        res_batch['nid'] = data.indices.cpu().numpy()
        res_df_list.append(res_batch)

    res_df_duplicate = pd.concat(res_df_list)
    length = res_df_duplicate.groupby(['nid']).size().values
    tmp = res_df_duplicate.groupby(['nid']).sum()
    prob = tmp.values
    res_matrix = []
    for i in range(prob.shape[1]):
        a = prob[:, i] / length
        a[a >= 0.5] = 1
        a[a < 0.5] = 0
        res_matrix.append(a)
    res_matrix = np.array(res_matrix).T
    accs = []
    for mask in [train_nid, val_nid, test_nid]:
        accs.append(f1_score(label_matrix[mask], res_matrix[mask], average='micro'))

    return accs


def func(x):
    if x in train_nid:
        return 0
    elif x in val_nid:
        return 1
    elif x in test_nid:
        return 2
    else:
        return -1


if __name__ == '__main__':

    args = parse_args(config_path='./default_hparams.yml')
    log_name = get_log_name(args, prefix='test')
    if args.save_log == 1:
        logger = LightLogging(log_path=log_path, log_name=log_name)
    else:
        logger = LightLogging(log_name=log_name)

    logger.info('Model setting: {}'.format(args))

    dataset = load_dataset(args.dataset)
    logger.info('Dataset: {}'.format(args.dataset))

    if args.dataset in ['flickr', 'reddit']:
        is_multi = False
    else:
        is_multi = True

    data = dataset[0]

    # if args.self_loop == 1 and contains_self_loops(data.edge_index) is False:
    #     logger.info('Raw graph do not contains self loop, add self loop now.')
    #     data.edge_index, _ = add_self_loops(data.edge_index, num_nodes=data.num_nodes)
    # if args.self_loop == 0 and contains_self_loops(data.edge_index):
    #     logger.info('Raw graph contains self loop, remove self loop now.')
    #     data.edge_index, _ = remove_self_loops(data.edge_index)

    row, col = data.edge_index
    data.edge_attr = 1. / degree(col, data.num_nodes)[col]  # Norm by in-degree.
    data.indices = torch.arange(0, data.num_nodes).int()
    data.y = data.y.long()

    # todo add it into dataset or rewrite it in easy way
    if not is_multi:
        node_df = pd.DataFrame()
        node_df['nid'] = range(data.num_nodes)
        node_df['y'] = data.y.cpu().numpy()
        node_df['mask'] = -1
        train_nid = data.indices[data.train_mask].numpy()
        test_nid = data.indices[data.test_mask].numpy()
        val_nid = data.indices[data.val_mask].numpy()
        node_df['mask'] = node_df['nid'].apply(lambda x: func(x))
    else:
        train_nid = data.indices[data.train_mask].numpy()
        test_nid = data.indices[data.test_mask].numpy()
        val_nid = data.indices[data.val_mask].numpy()
        label_matrix = data.y.numpy()

    loader, msg = build_sampler(args, data, dataset.processed_dir)
    logger.info(msg)

    if args.use_gpu == 1:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device('cpu')

    model = {'sage': SAGENet(in_channels=dataset.num_node_features,
                             hidden_channels=args.hidden_units,
                             out_channels=dataset.num_classes),
             'gat': GATNet(in_channels=dataset.num_node_features,
                           hidden_channels=args.hidden_units,
                           num_heads=args.num_heads,
                           out_channels=dataset.num_classes)}.get(args.gcn_type)
    logger.info('GCN type: {}'.format(args.gcn_type))

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_op = build_loss_op(args)

    # todo replace by tensorboard
    summary_accs_train = []
    summary_accs_test = []

    for epoch in range(1, args.epochs + 1):
        if args.train_sample == 1:
            loss, sub_graph_nodes, sub_graph_edges = train_sample(norm_loss=args.loss_norm, loss_op=loss_op)
        else:
            loss, sub_graph_nodes, sub_graph_edges = train_full(loss_op=loss_op)
        if args.eval_sample == 1:
            if is_multi:
                accs = eval_sample_multi(norm_loss=args.loss_norm)
            else:
                accs = eval_sample(norm_loss=args.loss_norm)
        else:
            if is_multi:
                accs = eval_full_multi()
            else:
                accs = eval_full()
        if epoch % args.log_interval == 0:
            logger.info(f'Epoch: {epoch:02d}, Sub graph: ({sub_graph_nodes}, {sub_graph_edges}), '
                        f'Loss: {loss:.4f}, Train-acc: {accs[0]:.4f}, Val-acc: {accs[1]:.4f}, Test-acc: {accs[2]:.4f}')

        summary_accs_train.append(accs[0])
        summary_accs_test.append(accs[2])

    summary_accs_train = np.array(summary_accs_train)
    summary_accs_test = np.array(summary_accs_test)

    logger.info('Experiment Results:')
    logger.info('Experiment setting: {}'.format(log_name))
    logger.info('Best acc: {}, epoch: {}'.format(summary_accs_test.max(), summary_accs_test.argmax()))

    summary_path = summary_path + '/' + log_name + '.npz'
    np.savez(summary_path, train_acc=summary_accs_train, test_acc=summary_accs_test)
    logger.info('Save summary to file')
    logger.info('Save logs to file')
