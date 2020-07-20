from sklearn.cluster import KMeans, DBSCAN, SpectralClustering
from torch_geometric.data.cluster import ClusterData, ClusterLoader
from torch_geometric.utils import subgraph
from torch_geometric.nn import Node2Vec
import copy
import numpy as np
import os.path as osp
import torch
from torch_sparse import SparseTensor, rw, saint


class GECData(object):
    def __init__(self, data=None, embedding_size=128, ge_method='node2vec', use_gpu=1, save_dir=None, logging=print):

        if ge_method not in ['node2vec']:
            raise ValueError('ge method error')

        self._ge_method = ge_method
        self._embedding_size = embedding_size
        self._file_path = osp.join(save_dir or '', self.__filename__)

        if save_dir is not None and osp.exists(self._file_path):
            logging('Load saved graph embedding results from {}.'.format(self._file_path))
        else:
            logging('Train graph embedding using Node2vec.')
            if use_gpu == 1:
                device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            else:
                device = torch.device('cpu')
            self.__train_embedding__(data, device, logging)

    @property
    def __filename__(self):
        return f'{self.__class__.__name__.lower()}_{self._ge_method}_{self._embedding_size}.npy'

    def __train_step__(self, model, optimizer, loader, device):
        model.train()
        total_loss = 0
        for pos_rw, neg_rw in loader:
            optimizer.zero_grad()
            loss = model.loss(pos_rw.to(device), neg_rw.to(device))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        return total_loss / len(loader)

    @torch.no_grad()
    def __test_step__(self, model, data):
        # TODO: in real task, we can not see labels
        model.eval()
        z = model()
        acc = model.test(z[data.train_mask], data.y[data.train_mask],
                         z[data.test_mask], data.y[data.test_mask], max_iter=150)
        return acc

    @torch.no_grad()
    def __save_results__(self, model, data, device, save_dir):
        model.eval()
        z = model(torch.arange(data.num_nodes, device=device))
        z = z.cpu().numpy()
        np.save(save_dir, z)

    def __train_embedding__(self, data, device, logging):
        model = Node2Vec(data.edge_index, embedding_dim=self._embedding_size, walk_length=20,
                         context_size=10, walks_per_node=10, num_negative_samples=1,
                         sparse=True).to(device)

        loader = model.loader(batch_size=128, shuffle=True, num_workers=4)

        optimizer = torch.optim.SparseAdam(model.parameters(), lr=0.01)
        for epoch in range(30):
            loss = self.__train_step__(model,
                                       optimizer,
                                       loader,
                                       device)
            # acc = self.__test_step__(model, data)
            # logging(f'Epoch: {epoch:02d}, Loss: {loss:.4f}, Acc: {acc:.4f}')
            logging(f'Epoch-{epoch:02d}, Loss: {loss:.4f}')

        self.__save_results__(model, data, device, self._file_path)
        logging('Save node embedding to: {}'.format(self._file_path))

    def get_node_emb(self):
        return np.load(self._file_path)


class GECSampler(object):
    def __init__(self, data, node_emb, num_clusters=25, cluster_type='kmeans', min_nodes='auto', walk_length=0,
                 save_dir=None, logging=print):
        """

        Args:
            data:
            node_emb:
            num_clusters:
            cluster_type:
            min_nodes(int): min nodes for a sub graph
            walk_length:
            save_dir:
            logging:
        """
        assert data.edge_index is not None
        assert 'node_norm' not in data
        assert 'edge_norm' not in data

        if cluster_type not in ['kmeans', 'dbscan', 'spectral']:
            raise ValueError('Cluster type error.')

        self._cluster_type = cluster_type
        self._num_clusters = num_clusters
        self._node_emb = node_emb
        self._data = copy.copy(data)
        self._N = N = data.num_nodes
        self._E = data.num_edges
        self._walk_length = walk_length
        if min_nodes == 'auto':
            self._min_nodes = N // num_clusters
        else:
            self._min_nodes = min_nodes

        self._adj = SparseTensor(row=data.edge_index[0], col=data.edge_index[1],
                                 value=data.edge_attr, sparse_sizes=(N, N))

        file_path = osp.join(save_dir or '', self.__filename__)
        if save_dir is not None and osp.exists(file_path):
            self._cluster_labels = np.load(file_path)
            logging('Load saved cluster results from {}'.format(save_dir))
        else:
            self._cluster_labels = self.__cluster__()
            logging('Training cluster model to partition graph')
            np.save(file_path, self._cluster_labels, )

    @property
    def __filename__(self):
        if self._cluster_type in ['kmeans', 'spectral']:
            return f'{self.__class__.__name__.lower()}_{self._cluster_type}_{self._num_clusters}.npy'
        elif self._cluster_type == 'dbscan':
            return f'{self.__class__.__name__.lower()}_{self._cluster_type}.npy'
        raise ValueError()

    def __len__(self):
        return self._num_clusters

    def __cluster__(self):
        if self._cluster_type == 'kmeans':
            cluster_model = KMeans(n_clusters=self._num_clusters,
                                   verbose=10)
            return cluster_model.fit_predict(self._node_emb)
        elif self._cluster_type == 'dbscan':
            cluster_model = DBSCAN()
            return cluster_model.fit_predict(self._node_emb)
        elif self._cluster_type == 'spectral':
            cluster_model = SpectralClustering(n_clusters=self._num_clusters)
            return cluster_model.fit_predict(self._node_emb)
        else:
            raise ValueError()

    def __get_data_from_sample(self, sample):
        nid, adj, eid = sample
        data = self._data.__class__()
        data.num_nodes = nid.size(0)
        row, col, value = adj.coo()
        data.edge_index = torch.stack([row, col], dim=0)
        data.edge_attr = value

        for key, item in self._data:
            if item.size(0) == self._N:
                data[key] = item[nid]
            elif item.size(0) == self._E:
                data[key] = item[eid]
            elif key == 'edge_index':
                pass
            else:
                data[key] = item

        data.n_id = nid
        # data.res_n_id = res_n_id
        return data

    # def __sample_nodes__(self):
    #     clusters = np.random.permutation(self._num_clusters)  # for shuffle
    #     all_nodes = np.arange(self._N)
    #     sample_nodes_list = []
    #     for i in clusters:
    #         init_nid = torch.from_numpy(all_nodes[self._cluster_labels == i])
    #         if self._walk_length:
    #             nid = self._adj.random_walk(init_nid, self._walk_length)
    #             nid = nid.flatten().unique()
    #         else:
    #             nid = init_nid
    #             # res_nid = torch.arange(nid.size(0), dtype=torch.long)
    #         sample_nodes_list.append(nid)
    #     return sample_nodes_list

    def __sample_nodes__(self):
        clusters = np.random.permutation(self._num_clusters)
        all_nodes = np.arange(self._N)
        sample_nodes_list = []
        for i in clusters:

            init_nid = torch.from_numpy(all_nodes[self._cluster_labels == i])
            num_nodes = init_nid.size()[0]
            if num_nodes < self._min_nodes and self._min_nodes != 0:
                walk_length = self._min_nodes // num_nodes
                nid = self._adj.random_walk(init_nid, walk_length)
                nid = nid.flatten().unique()
            else:
                nid = init_nid
            sample_nodes_list.append(nid)
        return sample_nodes_list

    def __sample_graph__(self):
        sample_graph_list = []
        for nid in self.__sample_nodes__():
            adj, eid = self._adj.saint_subgraph(nid)
            sample_graph_list.append((nid, adj, eid))
        return sample_graph_list

    def __iter__(self):
        for sub_graph in self.__sample_graph__():
            data = self.__get_data_from_sample(sub_graph)
            yield data




