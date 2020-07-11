from sklearn.cluster import KMeans, DBSCAN
from torch_geometric.data.cluster import ClusterData,ClusterLoader


class GraphEmbeddingClusterSampler(object):
    def __init__(self, data, num_cluster=25, cluster_type='kmeans', logger=print):
        assert data.edge_index is not None
        assert 'node_norm' not in data
        assert 'edge_norm' not in data

        if cluster_type not in ['kmeans']:
            raise ValueError('Cluster type error.')

