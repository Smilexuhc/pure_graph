from openTSNE import TSNE
from openTSNE.callbacks import ErrorLogger
from utils.utils import plot
import numpy as np
import os.path as osp
from utils.utils import load_dataset


def plot_graphs(node_emb, label):
    tsne = TSNE(
        perplexity=30,
        metric="euclidean",
        callbacks=ErrorLogger(),
        n_jobs=8,
        random_state=42,
    )
    emb_tsne = tsne.fit(node_emb)
    plot(node_emb, label)


path = osp.join(osp.dirname(osp.realpath(__file__)), 'data', 'flickr')

dataset = load_dataset('flickr')
data = dataset[0]

node_emb = np.load('./data/flickr/flick_ne.npy')
plot(node_emb,data.y.numpy())
