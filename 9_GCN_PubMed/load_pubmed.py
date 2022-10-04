import numpy as np
from scipy import sparse as sp
import scipy.io as sio


def load_edges(file_name):
    with open(file_name) as fin:
        edges = []
        for line in fin:
            edges.append(line.split(','))
    edges = np.array(edges)
    return edges.astype(np.int)
    

def load_pubmed():
    folder = "pubmed"
    data = sio.loadmat(f'datasets/pubmed/pubmed.mat')
    adj_complete = data['W'].toarray()
    adj_shape = adj_complete.shape
    edges = load_edges(f"datasets/{folder}/edges.csv")
    adj_complete = sp.csr_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])), shape=adj_shape)
    all_features = np.array(data['fea'])
    all_labels = []
    with open(f"datasets/pubmed/labels_pubmed.csv", "r") as fin:
        for label in fin:
            all_labels.append(label.strip())

    return adj_complete, all_features, all_labels
