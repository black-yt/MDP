import scipy.sparse as sp
import numpy as np
from utils import normalize, sparse_mx_to_torch_sparse_tensor


def attention(support, adj):
    support = support.detach().numpy()
    adj = adj.to_dense().detach().numpy()
    relationship = np.dot(support, support.T)
    relationship = (relationship-np.min(relationship))/(np.max(relationship)-np.min(relationship))
    # a - np.min(a, axis=1)[:,None]
    relationship = adj * relationship
    relationship = sp.coo_matrix(relationship)

    relationship = normalize(relationship)
    relationship = sparse_mx_to_torch_sparse_tensor(relationship)
    relationship = relationship.float()
    return relationship
