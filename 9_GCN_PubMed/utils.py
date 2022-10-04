import numpy as np
import scipy.sparse as sp
import torch
import h5py
from load_pubmed import load_pubmed

'''
先将所有由字符串表示的标签数组用set保存，set的重要特征就是元素没有重复，
因此表示成set后可以直接得到所有标签的总数，随后为每个标签分配一个编号，创建一个单位矩阵，
单位矩阵的每一行对应一个one-hot向量，也就是np.identity(len(classes))[i, :]，
再将每个数据对应的标签表示成的one-hot向量，类型为numpy数组
'''


def encode_onehot(labels):
    classes = set(labels)  # 得到所有类别，利用set去重
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in enumerate(classes)}  # 为类别分配one-hot编码
    labels_onehot = np.array(list(map(classes_dict.get, labels)),
                             dtype=np.int32)
    return labels_onehot

# def load_data(miss_1_or_2, path="./datasets/cora/", dataset="cora"):
def load_data(miss_1_or_2, path="./datasets/pubmed/", dataset="pubmed"):
    global laplace, laplace_a, laplace_b, features, labels, idx_train, idx_val, idx_test, \
        laplace_up, laplace_down, laplace_a_up, laplace_a_down, laplace_b_up, laplace_b_down, \
        idx_train_up, idx_train_down

    if miss_1_or_2 == 0:
        """Load citation network dataset (cora only for now)"""
        print('Loading {} dataset...'.format(dataset))

        adj, features, labels = load_pubmed()
        labels = encode_onehot(labels)

        nodes_count = adj.shape[0]

        mat = h5py.File(f'datasets/pubmed/A.mat')
        laplace = np.array(mat['A'], dtype='float16')
        mat = h5py.File(f'datasets/pubmed/L1.mat')
        laplace_a = np.array(mat['out_1'], dtype='float16')
        mat = h5py.File(f'datasets/pubmed/L2.mat')
        laplace_b = np.array(mat['out_2'], dtype='float16')

        features = normalize(features)  # 对属性矩阵采用行归一化
        the_h = features.max(0)
        epsilon = 30
        the_lambda = the_h / epsilon
        noise = np.random.laplace(0, the_lambda, features.shape)
        features = features + noise
        # 分别构建训练集、验证集、测试集

        features = torch.FloatTensor(np.array(features))  # tensor为pytorch的数据结构
        labels = torch.LongTensor(np.where(labels)[1])
        laplace = torch.FloatTensor(laplace)
        laplace_a = torch.FloatTensor(laplace_a)
        laplace_b = torch.FloatTensor(laplace_b)
        # 分别构建训练集、验证集、测试集
        idx_train_up = np.array(range(0, 15000, 3))
        idx_train_down = np.array(range(1, 15001, 3))
        idx_train = np.append(idx_train_up, idx_train_down)
        idx_val = range(16000, 18000)
        idx_test = range(2, 15002, 3)
        idx_val = torch.LongTensor(idx_val)
        idx_test = torch.LongTensor(idx_test)
        idx_train = torch.LongTensor(idx_train)

    if miss_1_or_2 == 1:
        return laplace_a, features, labels, idx_train, idx_val, idx_test, laplace

    if miss_1_or_2 == 2:
        return laplace_b, features, labels, idx_train, idx_val, idx_test, laplace


def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))  # 对每一行求和
    r_inv = np.power(rowsum, -1).flatten()  # 求倒数
    r_inv[np.isinf(r_inv)] = 0.  # 如果某一行全为0，则r_inv算出来会等于无穷大，将这些行的r_inv置为0
    r_mat_inv = sp.diags(r_inv)  # 构建对角元素为r_inv的对角矩阵
    mx = r_mat_inv.dot(mx)
    # 用对角矩阵与原始矩阵的点积起到标准化的作用，原始矩阵中每一行元素都会与对应的r_inv相乘，相当于除以了sum
    return mx


def degree(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))  # 对每一行求和
    r_inv = np.power(rowsum, 1).flatten()  # 求倒数
    r_mat_inv = sp.diags(r_inv)  # 构建对角元素为r_inv的对角矩阵
    return r_mat_inv


def degree_0_5(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))  # 对每一行求和
    r_inv = np.power(rowsum, -0.5).flatten()  # 求倒数
    r_mat_inv = sp.diags(r_inv)  # 构建对角元素为r_inv的对角矩阵
    return r_mat_inv


def sparse_mx_to_torch_sparse_tensor(sparse_mx):  # 把一个sparse matrix转为torch稀疏张量
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    """
    numpy中的ndarray转化成pytorch中的tensor : torch.from_numpy()
    pytorch中的tensor转化成numpy中的ndarray : numpy()
    """
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()  # 记录等于preds的label eq:equal
    correct = correct.sum()
    return correct / len(labels)