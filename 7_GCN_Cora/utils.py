import numpy as np
import scipy.sparse as sp
import torch
import heapq

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

def load_data(miss_1_or_2, path="./datasets/cora/", dataset="cora"):
    global laplace, laplace_a, laplace_b, features, labels, idx_train, idx_val, idx_test, \
        laplace_up, laplace_down, laplace_a_up, laplace_a_down, laplace_b_up, laplace_b_down, \
        idx_train_up, idx_train_down

    if miss_1_or_2 == 0:
        """Load citation network dataset (cora only for now)"""
        print('Loading {} dataset...'.format(dataset))

        idx_features_labels = np.genfromtxt("{}{}.content".format(path, dataset),
                                            dtype=np.dtype(str))
        features = np.array(idx_features_labels[:, 1:-1], dtype=np.float32)
        labels = encode_onehot(idx_features_labels[:, -1])
        # 这里的label为one-hot格式，如第一类代表[1,0,0,0,0,0,0]
        # content file的每一行的格式为 ： <paper_id> <word_attributes>+ <class_label>
        #    分别对应 0, 1:-1, -1
        # feature为第二列到倒数第二列，labels为最后一列

        # build graph
        # cites file的每一行格式为：  <cited paper ID>  <citing paper ID>
        # 根据前面的contents与这里的cites创建图，算出edges矩阵与adj 矩阵
        # idx = np.array(idx_features_labels[:, 0], dtype=np.int32)
        idx = np.array(idx_features_labels[:, 0], dtype=np.str)  # 论文编号
        nodes_count = idx.shape[0]
        idx_map = {j: i for i, j in enumerate(idx)}  # 每篇论文的索引是多少
        # 由于文件中节点并非是按顺序排列的，因此建立一个编号为0-(node_size-1)的哈希表idx_map，
        # 哈希表中每一项为id: number，即节点id对应的编号为number
        edges_unordered = np.genfromtxt("{}{}.cites".format(path, dataset), dtype=np.str)
        # edges_unordered为直接从边表文件中直接读取的结果，是一个(edge_num, 2)的数组，每一行表示一条边两个端点的idx
        edges = np.array(list(map(idx_map.get, edges_unordered.flatten())),  # flatten展开成一维向量
                         dtype=np.str).reshape(edges_unordered.shape)  # 将id相对应的边，改成索引相对应的边。将edges_unordered.flatten()中的值，输入get函数中，返回value
        adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),  # (edges[:, 0], edges[:, 1])这些位置的值为1
                            shape=(labels.shape[0], labels.shape[0]),
                            dtype=np.float32)
        # 根据coo矩阵性质，这一段的作用就是，网络有多少条边，邻接矩阵就有多少个1，
        # 所以先创建一个长度为edge_num的全1数组，每个1的填充位置就是一条边中两个端点的编号，
        # 即edges[:, 0], edges[:, 1]，矩阵的形状为(node_size, node_size)。

        adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)  # 变成对称矩阵
        adj = adj + sp.eye(adj.shape[0])  # 加入自环，变成A~，对应公式A~=A+IN
        degree_2 = degree_0_5(adj)  # (D~)^(-0.5)
        adj = np.dot(degree_2, adj)
        adj = np.dot(adj, degree_2)

        # degree_1 = degree(adj)
        # adj = degree_1 - adj  # L=D-A

        laplace = adj.toarray()
        eigen_value, eigen_vector = np.linalg.eig(laplace)

        # AB随机取特征值
        creat_miss = np.random.randint(0, 100, nodes_count)
        miss_1 = np.int64(creat_miss > 50)

        eigen_value_1 = sp.diags(eigen_value * miss_1).A
        laplace_a = np.dot(eigen_vector, eigen_value_1)
        laplace_a = np.dot(laplace_a, np.linalg.inv(eigen_vector))
        laplace_a = laplace_a.real
        laplace_b = laplace - laplace_a

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
        idx_train_up = np.array(range(0, 200))
        idx_train_down = np.array(range(1500, 1700))
        idx_train = np.append(idx_train_up, idx_train_down)
        idx_val = range(200, 500)
        idx_test = range(500, 1500)
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