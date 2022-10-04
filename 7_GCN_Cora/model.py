import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class GraphConvolution(nn.Module):
	# 初始化层：输入feature，输出feature，权重，偏移
	def __init__(self, in_features, out_features, bias=True):
		super(GraphConvolution, self).__init__()
		self.in_features = in_features
		self.out_features = out_features
		self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
		# 常见用法self.v = torch.nn.Parameter(torch.FloatTensor(hidden_size))：
		# 首先可以把这个函数理解为类型转换函数，将一个不可训练的类型Tensor转换成可以训练的类型parameter并将这个parameter
		# 绑定到这个module里面，所以经过类型转换这个self.v变成了模型的一部分，成为了模型中根据训练可以改动的参数了。
		# 使用这个函数的目的也是想让某些变量在学习的过程中不断的修改其值以达到最优化。
		if bias:
			self.bias = nn.Parameter(torch.FloatTensor(out_features))
		else:
			self.register_parameter('bias', None)
		# Parameters与register_parameter都会向parameters写入参数，但是后者可以支持字符串命名
		self.reset_parameters()

	# 初始化权重
	def reset_parameters(self):
		stdv = 1. / math.sqrt(self.weight.size(1))
		# size()函数主要是用来统计矩阵元素个数，或矩阵某一维上的元素个数的函数  size（1）为行
		self.weight.data.uniform_(-stdv, stdv)  # uniform() 方法将随机生成下一个实数，它在 [x, y] 范围内
		if self.bias is not None:
			self.bias.data.uniform_(-stdv, stdv)

	'''
	前馈运算 即计算A~ X W(0)
	input X与权重W相乘，然后adj矩阵与他们的积稀疏乘
	直接输入与权重之间进行torch.mm操作，得到support，即XW
	support与adj进行torch.spmm操作，得到output，即AXW选择是否加bias
	'''
	def forward(self, x, adj):
		support = torch.mm(x, self.weight)
		# torch.mm(a, b)是矩阵a和b矩阵相乘，torch.mul(a, b)是矩阵a和b对应位相乘，a和b的维度必须相等
		output = torch.mm(adj, support)  # adj是稀疏矩阵
		if self.bias is not None:
			return output + self.bias
		else:
			return output

	# 通过设置断点，可以看出output的形式是0.01，0.01，0.01，0.01，0.01，0.01，0.94]
	# 里面的值代表该x对应标签不同的概率，故此值可转换为#[0,0,0,0,0,0,1]，对应我们之前把标签one-hot后的第七种标签

	def __repr__(self):
		return self.__class__.__name__ + ' (' \
			   + str(self.in_features) + ' -> ' \
			   + str(self.out_features) + ')'


class GCN(nn.Module):
	def __init__(self, in_features, nhid, nclass, dropout):  # 底层节点的参数，feature的个数；隐层节点个数；最终的分类数
		super(GCN, self).__init__()  # super()._init_()在利用父类里的对象构造函数
		self.in_features = in_features
		self.nhid = nhid
		self.nclass = nclass
		self.dropout = dropout
		self.gcn1 = GraphConvolution(in_features, nhid)  # gc1输入尺寸in_features，输出尺寸nhid
		self.gcn2 = GraphConvolution(nhid, nclass)  # gc2输入尺寸nhid，输出尺寸ncalss

	# 输入分别是特征和邻接矩阵。最后输出为输出层做log_softmax变换的结果
	def forward(self, x, adj):
		h1 = F.relu(self.gcn1(x, adj))  # adj即公式Z=softmax(A~Relu(A~XW(0))W(1))中的A~
		h1_d = F.dropout(h1, self.dropout, training=self.training)  # h1_d要dropout
		logits = self.gcn2(h1_d, adj)
		output = F.log_softmax(logits, dim=1)
		return output
