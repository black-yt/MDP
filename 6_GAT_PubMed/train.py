import time
import argparse  # argparse 是python自带的命令行参数解析包，可以用来方便地读取命令行参数
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from utils import load_data, accuracy
from model import GCN


# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='Disables CUDA training.')
parser.add_argument('--fastmode', action='store_true', default=False,
                    help='Validate during training pass.')
parser.add_argument('--seed', type=int, default=42, help='Random seed.')
parser.add_argument('--epochs', type=int, default=100,
                    help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.01,
                    help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=5e-4,
                    help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hidden', type=int, default=16,
                    help='Number of hidden units.')
parser.add_argument('--dropout', type=float, default=0.5,
                    help='Dropout rate (1 - keep probability).')

# 如果程序不禁止使用gpu且当前主机的gpu可用，arg.cuda就为True
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

np.random.seed(args.seed)
torch.manual_seed(args.seed)  # 为CPU设置种子用于生成随机数，以使得结果是确定的
if args.cuda:
    torch.cuda.manual_seed(args.seed)


def computer(no_1_or_2):
    # Load data
    adj, features, labels, idx_train, idx_val, idx_test = load_data(no_1_or_2)

    # Model and optimizer
    model = GCN(in_features=features.shape[1],
                nhid=args.hidden,
                nclass=labels.max().item() + 1,
                dropout=args.dropout)
    optimizer = optim.Adam(model.parameters(),
                           lr=args.lr, weight_decay=args.weight_decay)

    # 数据写入cuda，便于后续加速
    if args.cuda:
        model.cuda()
        features = features.cuda()
        adj = adj.cuda()
        labels = labels.cuda()
        idx_train = idx_train.cuda()
        idx_val = idx_val.cuda()
        idx_test = idx_test.cuda()

    t_total = time.time()
    for epoch in range(args.epochs):
        train(model, optimizer, adj, features, labels, idx_train, idx_val, epoch)
    print("Optimization Finished!")
    print("Total time elapsed: {:.4f}s".format(time.time() - t_total))
    # Testing
    accuracy = test(model, adj, features, labels, idx_test)
    output = model(features, adj)
    return output, accuracy


def train(model, optimizer, adj, features, labels, idx_train, idx_val, epoch):
    t = time.time()  # 返回当前时间
    model.train()
    optimizer.zero_grad()
    # optimizer.zero_grad()意思是把梯度置零，也就是把loss关于weight的导数变成0.
    # pytorch中每一轮batch需要设置optimizer.zero_gra
    output = model(features, adj)
    loss_train = F.nll_loss(output[idx_train], labels[idx_train])
    # 由于在算output时已经使用了log_softmax，这里使用的损失函数就是NLLloss，如果前面没有加log运算，
    # 这里就要使用CrossEntropyLoss了
    # 损失函数NLLLoss() 的输入是一个对数概率向量和一个目标标签. 它不会为我们计算对数概率，
    # 适合最后一层是log_softmax()的网络. 损失函数 CrossEntropyLoss() 与 NLLLoss() 类似,
    # 唯一的不同是它为我们去做 softmax.可以理解为：CrossEntropyLoss()=log_softmax() + NLLLoss()
    acc_train = accuracy(output[idx_train], labels[idx_train])  # 计算准确率
    loss_train.backward()  # 反向求导  Back Propagation
    optimizer.step()  # 更新所有的参数  Gradient Descent

    if not args.fastmode:
        model.eval()  # eval() 函数用来执行一个字符串表达式，并返回表达式的值
        output = model(features, adj)

    loss_val = F.nll_loss(output[idx_val], labels[idx_val])  # 验证集的损失函数
    acc_val = accuracy(output[idx_val], labels[idx_val])
    print('Epoch: {:04d}'.format(epoch+1),
          'loss_train: {:.4f}'.format(loss_train.item()),
          'acc_train: {:.4f}'.format(acc_train.item()),
          'loss_val: {:.4f}'.format(loss_val.item()),
          'acc_val: {:.4f}'.format(acc_val.item()),
          'time: {:.4f}s'.format(time.time() - t))


def train_4xl():
    adj1, features1, labels1, idx_train1, idx_val1, idx_test1, adj = load_data(1)
    adj2, features2, labels2, idx_train2, idx_val2, idx_test2, adj = load_data(2)
    adj3, features3, labels3, idx_train3, idx_val3, idx_test3, adj = load_data(3)
    adj4, features4, labels4, idx_train4, idx_val4, idx_test4, adj = load_data(4)

    model1 = GCN(in_features=features1.shape[1],
                nhid=args.hidden,
                nclass=labels1.max().item() + 1,
                dropout=args.dropout)
    optimizer1 = optim.Adam(model1.parameters(),
                           lr=args.lr, weight_decay=args.weight_decay)

    model2 = GCN(in_features=features2.shape[1],
                 nhid=args.hidden,
                 nclass=labels2.max().item() + 1,
                 dropout=args.dropout)
    optimizer2 = optim.Adam(model2.parameters(),
                            lr=args.lr, weight_decay=args.weight_decay)

    model3 = GCN(in_features=features3.shape[1],
                 nhid=args.hidden,
                 nclass=labels3.max().item() + 1,
                 dropout=args.dropout)
    optimizer3 = optim.Adam(model3.parameters(),
                            lr=args.lr, weight_decay=args.weight_decay)

    model4 = GCN(in_features=features4.shape[1],
                 nhid=args.hidden,
                 nclass=labels4.max().item() + 1,
                 dropout=args.dropout)
    optimizer4 = optim.Adam(model4.parameters(),
                            lr=args.lr, weight_decay=args.weight_decay)

    t_total = time.time()
    time_all = 0
    for epoch in range(args.epochs):
        t = time.time()
        model1.train()
        optimizer1.zero_grad()
        output_1 = model1(features1, adj1)
        loss_train1 = F.nll_loss(output_1[idx_train1], labels1[idx_train1])
        acc_train1 = accuracy(output_1[idx_train1], labels1[idx_train1])  # 计算准确率
        loss_train1.backward()  # 反向求导  Back Propagation
        optimizer1.step()  # 更新所有的参数  Gradient Descent

        if not args.fastmode:
            model1.eval()  # eval() 函数用来执行一个字符串表达式，并返回表达式的值
            output_1 = model1(features1, adj1)
        loss_val1 = F.nll_loss(output_1[idx_val1], labels1[idx_val1])  # 验证集的损失函数
        acc_val1 = accuracy(output_1[idx_val1], labels1[idx_val1])
        time_all = time_all + time.time() - t
        print('DEVICE1 Epoch: {:04d}'.format(epoch + 1),
              'loss_train: {:.4f}'.format(loss_train1.item()),
              'acc_train: {:.4f}'.format(acc_train1.item()),
              'loss_val: {:.4f}'.format(loss_val1.item()),
              'acc_val: {:.4f}'.format(acc_val1.item()),
              'time: {:.4f}s'.format(time.time() - t))

        t = time.time()
        model2.train()
        optimizer2.zero_grad()
        output_2 = model2(features2, adj2)
        loss_train2 = F.nll_loss(output_2[idx_train2], labels2[idx_train2])
        acc_train2 = accuracy(output_2[idx_train2], labels2[idx_train2])  # 计算准确率
        loss_train2.backward()  # 反向求导  Back Propagation
        optimizer2.step()  # 更新所有的参数  Gradient Descent
        if not args.fastmode:
            model2.eval()  # eval() 函数用来执行一个字符串表达式，并返回表达式的值
            output_2 = model2(features2, adj2)
        loss_val2 = F.nll_loss(output_2[idx_val2], labels2[idx_val2])  # 验证集的损失函数
        acc_val2 = accuracy(output_2[idx_val2], labels2[idx_val2])
        print('DEVICE2 Epoch: {:04d}'.format(epoch + 1),
              'loss_train: {:.4f}'.format(loss_train2.item()),
              'acc_train: {:.4f}'.format(acc_train2.item()),
              'loss_val: {:.4f}'.format(loss_val2.item()),
              'acc_val: {:.4f}'.format(acc_val2.item()),
              'time: {:.4f}s'.format(time.time() - t))

        t = time.time()
        model3.train()
        optimizer3.zero_grad()
        output_3 = model3(features3, adj3)
        loss_train3 = F.nll_loss(output_3[idx_train3], labels3[idx_train3])
        acc_train3 = accuracy(output_3[idx_train3], labels3[idx_train3])  # 计算准确率
        loss_train3.backward()  # 反向求导  Back Propagation
        optimizer3.step()  # 更新所有的参数  Gradient Descent

        if not args.fastmode:
            model3.eval()  # eval() 函数用来执行一个字符串表达式，并返回表达式的值
            output_3 = model3(features3, adj3)
        loss_val3 = F.nll_loss(output_3[idx_val3], labels3[idx_val3])  # 验证集的损失函数
        acc_val3 = accuracy(output_3[idx_val3], labels3[idx_val3])
        print('DEVICE3 Epoch: {:04d}'.format(epoch + 1),
              'loss_train: {:.4f}'.format(loss_train3.item()),
              'acc_train: {:.4f}'.format(acc_train3.item()),
              'loss_val: {:.4f}'.format(loss_val3.item()),
              'acc_val: {:.4f}'.format(acc_val3.item()),
              'time: {:.4f}s'.format(time.time() - t))

        t = time.time()
        model4.train()
        optimizer4.zero_grad()
        output_4 = model4(features4, adj4)
        loss_train4 = F.nll_loss(output_4[idx_train4], labels4[idx_train4])
        acc_train4 = accuracy(output_4[idx_train4], labels4[idx_train4])  # 计算准确率
        loss_train4.backward()  # 反向求导  Back Propagation
        optimizer4.step()  # 更新所有的参数  Gradient Descent

        if not args.fastmode:
            model4.eval()  # eval() 函数用来执行一个字符串表达式，并返回表达式的值
            output_4 = model4(features4, adj4)
        loss_val4 = F.nll_loss(output_4[idx_val4], labels4[idx_val4])  # 验证集的损失函数
        acc_val4 = accuracy(output_4[idx_val4], labels4[idx_val4])
        print('DEVICE4 Epoch: {:04d}'.format(epoch + 1),
              'loss_train: {:.4f}'.format(loss_train4.item()),
              'acc_train: {:.4f}'.format(acc_train4.item()),
              'loss_val: {:.4f}'.format(loss_val4.item()),
              'acc_val: {:.4f}'.format(acc_val4.item()),
              'time: {:.4f}s'.format(time.time() - t))

        t = time.time()
        weight1 = model1.gcn1.weight.data
        weight2 = model2.gcn1.weight.data
        weight3 = model3.gcn1.weight.data
        weight4 = model4.gcn1.weight.data
        weight_a = (weight1 + weight2 + weight3 + weight4) / 4
        model1.gcn1.weight.data = weight_a
        model2.gcn1.weight.data = weight_a
        model3.gcn1.weight.data = weight_a
        model4.gcn1.weight.data = weight_a

        weight1 = model1.gcn2.weight.data
        weight2 = model2.gcn2.weight.data
        weight3 = model3.gcn2.weight.data
        weight4 = model4.gcn2.weight.data
        weight_a = (weight1 + weight2 + weight3 + weight4) / 4
        model1.gcn2.weight.data = weight_a
        model2.gcn2.weight.data = weight_a
        model3.gcn2.weight.data = weight_a
        model4.gcn2.weight.data = weight_a

        bias1 = model1.gcn1.bias.data
        bias2 = model2.gcn1.bias.data
        bias3 = model3.gcn1.bias.data
        bias4 = model4.gcn1.bias.data
        bias_a = (bias1 + bias2 + bias3 + bias4) / 4
        model1.gcn1.bias.data = bias_a
        model2.gcn1.bias.data = bias_a
        model3.gcn1.bias.data = bias_a
        model4.gcn1.bias.data = bias_a

        bias1 = model1.gcn2.bias.data
        bias2 = model2.gcn2.bias.data
        bias3 = model3.gcn2.bias.data
        bias4 = model4.gcn2.bias.data
        bias_a = (bias1 + bias2 + bias3 + bias4) / 4
        model1.gcn2.bias.data = bias_a
        model2.gcn2.bias.data = bias_a
        model3.gcn2.bias.data = bias_a
        model4.gcn2.bias.data = bias_a
        time_all = time_all + time.time() - t
        test(model1, adj, features1, labels1, idx_test1)

    print("Optimization Finished!")
    print("Total time elapsed: {:.4f}s".format(time.time() - t_total))
    print("Estimated time required: {:.4f}s".format(time_all))
    print("device with complete adj:")
    final_accuracy = test(model1, adj, features1, labels1, idx_test1)
    print("final accuracy with complete adj:")
    print(final_accuracy)


def test(model, adj, features, labels, idx_test):
    # 定义测试函数，相当于对已有的模型在测试集上运行对应的loss与accuracy
    model.eval()
    output = model(features, adj)
    loss_test = F.nll_loss(output[idx_test], labels[idx_test])
    acc_test = accuracy(output[idx_test], labels[idx_test])
    print("Test set results:",
          "loss= {:.4f}".format(loss_test.item()),
          "accuracy= {:.4f}".format(acc_test.item()))
    return acc_test.item()


if __name__ == '__main__':
    load_data(0)
    train_4xl()
