import numpy as np
import scipy.sparse as sp
import torch
from collections import defaultdict
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import cosine_similarity as cos


def load_data(dataset):
    edge_file = open("data/{}/{}.edge".format(dataset, dataset), 'r')
    attr_file = open("data/{}/{}.node".format(dataset, dataset), 'r')
    
    edge = edge_file.readlines()
    attributes = attr_file.readlines()
    node_num = int(edge[0].split('\t')[1].strip())
    edge_num = int(edge[1].split('\t')[1].strip())
    attribute_number = int(attributes[1].split('\t')[1].strip())
    print("dataset:{}, node_num:{},edge_num:{},attribute_num:{}".format(dataset, node_num, edge_num, attribute_number))

    attributes.pop(0)
    attributes.pop(0)
    att_row = []
    att_col = []

    for line in attributes:
        node1 = int(line.split('\t')[0].strip())
        attribute1 = int(line.split('\t')[1].strip())
        att_row.append(node1)
        att_col.append(attribute1)
    feature = sp.csc_matrix((np.ones(len(att_row)), (att_row, att_col)), shape=(node_num, attribute_number))
    features = normalize_features(feature)
    features = torch.FloatTensor(np.array(features.todense()))

    # load node labels
    label_file = "data/{}/{}.label".format(dataset, dataset)
    y = read_label(label_file)
    labels = torch.LongTensor(y)

    return features, labels, node_num

def generator(dataset, features, node_num, topK):

    # 构造结构超图H1
    path = "./data/{}/".format(dataset)
    edges = np.loadtxt(path + "/" + dataset + ".edge", dtype=str)
    H1 = construct_hypergraph_H1(edges, node_num)

    # 构造属性超图H2
    H2 = construct_hypergraph_H2(features, topK)

    # 构造随机游走H3
    H3 = construct_hypergraph_H3(edges, node_num)

    # 构造属性超图H4
   # H4 = construct_hypergraph_H4(edges, node_num)

    return H1, H2, H3


def get_mask(H, p):
    drop_prob = torch.empty(H.shape, dtype=torch.float32, device=H.device).uniform_(0, 1)
    HH = H.clone()

    mask = drop_prob < p if p > 0 else torch.bernoulli(drop_prob).to(torch.bool)
    HH[~mask] = 0
    return HH

def get_mask_I(H, p):
    drop_prob = torch.empty(H.shape, dtype=torch.float32, device=H.device).uniform_(0, 1)
    HH = H.clone()
    return torch.softmax(drop_prob * HH, dim=1)

def get_mask_II(H, p):
    # drop hyperedge
    drop_prob = torch.empty((H.shape[1],), dtype=torch.float32, device=H.device).uniform_(0, 1) < (p / 10)
    HH = H.clone()
    HH[:, drop_prob] = 0
    return HH

def get_mask_III(H, p):
    #随机生成坐标，然后mask掉
    r, c = H.size()
    ran = (r * c) // p
    row = torch.randint(r, (ran,))
    col = torch.randint(c, (ran,))
    HH = H.clone()
    HH[row, col] = 0
    return HH

def get_mask_IV(H, p):
    drop_prob = torch.empty(H.shape, dtype=torch.float32).uniform_(0, 1)
    mask = torch.bernoulli(drop_prob)
    HH = H.clone()
    HH = HH * mask
    return HH

def construct_hypergraph_H1(edges, node_num):
    print("construct H1...")
    graph = defaultdict(list)
    for edge in edges:
        graph[edge[0]].append(edge[0])
        graph[edge[0]].append(edge[1])
        graph[edge[1]].append(edge[1])
        graph[edge[1]].append(edge[0])

    # 去重复, unique
    for item in graph.items():
        graph[item[0]] = np.unique(item[1])

    indice_matrix = torch.zeros((node_num, node_num))
    # column_names = self.hyperedges.keys()
    for hyperedge in graph.items():
        col = hyperedge[0]
        for node in hyperedge[1]:
            row = node
            indice_matrix[int(row), int(col)] = 1

    return indice_matrix

def construct_hypergraph_H2(X, topk):
    print('construct H2...')
    X = X.numpy()
    # 求特征相似度矩阵
    dist = cos(X)
    # 根据相似度矩阵，求每个节点的topk个最相似的节点（包括自己）
    inds = []
    for i in range(dist.shape[0]):
        ind = np.argpartition(dist[i, :], -(topk + 1))[-(topk + 1):]
        inds.append(ind)

    # 初始化属性超图矩阵
    H2 = torch.zeros((len(X), len(X)))
    # 构造超图
    col = 0
    for hyperedge in inds:
        row = hyperedge
        H2[row, col] = 1
        col = col + 1

    return H2

def construct_hypergraph_H3(edges, node_num):
    print("construct H3...")
    A = np.zeros((node_num, node_num))
    for edge in edges:
        i, j = int(edge[0]), int(edge[1])
        A[i][j] = 1

    A_ = compute_ppr(A)
    return torch.tensor(A_, dtype=torch.float)

def construct_hypergraph_H4(edges, node_num):
    print("construct H4...")
    A = np.zeros((node_num, node_num))
    for edge in edges:
        i, j = int(edge[0]), int(edge[1])
        A[i][j] = 1

    A_ = compute_heat(A)
    return torch.tensor(A_, dtype=torch.float)
#def construct_hypergraph_H4_V(X):
#    return  torch.Tensor(np.array(X.todense()))

def read_label(inputFileName):
    f = open(inputFileName, "r")
    lines = f.readlines()
    f.close()
    N = len(lines)
    y = np.zeros(N, dtype=int)
    i = 0
    for line in lines:
        l = line.strip("\n\r")
        y[i] = int(l)
        i += 1
    return y

def normalize_features(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx

def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)

def training_split(args, ncount):
    nodes = [x for x in range(ncount)]
    train_nodes, test_nodes = train_test_split(nodes, test_size=args.test_size, random_state=args.seed)
    train_nodes = torch.LongTensor(train_nodes)
    test_nodes = torch.LongTensor(test_nodes)
    return train_nodes, test_nodes

def normalize_h1(H1):
    W_e_diag = torch.ones(H1.size()[1])

    D_e_diag = torch.sum(H1, 0)
    D_e_diag = D_e_diag.view((D_e_diag.size()[0]))

    D_v_diag = H1.mm(W_e_diag.view((W_e_diag.size()[0]), 1))
    D_v_diag = D_v_diag.view((D_v_diag.size()[0]))

    Theta = torch.diag(torch.pow(D_v_diag, -0.5)) @ \
            H1 @ torch.diag(W_e_diag) @ \
            torch.diag(torch.pow(D_e_diag, -1)) @ \
            torch.transpose(H1, 0, 1) @ \
            torch.diag(torch.pow(D_v_diag, -0.5))
    return Theta

import networkx as nx
from scipy.linalg import fractional_matrix_power, inv

def compute_ppr(graph, alpha=0.2, self_loop=True):
    # a = nx.convert_matrix.to_numpy_array(graph)
    a = graph
    if self_loop:
        a = a + np.eye(a.shape[0])                                # A^ = A + I_n
    d = np.diag(np.sum(a, 1))                                     # D^ = Sigma A^_ii
    dinv = fractional_matrix_power(d, -0.5)                       # D^(-1/2)
    at = np.matmul(np.matmul(dinv, a), dinv)                      # A~ = D^(-1/2) x A^ x D^(-1/2)
    return alpha * inv((np.eye(a.shape[0]) - (1 - alpha) * at))   # a(I_n-(1-a)A~)^-1


def compute_heat(graph, t=5, self_loop=True):
    # a = nx.convert_matrix.to_numpy_array(graph)
    a = graph
    if self_loop:
        a = a + np.eye(a.shape[0])
    d = np.diag(np.sum(a, 1))
    return np.exp(t * (np.matmul(a, inv(d)) - 1))


#-----------------

def normalize_NodeandEdge(H1, H2, alpha=1, beta=1):
    D_v_t = torch.sum(H1, 1)
    D_v_f = torch.sum(H2, 1)
    D_v_t = torch.diag(torch.pow(D_v_t, -alpha))
    D_v_f = torch.diag(torch.pow(D_v_f, -alpha))

    D_e_t = torch.sum(H1, 0)
    D_e_f = torch.sum(H2, 0)
    D_e_t = torch.diag(torch.pow(D_e_t, -beta))
    D_e_f = torch.diag(torch.pow(D_e_f, -beta))
    return D_v_f, D_v_t, D_e_f, D_e_t

# 函数h1只去重了一个超边中可能重复的结点并没有去重整个超图中重复的超边
def construct_topology_hypergraph(edges, node_num):
    graph = defaultdict(list)
    for edge in edges:
        if(edge[0] not in graph[edge[0]]): graph[edge[0]].append(edge[0])
        if(edge[1] not in graph[edge[0]]): graph[edge[0]].append(edge[1])
        if(edge[1] not in graph[edge[1]]): graph[edge[1]].append(edge[1])
        if(edge[0] not in graph[edge[1]]): graph[edge[1]].append(edge[0])

    hyperedges = []
    for edge in list(graph.values()):
        edge.sort()
        # 每个edge都是list，长度可能不同
        if(edge not in hyperedges):
            hyperedges.append(edge)
        
    indice_matrix = torch.zeros(node_num, len(hyperedges))
    for index, edge in enumerate(hyperedges):
        for node in edge:
            indice_matrix[int(node), index] = 1

    print("H1 "+ str(indice_matrix.shape))
    return indice_matrix

#对于H2，多个结点的相似结点会存在相同，比如k=3时，该3个结点彼此相似，构成同一超边,需要对超边去重
def construct_feature_hypergraph(X, topk=3):
    #对于H2，可以直接使用feature作为关联矩阵的初始状态，即对于所有结点按照相同属性构造一条超边
    if topk >= X.shape[1] or topk <= 0:
        print("H2 "+ str(X.shape))
        # return Parameter(X, requires_grad=True)
        return X

    X = X.numpy()
    dist = cos(X)

    hyperedges = []
    for i in range(dist.shape[0]):
        ind = np.argpartition(dist[i, :], -(topk + 1))[-(topk + 1):]
        ind.sort()
        #if ind not in hyperedges:
        #每一个ind都是大小相同的ndarray
        if (len(hyperedges) == 0) or (ind != hyperedges).all():
            hyperedges.append(ind)

    indice_matrix = torch.zeros((len(X), len(hyperedges)))
    col = 0
    for edge in hyperedges:
        indice_matrix[edge, col] = 1
        col = col + 1
    print("H2 "+ str(indice_matrix.shape))
    return indice_matrix

#-----------------
