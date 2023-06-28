import numpy as np
import functools
import scipy.sparse as sp

from sklearn.metrics import f1_score
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import normalize, OneHotEncoder

import torch
import torch.nn.functional as F
import torch_sparse
from torch_scatter import scatter
import torch_geometric.transforms as T
from torch_geometric.datasets import Planetoid, CitationFull

def prob_to_one_hot(y_pred):
    ret = np.zeros(y_pred.shape, np.bool)
    indices = np.argmax(y_pred, axis=1)
    for i in range(y_pred.shape[0]):
        ret[i][indices[i]] = True
    return ret


def print_statistics(statistics, function_name):
    print(f'(E) | {function_name}:', end=' ')
    for i, key in enumerate(statistics.keys()):
        mean = statistics[key]['mean']
        std = statistics[key]['std']
        print(f'{key}={mean:.4f}+-{std:.4f}', end='')
        if i != len(statistics.keys()) - 1:
            print(',', end=' ')
        else:
            print()

def repeat(n_times):
    def decorator(f):
        @functools.wraps(f)
        def wrapper(*args, **kwargs):
            results = [f(*args, **kwargs) for _ in range(n_times)]
            statistics = {}
            for key in results[0].keys():
                values = [r[key] for r in results]
                statistics[key] = {
                    'mean': np.mean(values),
                    'std': np.std(values)}
            print_statistics(statistics, f.__name__)
            return statistics
        return wrapper
    return decorator



@repeat(3)
def label_classification(embeddings, h_emb, y, ratio):
     
    X = embeddings.detach().cpu().numpy()
    Y = y.detach().cpu().numpy()
    Y = Y.reshape(-1, 1)
    onehot_encoder = OneHotEncoder(categories='auto').fit(Y)
    Y = onehot_encoder.transform(Y).toarray().astype(np.bool)
   
    if h_emb is not None:
        H = h_emb.detach().cpu().numpy()
        X = X + H

    X = normalize(X, norm='l2')

    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=1 - ratio)

    if h_emb is None:
        logreg = LogisticRegression(solver='liblinear')
        c = 2.0 ** np.arange(-10, 10)
        clf = GridSearchCV(estimator=OneVsRestClassifier(logreg),
                       param_grid=dict(estimator__C=c), n_jobs=8, cv=5,
                       verbose=0)
    else:
        mlp = MLPClassifier(learning_rate='adaptive', early_stopping=True)
        param_grids = {
            "solver": ['adam', 'sgd', 'lbfgs'],
            'hidden_layer_sizes': [(X.shape[-1],)]

        }
        clf = GridSearchCV(estimator=mlp,
                       param_grid=param_grids, n_jobs=8, cv=5,
                       verbose=0)
    
    
    clf.fit(X_train, y_train)

    y_pred = clf.predict_proba(X_test)
    y_pred = prob_to_one_hot(y_pred)

    # acc = sklearn.metrics.accuracy_score(y_test, y_pred)
    micro = f1_score(y_test, y_pred, average="micro")
    macro = f1_score(y_test, y_pred, average="macro")

    return {
        'acc': micro,
        'F1Mi': macro
    }


def get_dataset(path, name):
        assert name in ['Cora', 'CiteSeer', 'PubMed', 'DBLP']
        name = 'dblp' if name == 'DBLP' else name

        return (CitationFull if name == 'dblp' else Planetoid)(
            path,
            name,
            transform=T.NormalizeFeatures())
    
def drop_feature(x, drop_prob):
    drop_mask = torch.empty(
        (x.size(1), ),
        dtype=torch.float32,
        device=x.device).uniform_(0, 1) < drop_prob
    x = x.clone()
    x[:, drop_mask] = 0

    return x

def build_hypergraph(args, data, alpha = 0.8, bin = False):
    # feature similarity hypergraph

    S = torch.mm(data.x, data.x.T).cpu()

    if bin :
        S = S.numpy()
        # use alpha to get a binary incidence matrix
        H = np.where(S > alpha, 1, 0)
    else:
        # get non-real value matrix
        # S is dense matrix using large memeory 
        H = F.softmax(S, dim=1)
        H = H.numpy()

    H = sp.csr_matrix(H, dtype=int)
    (V, E), _ = torch_sparse.from_scipy(H)

    if "UniGCNConv" in args.model_name:
        degV = torch.from_numpy(H.sum(1)).view(-1, 1).float()
        degE = scatter(degV[V], E, dim=0, reduce=args.first_aggregate)
        degE = degE.pow(-0.5)
        degV = degV.pow(-0.5)
        degV[degV.isinf()] = 1

        args.degE = degE.cuda()
        args.degV = degV.cuda()

    return args, V.cuda(), E.cuda()

def build_hypergraph_I(args, data, bin=False):
    # topoloy hypergraph

    N = data.num_nodes
    row = data.edge_index[0].cpu().numpy()
    col = data.edge_index[1].cpu().numpy()
    val = np.ones((len(row),), dtype=int)

    if not data.has_self_loops():
        nodes = np.unique(np.concatenate((row, col)))
        row = np.concatenate((row, nodes))
        col = np.concatenate((col, nodes))
        val = np.ones((len(row),), dtype=int)

    H = sp.coo_matrix((val, (row, col)), shape=(N, N), dtype=int).tocsr()

    (V, E), _ = torch_sparse.from_scipy(H)

    if "UniGCNConv" in args.model_name:
        degV = torch.from_numpy(H.sum(1)).view(-1, 1).float()
        degE = scatter(degV[V], E, dim=0, reduce=args.first_aggregate)
        degE = degE.pow(-0.5)
        degV = degV.pow(-0.5)
        degV[degV.isinf()] = 1

        args.degE = degE.cuda()
        args.degV = degV.cuda()

    return args, V.cuda(), E.cuda()

def build_hypergraph_II(args, data, alpha= 0.6):
    # topoloy + similarity  hypergraph

    N = data.num_nodes
    row = data.edge_index[0].cpu().numpy()
    col = data.edge_index[1].cpu().numpy()
    val = np.ones((len(row),), dtype=int)

    if not data.has_self_loops():
        nodes = np.unique(np.concatenate((row, col)))
        row = np.concatenate((row, nodes))
        col = np.concatenate((col, nodes))
        val = np.ones((len(row),), dtype=int)

    H = sp.coo_matrix((val, (row, col)), shape=(N, N), dtype=int).tocsr()
    # should S be sparselize?  
    S = torch.mm(data.x, data.x.T).cpu().numpy()
    S = S * H
    H = np.where(S > alpha, 1, 0)
    H = sp.csr_matrix(H, dtype=float)

    (V, E), _ = torch_sparse.from_scipy(H)

    if "UniGCNConv" in args.model_name:
        degV = torch.from_numpy(H.sum(1)).view(-1, 1).float()
        degE = scatter(degV[V], E, dim=0, reduce=args.first_aggregate)
        degE = degE.pow(-0.5)
        degV = degV.pow(-0.5)
        degV[degV.isinf()] = 1

        args.degE = degE.cuda()
        args.degV = degV.cuda()

    return args, V.cuda(), E.cuda()

def build_vanilla_graph(data):
    N = data.num_nodes
    row = data.edge_index[0].cpu().numpy()
    col = data.edge_index[1].cpu().numpy()
    val = np.ones((len(row),), dtype=int)

    if not data.has_self_loops():
        nodes = np.unique(np.concatenate((row, col)))
        row = np.concatenate((row, nodes))
        col = np.concatenate((col, nodes))
        val = np.ones((len(row),), dtype=int)

    A = sp.coo_matrix((val, (row, col)), shape=(N, N), dtype=int).toarray()
    S = torch.mm(data.x, data.x.T).cpu().numpy()

    zero = 0.5 * A
    no_zero = A + S - A * S
    H = np.where(S == 0, zero, no_zero)
    H = torch.tensor(H,  dtype=float)
    
    # 内存不足再考虑稀疏存储
    # H = sp.coo_matrix(H, dtype=float)

    # src, tar = H.data, H.row, H.col
    # edge_index = torch.tensor(np.stack([src, tar]).tolist(), dtype=torch.long)
    # edge_index = torch.tensor(np.stack([src, tar]), dtype=torch.long)

    return H.cuda()
