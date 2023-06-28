import argparse
import os.path as osp
import random
import yaml
from yaml import SafeLoader

import torch
import torch.nn.functional as F
import torch.nn as nn
from torch_geometric.nn import GCNConv, global_mean_pool

from UniGNN import UniGNN
from model import Encoder, Model, RVGCN
from util import label_classification, get_dataset, build_hypergraph, build_hypergraph_I, build_hypergraph_II, build_vanilla_graph


def train(model: Model, base_model, x, edge_index, A, V, E, name='JSD'):
    model.train()
    optimizer.zero_grad()

    z1, z2 = model(base_model, x, edge_index, A)
    # loss = model.lossI(z1, z2, batch_size=0)    # node-node only
    # loss = model.lossII(z1, z2, batch_size=0)   # node-node & node-graph
    loss = model.lossIII(z1, z2, V, E, name = name)   # node-node & node-subgraph & sungraph-graph

    loss.backward()
    optimizer.step()

    return loss.item()


def test(model: Model, base_model, x, edge_index, y, A):
    model.eval()
    
    z1, z2 = model(base_model, x, edge_index, A)

    print("---------- common graph node result ---------------")
    label_classification(z1, None, y, ratio=0.1)
    print("---------- hyper graph node result ----------------")
    label_classification(z2['node'], None, y, ratio=0.1)
    print("---------- average graph node result --------------")
    label_classification((z1 + z2['node']) / 2, None, y, ratio=0.1)
    print("---------- sum graph node result ------------------")
    label_classification((z1 + z2['node']), None, y, ratio=0.1)
    # print("---------- fusion graph node result use mlp---------------")
    # label_classification(z1, z2['node'], y, ratio=0.1)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='DBLP')
    parser.add_argument('--gpu_id', type=int, default=0)
    parser.add_argument('--config', type=str, default='config.yaml')
    parser.add_argument('--base_model', type=str, default='RVGCN', help='real value GCN')
    parser.add_argument('--use-norm', action="store_true", help='use norm in the final layer')
    parser.add_argument('--dropout', type=float, default=0.6, help='dropout probability after UniConv layer')
    parser.add_argument('--input-drop', type=float, default=0.6, help='dropout probability for input layer')
    parser.add_argument('--attn-drop', type=float, default=0.6, help='dropout probability for attentions in UniGATConv')
    parser.add_argument('--model-name', type=str, default='UniSAGE', help='UniGNN Model(UniGCN, UniGAT, UniGIN, UniSAGE...)') 
    parser.add_argument('--first-aggregate', type=str, default='mean', help='aggregation for hyperedge h_e: max, sum, mean')
    parser.add_argument('--second-aggregate', type=str, default='sum', help='aggregation for node x_i: max, sum, mean')
    parser.add_argument('--name', type=str, default='JSD', help='Contrastive Loss(NCE, JSD, TMR, BPR...)') 
    parser.add_argument('--alpha', type=float, default=0.8, help='threshold of similarity') 
    args = parser.parse_args()

    assert args.gpu_id in range(0, 8)
    torch.cuda.set_device(args.gpu_id)

    config = yaml.load(open(args.config), Loader=SafeLoader)[args.dataset]  
    
    activation = ({'relu': F.relu, 'prelu': nn.PReLU()})[config['activation']]
    base_model = ({'GCNConv': GCNConv})[config['base_model']]
    num_proj_hidden = config['num_proj_hidden']
    learning_rate = config['learning_rate']
    weight_decay = config['weight_decay']
    num_epochs = config['num_epochs']
    num_hidden = config['num_hidden']
    num_layers = config['num_layers']
    num_head = config['num_head']
    tau = config['tau']

    args.activation = activation
    args.readout = global_mean_pool
    if args.base_model == 'RVGCN':
        base_model = RVGCN
    
    torch.manual_seed(config['seed'])
    random.seed(12345)  


    path = osp.join(osp.expanduser('~'), 'datasets', args.dataset)
    dataset = get_dataset(path, args.dataset)
    data = dataset[0]

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data = data.to(device)
    args.device = device

    A = build_vanilla_graph(data)
    # out of memeory in Pubmed and DBLP
    args, V, E = build_hypergraph_I(args, data, args.alpha)
    #args, V, E = build_hypergraph_I(args, data, args.alpha)
    #args, V, E = build_hypergraph_II(args, data, args.alpha)

    # 如果动态更新超图结构时，需要放在训练的for循环里，且不要用GCN,修改时注意缩进
    # for epoch in range(1, num_epochs + 1):
    # args, V, E = build_hypergraph(args, data, args.bin)

    encoderI = Encoder(dataset.num_features, num_hidden, activation, base_model=base_model, k=num_layers).to(device)
    encoderII = UniGNN(args, dataset.num_features, num_hidden, num_hidden, num_layers, num_head, V, E).to(device)

    model = Model(args, encoderI, encoderII, num_hidden, num_proj_hidden, tau).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)


    for epoch in range(1, num_epochs + 1):
        loss = train(model, base_model, data.x, data.edge_index, A, V, E, args.name)
        print(f' Epoch={epoch:03d}, loss={loss:.4f}')


    print("=== Final ===")
    test(model, base_model, data.x, data.edge_index, data.y, A)
