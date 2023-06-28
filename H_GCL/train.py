import argparse
import os.path as osp
import random
import yaml
from yaml import SafeLoader

import torch
import torch.nn.functional as F
import torch.nn as nn
from torch_geometric.nn import GCNConv

from UniGNN import UniGNN
from model import Encoder, Model
from util import label_classification, get_dataset, build_hypergrah


def train(model: Model, x, edge_index):
    model.train()
    optimizer.zero_grad()

    z1, z2 = model(x, edge_index)
    loss = model.loss(z1, z2, batch_size=0)

    loss.backward()
    optimizer.step()

    return loss.item()


def test(model: Model, x, edge_index, y):
    model.eval()
    
    z1, z2 = model(x, edge_index)
    label_classification((z1 + z2) / 2, y, ratio=0.1)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='DBLP')
    parser.add_argument('--gpu_id', type=int, default=0)
    parser.add_argument('--config', type=str, default='config.yaml')
    parser.add_argument('--use-norm', action="store_true", help='use norm in the final layer')
    parser.add_argument('--dropout', type=float, default=0.6, help='dropout probability after UniConv layer')
    parser.add_argument('--input-drop', type=float, default=0.6, help='dropout probability for input layer')
    parser.add_argument('--attn-drop', type=float, default=0.6, help='dropout probability for attentions in UniGATConv')
    parser.add_argument('--model-name', type=str, default='UniSAGE', help='UniGNN Model(UniGCN, UniGAT, UniGIN, UniSAGE...)') 
    parser.add_argument('--first-aggregate', type=str, default='mean', help='aggregation for hyperedge h_e: max, sum, mean')
    parser.add_argument('--second-aggregate', type=str, default='sum', help='aggregation for node x_i: max, sum, mean')
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
    
    torch.manual_seed(config['seed'])
    random.seed(12345)  


    path = osp.join(osp.expanduser('~'), 'datasets', args.dataset)
    dataset = get_dataset(path, args.dataset)
    data = dataset[0]

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data = data.to(device)

    encoderI = Encoder(dataset.num_features, num_hidden, activation,
                      base_model=base_model, k=num_layers).to(device)

    args.activation = config['activation']
    args, V, E = build_hypergrah(args, dataset[0])
    encoderII = UniGNN(args, dataset.num_features, num_hidden, num_hidden, num_layers, num_head, V.to(device), E.to(device)).to(device)

    model = Model(encoderI, encoderII, num_hidden, num_proj_hidden, tau).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)


    for epoch in range(1, num_epochs + 1):
        loss = train(model, data.x, data.edge_index)
        print(f' Epoch={epoch:03d}, loss={loss:.4f}')


    print("=== Final ===")
    test(model, data.x, data.edge_index, data.y)
