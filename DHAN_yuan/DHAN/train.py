from __future__ import division
from __future__ import print_function

import random
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import f1_score
from torch.autograd import Variable
from tqdm import tqdm

from models import HFN, HFNIII
from process import load_data, generator, accuracy, training_split, normalize_h1
from process import get_mask, get_mask_I, get_mask_II, get_mask_III, get_mask_IV

parser = argparse.ArgumentParser()
parser.add_argument('--no-cuda', action='store_true', default=False, help='Disables CUDA training.')
parser.add_argument('--fastmode', action='store_true', default=False, help='Validate during training pass.')
parser.add_argument('--dataset', default='cora', help='name of dataset. ')
parser.add_argument('--model', default='HFN', help='name of model. ')

parser.add_argument('--seed', type=int, default=56, help='Random seed.')
parser.add_argument('--epochs', type=int, default=1000, help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.005, help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=1e-4, help='Weight decay (L2 loss on parameters).')
parser.add_argument('--dropout', type=float, default=0.5, help='Dropout rate (1 - keep probability).')
parser.add_argument('--alpha', type=float, default=0.2, help='Alpha for the leaky_relu.')
parser.add_argument('--patience', type=int, default=100, help='Patience')

parser.add_argument("--hid1", type=int, default=128, help=" Hidden layer 1.")
parser.add_argument("--hid2", type=int, default=64, help=" Hidden layer 2.")

parser.add_argument("--mlp", type=bool, default=True, help=" Hidden layer 1.")

parser.add_argument("--test-size", type=float, default=0.2, help="Ratio of testing samples. Default is 0.2")
parser.add_argument("--index", type=int, nargs=2, default=[0,2], help="number of views")
parser.add_argument("--k", type=int, default=3, help="K of KNN")
parser.add_argument("--p", type=int, default=5, help="drop size")
parser.add_argument("--m", type=int, default=5, help="index of mask")

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

features, labels, node_num = load_data(args.dataset)
train_nodes, test_nodes = training_split(args, features.shape[0])
H1, H2, H3 = generator(args.dataset, features, node_num, args.k)

if args.m == 0:
    H1 = get_mask(H1, args.p)
    H2 = get_mask(H2, args.p)
    H3 = get_mask(H3, args.p)
elif args.m == 1:
    H1 = get_mask_I(H1, args.p)
    H2 = get_mask_I(H2, args.p)
    H3 = get_mask_I(H3, args.p)
elif args.m == 2:
    H1 = get_mask_II(H1, args.p)
    H2 = get_mask_II(H2, args.p)
    H3 = get_mask_II(H3, args.p)
elif args.m == 3:
    H1 = get_mask_III(H1, args.p)
    H2 = get_mask_III(H2, args.p)
    H3 = get_mask_III(H3, args.p)
elif args.m == 4:
    H1 = get_mask_IV(H1, args.p)
    H2 = get_mask_IV(H2, args.p)
    H3 = get_mask_IV(H3, args.p)

# Model and optimizer
H1 = normalize_h1(H1)
H2 = normalize_h1(H2)
H3 = normalize_h1(H3)

if args.model == "HFN":
    model = HFN(args=args, nfeat=features.shape[1], nclass=int(labels.max()) + 1)
else:
    model = HFNIII(args=args, nfeat=features.shape[1], nclass=int(labels.max()) + 1)

optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

if args.cuda:
    model.cuda()
    features = features.cuda()
    H1 = H1.cuda()
    H2 = H2.cuda()
    H3 = H3.cuda()
    labels = labels.cuda()
    train_nodes = train_nodes.cuda()
    test_nodes = test_nodes.cuda()

#    features, labels, H1, H2, H3, H4 = Variable(features), Variable(labels), Variable(H1), Variable(H2), Variable(H3), Variable(H4)

def train(model, args):
    model.train()

    for epoch in tqdm(range(args.epochs)):
        optimizer.zero_grad()
        #output = model(features, H1, H2, H3)
        output = model(features, [H1, H2, H3], args.index)
        loss = F.nll_loss(output[train_nodes], labels[train_nodes])
        acc = accuracy(output[train_nodes], labels[train_nodes])
        print("loss: ", float(loss), ", acc: ", float(acc))

        loss.backward()
        optimizer.step()

    return model


def test(model):
    model.eval()
    #output = model(features, H1, H2, H3)
    output = model(features, [H1, H2, H3], args.index)
    acc = accuracy(output[test_nodes], labels[test_nodes])

    label_max = []
    for idx in test_nodes:
        label_max.append(torch.argmax(output[idx]).item())
    labelcpu = labels[test_nodes].data.cpu()
    macro_f1 = f1_score(labelcpu, label_max, average='macro')
    micro_f1 = f1_score(labelcpu, label_max, average='micro')

    return acc, macro_f1, micro_f1


model = train(model, args)
acc, f1, f2 = test(model)
print("accuracy:", float(acc), ", error:", float(100 * (1 - acc)), ", f1:", float(f1), ", f2:", float(f2))

#with open("rst.txt",'a') as f:
#    f.write(f"dataset: {args.dataset}, size: {args.test_size}, k = {args.k}, acc:{float(acc):.5f}")
#    f.write("\r\n")
