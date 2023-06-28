import torch
from torch import nn
from torch.nn import functional as F

from layers import Attention, HGNN_conv

class HGCN(nn.Module):
    def __init__(self, nfeat, nhid, out, dropout):
        super(HGCN, self).__init__()
        self.dropout = dropout
        self.hgc1 = HGNN_conv(nfeat, nhid)
        self.hgc2 = HGNN_conv(nhid, out)

    def forward(self, x, G):
        x = F.relu(self.hgc1(x, G))
        x = F.dropout(x, self.dropout)
        x = self.hgc2(x, G)
        return x

class HFNII(nn.Module):
    def __init__(self, args, nfeat, nclass):
        super(HFNII, self).__init__()

        self.args = args
        self.dropout = self.args.dropout
        self.nhid1 = self.args.hid1
        self.nhid2 = self.args.hid2

        if self.args.mlp:
            self.SGCN1 = HGCN(nfeat, self.nhid1, self.nhid2, self.dropout)
            self.SGCN2 = HGCN(nfeat, self.nhid1, self.nhid2, self.dropout)
            self.SGCN3 = HGCN(nfeat, self.nhid1, self.nhid2, self.dropout)
            self.SGCN4 = HGCN(nfeat, self.nhid1, self.nhid2, self.dropout)
        else:
            self.SGCN1 = HGCN(nfeat, self.nhid1, nclass, self.dropout)
            self.SGCN2 = HGCN(nfeat, self.nhid1, nclass, self.dropout)
            self.SGCN3 = HGCN(nfeat, self.nhid1, nclass, self.dropout)
            self.SGCN4 = HGCN(nfeat, self.nhid1, nclass, self.dropout)

        self.attention = Attention(self.nhid2)
        self.tanh = nn.Tanh()

        self.MLP = nn.Sequential(
            nn.Linear(self.nhid2, nclass),
            nn.LogSoftmax(dim=1)
        )

    def forward(self, x, sadj, fadj, diff, diff2):
        emb1 = self.SGCN1(x, sadj)  # structure hypergraph
        emb2 = self.SGCN2(x, fadj)  # feature hypergraph
        emb3 = self.SGCN3(x, diff)  # diffusion
        emb4 = self.SGCN4(x, diff2)  # diffusion

        # attention
        emb = torch.stack([emb1, emb2, emb3, emb4], dim=1)
        emb, att = self.attention(emb)

        if self.args.mlp:
            emb = self.MLP(emb)
        return emb

class HFNIII(nn.Module):
    def __init__(self, args, nfeat, nclass):
        super(HFNIII, self).__init__()

        self.args = args
        self.dropout = self.args.dropout
        self.nhid1 = self.args.hid1
        self.nhid2 = self.args.hid2

        if self.args.mlp:
            self.SGCN = HGCN(nfeat, self.nhid1, self.nhid2, self.dropout)
        else:
            self.SGCN = HGCN(nfeat, self.nhid1, nclass, self.dropout)

        self.attention = Attention(self.nhid2)
        self.tanh = nn.Tanh()

        self.MLP = nn.Sequential(
            nn.Linear(self.nhid2, nclass),
            nn.LogSoftmax(dim=1)
        )

    def forward(self, x, adj, index):
        #emb1 = self.SGCN(x, sadj)  # structure hypergraph
        #emb2 = self.SGCN(x, fadj)  # feature hypergraph
        #emb3 = self.SGCN(x, diff)  # diffusion
        emb_ = [self.SGCN(x, A) for A in adj[index[0]:index[1]]]

        # attention
        emb = sum(emb_) / len(emb_)
        
        if self.args.mlp:
            emb = self.MLP(emb)
        return emb

class HFN(nn.Module):
    def __init__(self, args, nfeat, nclass):
        super(HFN, self).__init__()

        self.args = args
        self.dropout = self.args.dropout
        self.nhid1 = self.args.hid1
        self.nhid2 = self.args.hid2

        if self.args.mlp:
            self.SGCN = HGCN(nfeat, self.nhid1, self.nhid2, self.dropout)
        else:
            self.SGCN = HGCN(nfeat, self.nhid1, nclass, self.dropout)

        self.attention = Attention(self.nhid2)
        self.tanh = nn.Tanh()

        self.MLP = nn.Sequential(
            nn.Linear(self.nhid2, nclass),
            nn.LogSoftmax(dim=1)
        )

    def forward(self, x, adj, index):
        #emb1 = self.SGCN(x, sadj)  # structure hypergraph
        #emb2 = self.SGCN(x, fadj)  # feature hypergraph
        #emb3 = self.SGCN(x, diff)  # diffusion
        emb_ = [self.SGCN(x, A) for A in adj[index[0]:index[1]]]

        # attention
        emb = torch.stack(emb_, dim=1)
        emb, att = self.attention(emb)

        if self.args.mlp:
            emb = self.MLP(emb)
        return emb
