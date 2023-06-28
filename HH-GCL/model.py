import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from UniGNN import UniGNN

from torch_scatter import scatter
from torch_geometric.utils import softmax

class BayesianPersonalizedRank(nn.Module):
    def __init__(self, n_h) -> None:
        super(BayesianPersonalizedRank, self).__init__()
        self.f_k = nn.Bilinear(n_h, n_h, 1)

        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, nn.Bilinear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, node, subgraph_pos, subgraph_neg):
        if node.size(0) == 1:
            # graph
            node = node.expand_as(subgraph_pos)

        sc_1 = self.f_k(node, subgraph_pos)
        sc_2 = self.f_k(node, subgraph_neg)

        logits = torch.log(torch.sigmoid(sc_1 - sc_2))
        logits = logits.mean()
        return logits

class Discriminator(nn.Module):
    def __init__(self, n_h):
        super(Discriminator, self).__init__()
        self.f_k = nn.Bilinear(n_h, n_h, 1)

        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, nn.Bilinear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, node, subgraph_pos, subgraph_neg):
        if node.size(0) == 1:
            # graph
            node = node.expand_as(subgraph_pos)

        sc_1 = self.f_k(node, subgraph_pos)
        sc_2 = self.f_k(node, subgraph_neg)

        logits = torch.cat((sc_1, sc_2), 1)

        return logits

class RVGCN(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(RVGCN, self).__init__()

        self.W = nn.Linear(in_channels, out_channels)
        torch.nn.init.xavier_uniform_(self.W.weight.data)

    def norm(self, A):
        bin_A = torch.where(A == 0, 0, 1)
        D_v = torch.sum(bin_A, 1)
        D_v_diag = torch.diag(torch.pow(D_v, -0.5))
        bin_A = torch.tensor(bin_A,dtype=torch.float32)
        Theta = D_v_diag @ bin_A @ D_v_diag
        
        return Theta
    
    def forward(self, X, A):
        X = self.W(X)
        Theta = self.norm(A)
        X = torch.matmul(Theta, X)

        return X

class Encoder(torch.nn.Module):
    def __init__(self, in_channels: int, out_channels: int, activation,
                 base_model=GCNConv, k: int = 2):
        super(Encoder, self).__init__()
        self.base_model = base_model

        assert k >= 2
        self.k = k
        self.conv = [base_model(in_channels, 2 * out_channels)]
        for _ in range(1, k-1):
            self.conv.append(base_model(2 * out_channels, 2 * out_channels))
        self.conv.append(base_model(2 * out_channels, out_channels))
        self.conv = nn.ModuleList(self.conv)

        self.activation = activation

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor):
        for i in range(self.k):
            x = self.activation(self.conv[i](x, edge_index))
        return x


class Model(torch.nn.Module):
    def __init__(self, args, encoderI: Encoder, encoderII: UniGNN,
                 num_hidden: int, num_proj_hidden: int, tau: float = 0.5,
                 heads: int = 1, dropout=0., negative_slope=0.2):

        super(Model, self).__init__()
        self.args = args
        self.encoderI: Encoder = encoderI
        self.encoderII: UniGNN = encoderII
        self.tau: float = tau

        self.fc1 = torch.nn.Linear(num_hidden, num_proj_hidden)
        self.fc2 = torch.nn.Linear(num_proj_hidden, num_hidden)

        self.att_v = nn.Parameter(torch.Tensor(1, heads, num_hidden))
        self.att_e = nn.Parameter(torch.Tensor(1, heads, num_hidden))
        self.heads = heads
        self.out_channels = num_hidden
        self.attn_drop  = nn.Dropout(dropout)
        self.leaky_relu = nn.LeakyReLU(negative_slope)

    def forward(self, base_model, x: torch.Tensor, edge_index: torch.Tensor, A) -> torch.Tensor:
        # if base_model == 'GCNConv':
        # if isinstance(base_model, GCNConv):
        if (base_model is GCNConv):
            h_gnn = self.encoderI(x, edge_index)
        else:
            h_gnn = self.encoderI(x, A)
          
        h_node, h_subgraph, h_graph = self.encoderII(x)

        return h_gnn, {"node": h_node, "sub": h_subgraph, "graph": h_graph}



    ##################### Contrastive Loss Partition ##############################

    def projection(self, z: torch.Tensor) -> torch.Tensor:
        z = F.elu(self.fc1(z))
        return self.fc2(z)

    def sim(self, z1: torch.Tensor, z2: torch.Tensor):
        z1 = F.normalize(z1)
        z2 = F.normalize(z2)
        return torch.mm(z1, z2.t())

    def semi_loss(self, z1: torch.Tensor, z2: torch.Tensor):
        f = lambda x: torch.exp(x / self.tau)
        refl_sim = f(self.sim(z1, z1))
        between_sim = f(self.sim(z1, z2))

        return -torch.log(
            between_sim.diag()
            / (between_sim.sum(1) + refl_sim.sum(1) - refl_sim.diag()))

    def batched_semi_loss(self, z1: torch.Tensor, z2: torch.Tensor,
                          batch_size: int):
        # Space complexity: O(BN) (semi_loss: O(N^2))
        device = z1.device
        num_nodes = z1.size(0)
        num_batches = (num_nodes - 1) // batch_size + 1
        f = lambda x: torch.exp(x / self.tau)
        indices = torch.arange(0, num_nodes).to(device)
        losses = []

        for i in range(num_batches):
            mask = indices[i * batch_size:(i + 1) * batch_size]
            refl_sim = f(self.sim(z1[mask], z1))  # [B, N]
            between_sim = f(self.sim(z1[mask], z2))  # [B, N]

            losses.append(-torch.log(
                between_sim[:, i * batch_size:(i + 1) * batch_size].diag()
                / (refl_sim.sum(1) + between_sim.sum(1)
                   - refl_sim[:, i * batch_size:(i + 1) * batch_size].diag())))

        return torch.cat(losses)


    def get_subgraph(self, X, Xe, vertex, edges):
        H, C, N = self.heads, self.out_channels, X.shape[0]
        Xe = Xe.view(-1, H, C)

        # Xe 的最后一个维度要与self.att_e的最后一个维度一致
        alpha_e = (Xe * self.att_e).sum(-1) # [E, H, 1]
        a_ev = alpha_e[edges]
        alpha = a_ev # Recommed to use this
        alpha = self.leaky_relu(alpha)
        alpha = softmax(alpha, vertex, num_nodes=N)
        alpha = self.attn_drop( alpha )
        alpha = alpha.unsqueeze(-1)


        Xev = Xe[edges] # [nnz, H, C]
        Xev = Xev * alpha 
        Xv = scatter(Xev, vertex, dim=0, reduce='sum', dim_size=N) # [N, H, C]
        X = Xv.view(N, H * C)

        return X

    def get_info(self, z, V, E):
        node, super_edge, graph = z['node'], z['sub'], z['graph']
        subgraph_pos = self.get_subgraph(node, super_edge, V, E)

        # get negative sample
        shuf_index_row = torch.randperm(subgraph_pos.size(0))
        shuf_index_col = torch.randperm(subgraph_pos.size(1))
        subgraph_neg = subgraph_pos[:, shuf_index_col]
        subgraph_neg = subgraph_neg[shuf_index_row]

        return node, subgraph_pos, subgraph_neg, graph

    def node_sub_graph_I(self, z, V, E, mean = True):
        # X,  N * nclass
        # Xe, E * nclass, N == E
        # Xg, 1 * nclass
        # sg, N * nclass
        
        # InfoNCE
        node, subgraph_pos, subgraph_neg, graph = self.get_info(z, V, E)

        f = lambda x: torch.exp(x / self.tau)
        n2pos = f(self.sim(node, subgraph_pos))
        n2neg = f(self.sim(node, subgraph_neg))

        g2pos = f(self.sim(subgraph_pos, graph))
        g2neg = f(self.sim(subgraph_neg, graph))

        l1 = -torch.log(
            n2pos.diag() / (n2pos.diag() + n2neg.diag())    # n2pos.diag() / (n2pos.diag() + n2neg.sum(1))
        )
        l2 = -torch.log(
            g2pos.view(1, -1) / (g2pos.view(1, -1) + g2neg.view(1, -1))
        )

        ret = l1 + l2
        ret = ret.mean() if mean else ret.sum()
        return ret
   
    def node_sub_graph_II(self, z, V, E):
        # Jensen_Shannon
        node, subgraph_pos, subgraph_neg, graph = self.get_info(z, V, E)

        disc = Discriminator(subgraph_pos.size(1)).to(self.args.device)
        logits_I  = disc(node, subgraph_pos, subgraph_neg)
        logits_II = disc(graph, subgraph_pos, subgraph_neg)
        
        one = torch.ones(logits_I.size(0), logits_I.size(1)//2)
        zero = torch.zeros(logits_I.size(0), logits_I.size(1)//2)
        label = torch.cat((one, zero), 1).to(self.args.device)
        
        criterion = nn.BCEWithLogitsLoss().to(self.args.device)
        loss_I = criterion(logits_I, label)
        loss_II = criterion(logits_II, label)
        loss = (loss_I + loss_II) / 2
        return loss

    def node_sub_graph_III(self, z, V, E):
        # TripletLoss
        node, subgraph_pos, subgraph_neg, graph = self.get_info(z, V, E)

        graph = graph.expand_as(subgraph_pos)
        logits_np = torch.sigmoid(torch.sum(node  * subgraph_pos, dim = -1))
        logits_nn = torch.sigmoid(torch.sum(node  * subgraph_neg, dim = -1))
        logits_gp = torch.sigmoid(torch.sum(graph * subgraph_pos, dim = -1))
        logits_gn = torch.sigmoid(torch.sum(graph * subgraph_neg, dim = -1))
        ones = torch.ones(logits_np.size(0)).to(self.args.device)

        marginLoss = nn.MarginRankingLoss().to(self.args.device)
        loss_I = marginLoss(logits_np, logits_nn, ones)
        loss_II = marginLoss(logits_gp, logits_gn, ones)
        loss = (loss_I + loss_II) / 2

        return loss

    def node_sub_graph_IV(self, z, V, E):
        # Bayesian Personalized Ranking
        node, subgraph_pos, subgraph_neg, graph = self.get_info(z, V, E)

        bpr = BayesianPersonalizedRank(subgraph_pos.size(1)).to(self.args.device)
        loss_I = bpr(node, subgraph_pos, subgraph_neg)
        loss_II = bpr(graph, subgraph_pos, subgraph_neg)
        loss = (loss_I + loss_II) / 2
        return loss

    def lossI(self, z1: torch.Tensor, z2: torch.Tensor,
             mean: bool = True, batch_size: int = 0, alpha: float = 0.5):
        
        # node-node only
        z2 = z2['node']
        h1 = self.projection(z1)
        h2 = self.projection(z2)

        if batch_size == 0:
            l1 = self.semi_loss(h1, h2)
            l2 = self.semi_loss(h2, h1)
        else:
            l1 = self.batched_semi_loss(h1, h2, batch_size)
            l2 = self.batched_semi_loss(h2, h1, batch_size)

        ret = (l1 + l2) * alpha
        ret = ret.mean() if mean else ret.sum()

        return ret

    def lossII(self, z1: torch.Tensor, z2: torch.Tensor,
               mean: bool = True, batch_size: int = 0, alpha: float = 0.5):
        
        # node-node & node-graph
        channelI = self.lossI(z1, z2, batch_size)
        channelII= self.node_graph(z2)

        return alpha * channelI + (1 - alpha) * channelII

    def lossIII(self, z1: torch.Tensor, z2: torch.Tensor, V, E, name = 'NCE',
                batch_size: int = 0, alpha: float = 0.5):
        # node-node & node-subgraph & sungraph-graph

        hierarchical = {
            'NCE': self.node_sub_graph_I,
            'JSD': self.node_sub_graph_II,
            'TMR': self.node_sub_graph_III,
            'BPR': self.node_sub_graph_IV
        }
        node_sub_graph = hierarchical[name]

        channelI = self.lossI(z1, z2, batch_size)
        channelII= node_sub_graph(z2, V, E)

        return alpha * channelI + (1 - alpha) * channelII


    def node_graph(self, z: torch.Tensor):
        # 参考三级对比损失中效果最好的，去掉中间层，作为消融实验
        node, subgraph, graph = z['node'], z['sub'], z['graph']
        
        # write the logic here
        lossVal = 0
        return lossVal
