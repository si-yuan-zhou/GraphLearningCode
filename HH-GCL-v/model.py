import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from UniGNN import UniGNN

from torch_scatter import scatter
from torch_geometric.utils import softmax

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
    def __init__(self, encoderI: Encoder, encoderII: UniGNN,
                 num_hidden: int, num_proj_hidden: int, tau: float = 0.5,
                 heads=1, dropout=0., negative_slope=0.2):

        super(Model, self).__init__()
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

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        
        h_gnn = self.encoderI(x, edge_index)
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
        X = Xv 
        X = X.view(N, H * C)

        return X

    def node_sub_graph_I(self, z: torch.Tensor, V, E, mean = True):
        # X,  N * nclass
        # Xe, E * nclass
        # Xg, 1 * nclass
        # sg, N * nclass
        
        # InfoNCE
        node, super_edge, graph = z['node'], z['sub'], z['graph']
        subgraph_pos = self.get_subgraph(node, super_edge, V, E)

        # get negative sample
        shuf_index_row = torch.randperm(subgraph_pos.size(0))
        shuf_index_col = torch.randperm(subgraph_pos.size(1))
        subgraph_neg = subgraph_pos[:, shuf_index_col]
        subgraph_neg = subgraph_neg[shuf_index_row]

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


    

    def lossI(self, z1: torch.Tensor, z2: torch.Tensor,
             mean: bool = True, batch_size: int = 0):
        
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

        ret = (l1 + l2) * 0.5
        ret = ret.mean() if mean else ret.sum()

        return ret

    def lossII(self, z1: torch.Tensor, z2: torch.Tensor,
               mean: bool = True, batch_size: int = 0, alpha: float = 0.5):
        
        # node-node & node-graph
        channelI = self.lossI(z1, z2, batch_size)
        channelII= self.node_graph(z2)

        return alpha * channelI + (1 - alpha) * channelII

    def lossIII(self, z1: torch.Tensor, z2: torch.Tensor, V, E,
                mean: bool = True, batch_size: int = 0, alpha: float = 0.5):
        
        # node-node & node-subgraph & sungraph-graph
        channelI = self.lossI(z1, z2, batch_size)
        channelII= self.node_sub_graph_I(z2, V, E)

        return alpha * channelI + (1 - alpha) * channelII


    def node_graph(self, z: torch.Tensor):
        # 参考三级对比损失中效果最好的，去掉中间层，作为消融实验
        node, subgraph, graph = z['node'], z['sub'], z['graph']
        
        # write the logic here
        lossVal = 0
        return lossVal