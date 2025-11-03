import torch_geometric as ptg
import torch_geometric.nn as gnn
from torch_geometric.data import HeteroData
from torch_geometric.typing import EdgeType
import torch
import torch.nn as nn
from typing import NamedTuple, List, Any, Tuple, Iterable, Dict


# edge: token -> phrase; ...~
#gProblem_T = NamedTuple(, token_num = int, ...)
class gProblem_T(NamedTuple):
    token_num: int
    phrase_emb: torch.Tensor    # [P, dim]
    tp_edge: Iterable[Tuple[int, int]]

def build_ggraph(p: gProblem_T):
    g = HeteroData()
    tk, (phr, dim) = p.token_num, p.phrase_emb.shape
    tp_edge = torch.tensor(p.tp_edge).T

    g["token"].x = torch.zeros(tk, 1)
    g["phrase"].x = p.phrase_emb

    g["token", "occurs in", "phrase"].edge_index = tp_edge
    g["phrase", "with", "token"].edge_index = tp_edge[ [1, 0] ]

    return g

class TokenSemantics(nn.Module):
    def __init__(self, hidden, dim, dropout = 0.1):
        super().__init__()
        self.token_emb = nn.Linear(1, hidden)

        # fuck. (...)
        # add multi-attention ?
        convs: Dict[EdgeType, gnn.MessagePassing] = {
            ("token", "occurs in", "phrase"): gnn.TransformerConv( (hidden, -1) , hidden, dropout = dropout, beta = True),
            ("phrase", "with", "token"):      gnn.TransformerConv( (-1, hidden) , hidden, dropout = dropout, beta = True),
        }
        self.conv1 = gnn.HeteroConv(convs, aggr = "sum")
        self.conv2 = gnn.HeteroConv(convs, aggr = "sum")

        self.leaky_ = nn.LeakyReLU()
        self.token_head = nn.Linear(hidden, dim)
    
    def forward(self, x: HeteroData) -> torch.Tensor:
        h = x.collect("x").copy()   # x_dict ?
        _e: Dict[EdgeType, torch.Tensor] = x.edge_index_dict    # ...?
        h["token"] = self.token_emb(h["token"])
        h = self.conv1(h, _e)
        h = {k: self.leaky_(t) for k, t in h}
        h = self.conv2(h, _e)
        return self.token_head( h["token"] )

