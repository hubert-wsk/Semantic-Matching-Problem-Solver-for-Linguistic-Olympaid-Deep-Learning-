from g_model import gProblem_T
from utils.wmdl import Composer
from utils.exft import exFTModel
#import numpy as np
import torch
import argparse as aps
import pickle as pckl
import random as rdm
import networkx as nx
from typing import Union, Tuple, List
from tqdm import tqdm


pser = aps.ArgumentParser(
    prog = "datagen",
    description = "generate data of training geometric model...",
)
pser.add_argument("dst")
pser.add_argument("num", type = int)
pser.add_argument("-s", "--skeleton", type = int)
pser.add_argument("-l", "--leaves", type = int)
pser.add_argument("--smax", type = int, default = 4, help = "max degree of skeleton nodes")
pser.add_argument("--smin", type = int, default = 2, help = "min degree of skeleton nodes")
pser.add_argument("-t", "--tri", type = int, default = 2, help = "triple..")
pser.add_argument("--sg", type = int, default = 1, help = "'single' token spawns a phrase with itself at R.")

# (...)
comp_path = r"..."
ft_path = r"./ft_model/cczh128.bin"

def _gen_pairgraph(arg: aps.Namespace):
    smin, smax, skl = arg.smin, arg.smax, arg.skeleton; lv: int = arg.leaves 
    _skl_deg = list(range(smin, smax + 1))
    skeleton = [rdm.choice(_skl_deg) for _ in range(skl)]
    deg_seq = skeleton + [1] * lv
    if sum(deg_seq) % 2 == 1: deg_seq.append(1)     # *
    g = nx.Graph( nx.configuration_model(deg_seq) )
    g.remove_edges_from( nx.selfloop_edges(g) )     # (...)
    return g, deg_seq

Comb_R = List[Tuple[int, ...]]    # -> which tokens combine each right phrase
TP_Edge = List[Tuple[int, int]]

def get_tp_graph(arg: aps.Namespace) -> Tuple[TP_Edge, Comb_R, int]:
    """we just need keys of G as int here..."""
    g, _ = _gen_pairgraph(arg)
    _nd: List[int] = list(g)
    combr: Comb_R = []
    for i in g.edges: combr.append(i)

    for _ in range( rdm.randint(0, arg.tri) ):
        combr.append( tuple(rdm.choices(_nd, k = 3)) )
    for i in rdm.choices(_nd, k = arg.sg):
        combr.append( (i, ) )
    
    tp_edge: TP_Edge = []
    for i, tks in enumerate(combr):
        for j in tks: tp_edge.append( (j, i, ) )
    
    return tp_edge, combr, len(_nd)

_wordc_low, _wordc_upp = 200, 10_000    # *
with open(comp_path, mode = "rb") as f:
    comp: Composer = pckl.load(f)
_ftm = exFTModel(ft_path, _words_retain = 20_000)
wordvecs_ = _ftm.vectors[_wordc_low, _wordc_upp]

def get_wordvec(arg: aps.Namespace, combr: Comb_R, n_tok: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """ -> token_vecs & phrase_vecs"""
    tok_vecs = torch.from_numpy( rdm.choices(wordvecs_, k = n_tok) )
    phr_vecs: List[torch.Tensor] = []
    for tks in combr:
        phr_vecs.append( comp.forward_( * (tok_vecs[i] for i in tks) ).detach() )
    return tok_vecs, torch.stack(phr_vecs)

def main():
    args = pser.parse_args()
    res: List[Tuple[gProblem_T, torch.Tensor]] = []
    for _ in tqdm(range(args.num)):
        tp_edge, combr, n = get_tp_graph(args)
        tok_vec, phr_vec = get_wordvec(args, combr, n)
        res.append( (
            gProblem_T(n, phr_vec, tp_edge),
            tok_vec,
        ) )
    with open(args.dst, "wb") as f: pckl.dump(res, f)
    
if __name__ == "__main__": main()