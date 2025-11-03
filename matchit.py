from __future__ import annotations
import numpy as np
import networkx as nx
#import fasttext as ft
from utils.exft import exFTModel
from sklearn.metrics.pairwise import cosine_similarity
from itertools import permutations
from typing import (
    Tuple,
    NewType,
    Iterable,
    Dict,
    Optional,
    List,
    Any,
    Iterator,
    #Union,
    #NamedTuple,
)


class MatchingProblem:
    def __init__(self, left: nx.Graph, right: Iterable[str]):
        self.right: List[str] = list(right)
        self.left_grph, self.size, self.left_keys = left, len(left), list(left.nodes())
        self._rvec: Optional[np.ndarray]
        self.rrel: Optional[np.ndarray]

    @staticmethod
    def _pre_process(p: MatchingProblem, ft_model: exFTModel):
        p._rvec = np.stack(
            list(
                [ ft_model.get_word_vec(i) for i in p.right]
            )
        )
        p.rrel = cosine_similarity(p._rvec, p._rvec)
        return p.rrel, p._rvec

Solution = NewType("Solution", Dict[Any, int])

alpha = 1.4
k1 = 0.15
k2 = 0.1
def _evaluate(l, r):
    dom = l * k2 * np.exp( alpha * (1 - r) )
    sec = (1 - l) * k1 * r
    loss = np.log2( np.mean(dom + sec) )
    return loss

def evaluate(p: MatchingProblem, sl: Solution) -> float:
    assert p.rrel, "[evaluating]: pre-process the problem first"
    losses = []

    for u in p.left_keys:
        r_id, edgel, edger = sl[u], [], []

        for v in p.left_keys:
            if u == v: continue
            edgel.append( p.left_grph.has_edge(u, v) )
            edger.append( p.rrel[r_id][sl[v]] )
        
        losses.append( _evaluate(np.array(edgel), np.array(edger)) )
    
    return float(np.mean(losses))

Sol_Eval = Tuple[Solution, float]

# impl this with multi-processing ?
def _enumerate_evaluate(p: MatchingProblem) -> Iterator[Sol_Eval]:
    n = p.size
    for perm in permutations(range(n), n):
        sl = Solution({p.left_keys[i]: perm[i] for i in perm})
        yield sl, evaluate(p, sl)

def enum_with_threshold(p: MatchingProblem, threshold: float) -> Iterator[Sol_Eval]:
    for i in _enumerate_evaluate(p):
        if i[1] <= threshold: yield i

def enum_2bestk(p: MatchingProblem, k: int) -> List[Sol_Eval]:
    return sorted(_enumerate_evaluate(p), key = lambda x: x[1])[:k]

# /also use SA to solve this ?