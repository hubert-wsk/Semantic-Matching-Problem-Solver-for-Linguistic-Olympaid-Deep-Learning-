from copy import deepcopy
from abc import ABC, abstractmethod
from dataclasses import dataclass
#from numbers import Real
from typing import Callable, Union, Any
from random import random
import numpy as np


_real = Union[int, float, np.floating, np.int_, Any]

def _P(p: float): return random() < p

class SA_Problem(ABC):
    @abstractmethod
    def move(self) -> None: ...

    @abstractmethod
    def eval(self) -> _real: ...

def default_decay(k: float, iter_num: int):
    _t = iter_num / 2
    def fn(T: float, i: int) -> float:
        return k / (1 + T / _t)
    return fn

@dataclass
class SA_Config:
    iter_num: int
    T_update: Callable[[float, int], float]
    T_init: float
    verbose: ...


def SA_solve(p: SA_Problem, cfg: SA_Config):
    T = cfg.T_init
    best_ls = now_ls = p.eval(); best_mdl = now_mdl = p
    ls_ = [now_ls]

    for i in range(cfg.iter_num):
        new_mdl = deepcopy(now_mdl); new_mdl.move()
        new_ls = new_mdl.eval()

        if new_ls >= now_ls:
            now_ls, now_mdl = new_ls, new_mdl
            if new_ls >= best_ls:
                best_ls, best_mdl = new_ls, new_mdl
        
        elif _P( np.exp( (new_ls - now_ls)/ T) ):
            now_ls, now_mdl = new_ls, new_mdl
        
        T = cfg.T_update(T, i)
        # monitoring ?
    
    return best_mdl, ls_