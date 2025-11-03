from __future__ import annotations
from typing import Optional, Iterable, Set, Tuple, TypeVar, NewType
# why can't i import 'Self'T ??


class _Token:
    def __init__(self, lr: Optional[ Tuple[_Token, _Token] ], s: str):
        self.lr, self.s = lr, s
    
    @classmethod
    def merge(cls, l: _Token, r: _Token) -> _Token:
        return cls((l, r), l.s + r.s)
    
    @classmethod
    def create_new(cls, s) -> _Token: return cls(None, s)

    def __hash__(self): return id(self)

    def __eq__(self, value: _Token): return hash(self) == hash(value)

TokWord = TypeVar("TokWord", Tuple[_Token, ...])

class Splitting:
    def __init__(self, ):
        self.problem_exist = False
    
    def problem_as(self, words: Iterable[str]):
        self.prim_words = list(words)
        
        