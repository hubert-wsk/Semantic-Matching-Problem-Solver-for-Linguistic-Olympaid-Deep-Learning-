import torch, torch.nn as nn
from typing import Tuple, Sequence


# (... use nested-tensor ?)
TokensT = Tuple[torch.Tensor, ...]
class Composer(nn.Module):
    def __init__(self, dim, rk):
        super().__init__()
        self.featw = nn.Sequential(
            nn.Linear(4, rk, bias = False),
            nn.Linear(rk, 1, bias = False),
        )
        self.conrank = nn.Sequential(
            nn.Linear(dim, rk, bias = False),
            #nn.Tanh(),
            nn.Linear(rk, dim, bias = False),
            nn.Tanh(),
        )
    
    def _forward1(self, vecs: TokensT) -> torch.Tensor:
        X = torch.stack(vecs)
        X_avg, X_max, X_min = X.mean(0), X.max(0)[0], X.min(0)[0]
        X_absmax = X[ X.abs().argmax(dim = 0) , torch.arange(X.size(1)) ]   # ...
        F = torch.stack( [X_avg, X_max, X_min, X_absmax] , dim = 1)
        alpha = torch.squeeze( self.featw(F) )
        fused = alpha * X_avg + (1 - alpha) * X_absmax
        return fused + self.conrank(fused)

    def forward_(self, *vecs: torch.Tensor): return self._forward1(vecs)

    def forward(self, vecs: Sequence[TokensT]) -> torch.Tensor:
        return torch.stack( [self._forward1(i) for i in vecs] )

class Similarity_:
    ...
    # maybe use CosineEmbeddingLoss to train this one ? (...) ( -> no no no!)