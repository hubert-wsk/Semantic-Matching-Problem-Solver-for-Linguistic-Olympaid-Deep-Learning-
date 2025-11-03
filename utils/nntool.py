import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from typing import (
    Sequence,
    TypeVar,
    Generic,
    Tuple,
    Any,
    Union,
)
from dataclasses import dataclass, field
import numpy as np
import tqdm.auto as tqdm


T = TypeVar("T")
class UniDataset(Dataset, Generic[T]):
    def __init__(self, X: Sequence[T], y: torch.Tensor):
        self.X, self.y, self.ln = X, y, y.size(0)
    
    def __len__(self):
        return self.ln
    
    def __getitem__(self, idx) -> Tuple[T, torch.Tensor]:
        return self.X[idx], self.y[idx]
    
    def to_dataloader(self, batch_size: int, n_jobs: int = 1, _collate_fn = None):      # n_jobs: *
        return DataLoader(
            self,
            batch_size = batch_size,
            shuffle = True,
            num_workers= n_jobs,
            collate_fn = _collate_fn
        )


class SimpTrainer:
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        loss_fn,
        optimizer: optim.Optimizer,
    ):
        self.model = model
        self.train_loader = train_loader
        self.loss_fn = loss_fn
        self.opt = optimizer
        self._epoch_n = 0
    
    def train_epoch(self, verbose: Union[str, None] = None):
        self._epoch_n += 1
        self.model.train()
        los_ = []

        for Xb, yb in self.train_loader if not verbose else tqdm.tqdm(self.train_loader, desc = verbose, leave = False):
            self.opt.zero_grad()
            #loss: torch.Tensor
            ( loss := self.loss_fn(self.model(Xb), yb) ).backward()
            self.opt.step()
            los_.append( loss.item() )
        
        return np.mean(los_)
    
    @torch.no_grad()
    def eval_(self, Xtest: ..., ytest: torch.Tensor, metric: ... = None) -> Union[torch.Tensor, Any]:
        metric = metric or self.loss_fn
        self.model.eval()
        return metric( self.model(Xtest), ytest )
    
