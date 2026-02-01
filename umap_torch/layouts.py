from typing import Type

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data
from torch import Tensor

import numpy as np

from tqdm.auto import tqdm

from umap.umap_ import find_ab_params
from .utils import make_batches, KNNGraphSampleDataset


class EuclideanLayout(nn.Module):
    def __init__(
        self,
        init_embedding: Tensor,
        secondary_embedding: Tensor = None,
        spread: float = 1.0,
        min_dist: float = 0.1,
        gamma: float = 1.0,
        a: float = None,
        b: float = None,
        move_other: bool = False,
        push_tail: bool = True,
        eps: float = 1e-4,
    ) -> None:
        # TODO: add densmap stuff
        super().__init__()
        self.embedding = nn.Parameter(init_embedding)
        # self.embedding = nn.Embedding.from_pretrained(init_embedding, freeze=False)
        if secondary_embedding is not None:
            self.secondary_embedding = nn.Parameter(secondary_embedding)
            # self.secondary_embedding = nn.Embedding.from_pretrained(secondary_embedding, freeze=not move_other)
        self.min_dist = min_dist
        self.gamma = gamma
        self.eps = eps
        self.push_tail = push_tail

        if a is None or b is None:
            self._a, self._b = find_ab_params(spread, min_dist)
        else:
            self._a = self.a
            self._b = self.b
        self._a, self._b = (
            torch.tensor(self._a, device=init_embedding.device),
            torch.tensor(self._b, device=init_embedding.device),
        )

    def _approximate_membership(self, x, y):
        return 1.0 / (1 + self._a * (torch.norm(x - y, dim=-1) ** (2 * self._b)))

    def _exact_membership(self, x, y):
        d = torch.norm(x - y) - self.min_dist
        return torch.where(d > 0, torch.exp(-d), torch.tensor([1.0]))

    @staticmethod
    def _dot_membership(x, y):
        return torch.sigmoid(torch.sum(x * y, dim=-1))

    def forward(self, s, e, J, V):
        # non-contiguous is bad! help needed to speed this up
        emb_from = self.embedding[s:e, :]
        if not self.push_tail:
            with torch.no_grad():
                emb_to = self.embedding[J.T, :]
        else:
            emb_to = self.embedding[J.T, :]
        membership_score = self._approximate_membership(emb_from, emb_to)
        attractions = torch.log(torch.clip(membership_score, self.eps, 1.0)).T
        repulsions = torch.log(torch.clip(1 - membership_score, self.eps, 1.0)).T
        loss = -V * attractions - self.gamma * (1 - V) * repulsions
        # loss = - V * attractions - torch.where(N, self.gamma, 0.) * repulsions
        return torch.mean(loss, dim=-1)


def optimize_layout_euclidean(
    embedding: Tensor,
    graph: torch.sparse.Tensor,
    n_epochs: int,
    batch_size: int = 128,
    secondary_embedding: Tensor = None,
    initial_alpha: float = 1e-1,
    negative_sample_rate: int = 5,
    tqdm_kwds: dict = {},
    move_other: bool = False,
    push_tail: bool = True,
    layout_model_kwds: dict = {},
    random_state: int = 1111,
    optim_kwds: dict = {},
    scheduler_kwds: dict = {},
) -> Tensor:
    """
    Improve an embedding using stochastic gradient descent to minimize the
    fuzzy set cross entropy between the 1-skeletons of the high dimensional
    and low dimensional fuzzy simplicial sets. In practice this is done by
    sampling edges based on their membership strength (with the (1-p) terms
    coming from negative sampling similar to word2vec).
    :param embedding: The initial embedding to be improved by SGD.
    :param graph: Sparse matrix of the simplicial set
    :param n_epochs: The number of training epochs to use in optimization.
    :param batch_size: batch size used to draw positive samples (graph edges)
    :param secondary_embedding: The reference embedding of embedded points. If not embedding new
        previously unseen points with respect to an existing embedding this
        is simply the first embedding (again); otherwise it provides the
        existing embedding to embed with respect to.
    :param initial_alpha: Initial learning rate for the SGD.
    :param negative_sample_rate: Number of negative samples to use per positive sample.
    :param tqdm_kwds: Keyword arguments for tqdm progress bar.
    :param move_other: Whether to adjust secondary_embedding alongside embedding
    :param layout_model_kwds: keywords passed into Layout Model
    :return:
    """
    torch.manual_seed(random_state)
    np.random.seed(random_state)
    knn_dataset = KNNGraphSampleDataset(
        graph, negative_sample_rate=negative_sample_rate
    )
    # a batch will contain head indices, tail (knn and negative placeholder=-1) indices and similarities
    layout_model = EuclideanLayout(
        embedding,
        secondary_embedding,
        move_other=move_other,
        push_tail=push_tail,
        **layout_model_kwds,
    ).to(embedding.device)
    opt = optim.Adam(layout_model.parameters(), lr=initial_alpha, **optim_kwds)
    # opt = optim.SGD(layout_model.parameters(), initial_alpha, **optim_kwds)
    if scheduler_kwds == {}:
        scheduler_kwds = dict(mode="min", threshold=1e-2, patience=10, factor=0.5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(opt, **scheduler_kwds)
    # scheduler = optim.lr_scheduler.LambdaLR(opt, lr_lambda=lambda e: (n_epochs - e) / n_epochs)
    layout_model.train()
    batches = make_batches(len(knn_dataset), batch_size)
    pbar = tqdm(range(n_epochs), **tqdm_kwds)
    for n in pbar:
        pbar.set_description(f"training epoch {n}")
        # should pad to knn + negative sample length and replace "-1"s with random tail labels
        # the negative samples should still be drawn every epoch, but no longer every iteration
        knn_dataset.sample_negatives()
        np.random.shuffle(batches)
        loss_sum = 0.0
        for s, e in batches:
            # I = torch.arange(s, e, dtype=torch.long, device=graph.device)
            opt.zero_grad()
            loss = layout_model(
                s,
                e,
                knn_dataset.col_index_padded[s:e, ...],
                knn_dataset.col_value_padded[s:e, ...],
            )
            loss = torch.mean(loss)
            loss.backward()
            nn.utils.clip_grad_norm_(layout_model.parameters(), 10.0)
            loss_sum += loss.detach()
            opt.step()
        scheduler.step(loss_sum)
        pbar.set_postfix({"loss": loss_sum.cpu().numpy() / len(knn_dataset)})
    return layout_model.embedding.detach()
