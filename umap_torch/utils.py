import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import typing


def make_batches(data_size, batch_size, drop_last=False, stride=None):
    if stride is None:
        stride = batch_size
    s = np.arange(0, data_size - batch_size + stride, stride)
    e = s + batch_size
    if drop_last:
        s, e = s[e < data_size], e[e < data_size]
    else:
        s, e = s[s < data_size], e[s < data_size]
        e[-1] = data_size
    return list(zip(s, e))


class KNNGraphDataset(data.Dataset):
    def __init__(self, graph: torch.tensor) -> None:
        super().__init__()
        if not graph.is_coalesced():
            graph = graph.coalesce()
        self.dev = graph.device
        graph_csr = graph.to_sparse_csr()
        crow, col, values = (
            graph_csr.crow_indices(),
            graph_csr.col_indices(),
            graph_csr.values(),
        )
        indexer = torch.unbind(torch.stack([crow[:-1], crow[1:]], dim=-1))
        col_index_list = [col[s:e] for s, e in indexer]
        col_value_list = [values[s:e] for s, e in indexer]
        self.col_index_padded = nn.utils.rnn.pad_sequence(
            col_index_list, batch_first=True, padding_value=-1
        )
        self.col_value_padded = nn.utils.rnn.pad_sequence(
            col_value_list, batch_first=True
        )
        self.n_samples = graph.shape[0]
        self.masked_positions = self.col_index_padded == -1

    def __getitem__(self, index):
        return (
            index,
            self.col_index_padded[index, :],
            self.col_value_padded[index, :],
            self.masked_positions[index, :],
        )

    def __len__(self):
        return self.n_samples


class KNNGraphEdgeDataset(data.Dataset):
    def __init__(self, graph: torch.tensor) -> None:
        super().__init__()
        if not graph.is_coalesced():
            graph = graph.coalesce()
        self.indices = graph.indices()
        self.values = graph.values()

    def __getitem__(self, index):
        return self.indices[:, index], self.values[index]

    def __len__(self):
        return len(self.values)


class KNNMultiGraphEdgeDataset(data.Dataset):
    def __init__(self, graphs: typing.Sequence[torch.tensor]) -> None:
        super().__init__()
        for i, graph in enumerate(graphs):
            if not graph.is_coalesced():
                graphs[i].coalesce()
        X_instance_cuts = np.cumsum([0] + [g.shape[0] for g in graphs])
        I_list, V_list = [], []
        for graph, start in zip(graphs, X_instance_cuts[:-1]):
            I_list += [graph.indices() + start]
            V_list += [graph.values()]
        self.indices, self.values = torch.cat(I_list, dim=-1), torch.cat(V_list, dim=-1)
        self.total_size = X_instance_cuts[-1]

    def __getitem__(self, index: int) -> typing.Tuple[torch.tensor]:
        return self.indices[:, index], self.values[index]

    def __len__(self) -> int:
        return len(self.values)

    def get_combined_graph(self) -> torch.tensor:
        return torch.sparse_coo_tensor(
            self.indices, self.values, (self.total_size, self.total_size)
        ).coalesce()


class KNNGraphSampleDataset(data.Dataset):
    def __init__(self, graph: torch.tensor, negative_sample_rate: float = 5.0) -> None:
        super().__init__()
        if not graph.is_coalesced():
            graph = graph.coalesce()
        # invert sparse similarity matrix into (knn_other_indices, knn_distances) format, allowing different rows to
        # have different number of neighbours
        self.dev = graph.device
        graph_csr = graph.to_sparse_csr()
        crow, col, values = (
            graph_csr.crow_indices(),
            graph_csr.col_indices(),
            graph_csr.values(),
        )
        indexer = torch.unbind(torch.stack([crow[:-1], crow[1:]], dim=-1))
        col_index_list = [col[s:e] for s, e in indexer]
        col_value_list = [values[s:e] for s, e in indexer]
        self.col_index_padded = nn.utils.rnn.pad_sequence(
            col_index_list, batch_first=True, padding_value=-1
        )
        self.col_value_padded = nn.utils.rnn.pad_sequence(
            col_value_list, batch_first=True
        )
        self.knn_lengths = crow[1:] - crow[:-1]
        self.n_samples = graph.shape[0]
        mean_knn_length = torch.mean(self.knn_lengths.float())
        vertex_sample_length = int(
            max(
                mean_knn_length * (negative_sample_rate + 1),
                torch.max(self.knn_lengths),
            )
        )
        extra_padded_length = vertex_sample_length - self.col_index_padded.shape[-1]
        self.col_index_padded = torch.cat(
            [
                self.col_index_padded,
                -torch.ones(
                    self.n_samples,
                    extra_padded_length,
                    dtype=torch.long,
                    device=self.dev,
                ),
            ],
            dim=-1,
        )
        self.col_value_padded = torch.cat(
            [
                self.col_value_padded,
                torch.zeros(self.n_samples, extra_padded_length, device=self.dev),
            ],
            dim=-1,
        )
        self.negative_sample_positions = self.col_index_padded == -1

    def sample_negatives(self) -> None:
        negatives = torch.tensor(
            np.random.random_integers(
                0, self.n_samples - 1, self.col_index_padded.shape
            ),
            dtype=torch.long,
            device=self.dev,
        )
        self.col_index_padded[self.negative_sample_positions] = negatives[
            self.negative_sample_positions
        ]

    def __getitem__(self, index):
        return (
            index,
            self.col_index_padded[index, :],
            self.col_value_padded[index, :],
            self.negative_sample_positions[index, :],
        )

    def __len__(self):
        return self.n_samples


def euclidean_loss(ZX, ZY, V, N, gamma, membership, eps: float = 1e-4):
    membership_score = membership(ZX, ZY)
    attraction = torch.log(torch.clip(membership_score, eps, 1.0)).T
    repulsion = torch.log(torch.clip(1 - membership_score, eps, 1.0)).T
    losses = -V * attraction - torch.where(N, gamma, 0.0) * (1 - V) * repulsion
    return torch.sum(losses, dim=-1)  # sum of batch * neg terms over neg
