# %%
!source ~/switch-cuda.sh 12.8
# %%
# loading ECG data
import os
import sys
import numpy as np
import pandas as pd
import torch

import holoviews as hv
from scipy.io import loadmat
import hdbscan

hv.extension("bokeh")

sys.path.append("..")
sys.path.append("../..")

from tsprofiles.functions import *
from sigtools.transforms import *

from umap_torch.nonparametric_umap import (
    simplicial_set_embedding,
    compute_membership_strengths,
    add_id_diag,
)
from topological.utils.find_exemplar_ids import find_exemplar_ids

hv.opts.defaults(hv.opts.Curve(width=700, height=200))
# %%
DATASET_ROOT = "../data/mpsubseq/"
dat = loadmat(
    f"{DATASET_ROOT}/106.mat",
    squeeze_me=True,
    chars_as_strings=True,
    struct_as_record=True,
    simplify_cells=True,
)

ecg = dat["data"]
coord = dat["coord"]
coordIdx = dat["coordIdx"]
coordLab = dat["coordLab"]


# %%
def get_closest(array, values):
    # make sure array is a numpy array
    array = np.array(array)

    # get insert positions
    idxs = np.searchsorted(array, values, side="left")

    # find indexes where previous index is closer
    prev_idx_is_less = (idxs == len(array)) | (
        np.fabs(values - array[np.maximum(idxs - 1, 0)])
        < np.fabs(values - array[np.minimum(idxs, len(array) - 1)])
    )
    idxs[prev_idx_is_less] -= 1

    return array[idxs], idxs


# %%
closest_idx, closest_pos = get_closest(coordIdx, np.arange(len(ecg)))
closest_lab = coordLab[closest_pos]
is_closest_abnormal = closest_lab != 3
# %%
l_s = 178
ds_rate = 4
k = 20
skip_step = 8
bag_size = 178 // ds_rate

soft_interval_labels = (
    pd.Series(is_closest_abnormal[::-1])
    .rolling(l_s, 1, False)
    .mean()[::-1]
    .dropna()
    .values[:: (ds_rate * skip_step)]
)

trace_tensor = torch.tensor(ecg.astype(np.float32))[:, None]
trace_subseqs = extract_td_embeddings(trace_tensor, 1, l_s, ds_rate, "p_td")
trace_subseqs = znorm(trace_subseqs, -1).contiguous()
# %%
g1 = (
    hv.Curve(ecg, "index", "II")
    + hv.Spikes({"index": coordIdx, "c": coordLab}, "index", "c").opts(
        width=700, height=100, color="c", spike_length=1, cmap="bkr"
    )
    # + hv.Curve(
    #     (np.arange(len(ecg))[:: (ds_rate * skip_step)], soft_interval_labels),
    #     "index",
    #     "label",
    # ).opts(height=100)
).cols(1)
g1
# %%
D, I = mpdist_exclusion_knn_search(
    trace_subseqs, k, bag_size, skip_step=skip_step, quantile=0
)
# %%
# (
#     g1
#     + hv.Curve(
#         (np.arange(0, skip_step * ds_rate * len(D), skip_step * ds_rate), D[:, 0]),
#         "index",
#         "mpdist",
#     )
#     * hv.Curve(
#         (np.arange(0, skip_step * ds_rate * len(D), skip_step * ds_rate), D[:, 1]),
#         "index",
#         "mpdist",
#     )
#     # * hv.Curve((np.arange(0, ds_rate * len(D), ds_rate), D[:, 2]), "index", "mpdist")
#     # * hv.Curve((np.arange(0, ds_rate * len(D), ds_rate), D[:, 3]), "index", "mpdist")
#     # * hv.Curve((np.arange(0, ds_rate * len(D), ds_rate), D[:, 4]), "index", "mpdist")
# ).cols(1)
# %%
D
# %%
I
# %%
I_ = I.clone().long()
I_[I_ >= len(D) * skip_step] -= bag_size
I_ = I_ // skip_step
skd = find_local_density(D, k, local_connectivity=1, bandwidth=1)
G, SD = compute_membership_strengths(I_, D, skd, True, True)
G = add_id_diag(G)
G = sparse_make_symmetric(G)
# G = temporal_link(G, [-1, 1], [1 / k, 1 / k])
emb, _ = simplicial_set_embedding(
    G.cuda(),
    2,
    batch_size=512,
    initial_alpha=1,
    random_state=7777,
    min_dist=0.1,
    n_epochs=200,
    gamma=0.1,
)
# %%
clusterer = hdbscan.HDBSCAN(min_cluster_size=100, min_samples=10, prediction_data=True)
clusterer.fit(emb.cpu())
# %%
pd.Series(clusterer.labels_).value_counts()
# %%
g2 = (
    hv.Scatter(
        {
            "emb1": emb[:, 0].cpu(),
            "emb2": emb[:, 1].cpu(),
            "label": soft_interval_labels[: len(emb)],
        },
        "emb1",
        ["emb2", "label"],
    ).opts(
        width=500,
        height=500,
        color="label",
        alpha=0.5,
        cmap="bkr",
        colorbar=True,
        title="UMAP coloured by beat type labels (red = abnormal)",
        fontscale=1,
    )
    + hv.Scatter(
        {
            "emb1": emb[:, 0].cpu(),
            "emb2": emb[:, 1].cpu(),
            "cluster": clusterer.labels_,
        },
        "emb1",
        ["emb2", "cluster"],
    ).opts(
        width=500,
        height=500,
        color="cluster",
        alpha=0.5,
        cmap="Category20",
        colorbar=True,
        title="UMAP coloured by clusters",
        fontscale=1,
    )
).cols(2)
g2
# %%
hv.save(g2, "plots/mdl_scatter.png")
# %%
g3 = hv.Spikes(
    {
        "index": np.arange(len(ecg))[:: (ds_rate * skip_step)][: len(clusterer.labels_)]
        + l_s // 2,
        "cluster": clusterer.labels_,
    },
    "index",
    "cluster",
).opts(width=700, height=100, spike_length=1, color="cluster", cmap="Category20")
(g1 + g3).cols(1)
# %%
subseq_dat_ex = loadmat(
    f"{DATASET_ROOT}/heartbeat2_c10_b8.mat",
    squeeze_me=True,
    chars_as_strings=True,
    struct_as_record=True,
    simplify_cells=True,
)
is_not_N = coordLab != 3
cutoff = np.where(np.diff(subseq_dat_ex["idxBitsize"]) > 0)[0][0]
subseq_idx = subseq_dat_ex["idxList"][:cutoff]
g4 = (
    hv.Curve(ecg, "x", "value", label="ECG Waveform").opts(
        width=1400, height=300, fontscale=2
    )
    + hv.Spikes({"x": coordIdx, "c": coordLab}, "x", "c").opts(
        width=1400,
        height=100,
        color="c",
        fontscale=2,
        yaxis="bare",
        xaxis="bare",
        cmap="bkr",
        spike_length=1,
        title="V (Premature Ventricular Contractions) vs Normal",
    )
    + hv.Spikes(subseq_idx).opts(
        width=1400,
        height=100,
        title="Subsequences Selected by MDL",
        fontscale=2,
        yaxis="bare",
        xaxis="bare",
    )
    + hv.Spikes(
        {
            "x": np.arange(len(ecg))[:: (ds_rate * skip_step)][
                : len(clusterer.labels_)
            ][clusterer.labels_ != -1]
            + l_s // 2,
            "cluster": clusterer.labels_[clusterer.labels_ != -1],
        },
        "x",
        "cluster",
    ).opts(
        width=1400,
        height=100,
        spike_length=1,
        color="cluster",
        cmap="Category20",
        title="SPIDEC Cluster Labels",
        fontscale=2,
        yaxis="bare",
        xaxis="bare",
    )
).cols(1)
g4
# %%
hv.save(g4, "plots/mdl_coverage.png")
# %%
g5 = g4.select(x=(519000, 525000))
g5
# %%
hv.save(g5, "plots/mdl_zoom.png")
# %%
exemplar_ids = find_exemplar_ids(clusterer)

for i, ex in enumerate(exemplar_ids):
    ge = hv.Layout(
        [
            hv.Curve(
                ecg[
                    (c - skip_step)
                    * ds_rate
                    * skip_step : (c + skip_step)
                    * ds_rate
                    * skip_step
                ],
                label=f"cluster = {i}",
            ).opts(width=200, height=200, alpha=0.8, xaxis="bare", yaxis="bare")
            * hv.VSpan(l_s, 2 * l_s).opts(color="green", alpha=0.3)
            for c in np.random.choice(ex, 3, replace=False)
        ]
    ).cols(3)
    hv.output(ge)
    hv.save(ge, f"plots/exemplars/ex_{i}.png")

# %%
clusterer.condensed_tree_.plot(select_clusters=True)
# %%
