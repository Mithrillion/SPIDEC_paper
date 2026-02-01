import numpy as np
import pandas as pd
from scipy.interpolate import splrep, splev

try:
    import signatory

    HAS_SIGNATORY = True
except ImportError:
    HAS_SIGNATORY = False
    signatory = None

import torch
import warnings

# import torch.nn.functional as F
import math
from sklearn.decomposition import PCA

# import torchcde

try:
    import cuml
    import cupy as cp

    NO_CUML = False
except ImportError:
    NO_CUML = True
    warnings.warn("cuml import failed, falling back to sklearn!")


def to_td(X, tau=1, d=2, pad=None, time_last=True):
    # last dimension assumed to be time
    if not time_last:
        # then it must be (batch, time, channel)
        X = X.transpose(-2, -1)
    if d < 2:
        raise ValueError("TD dimension must be at least 2!")
    if pad == "last":
        X_padded = torch.cat(
            [X, torch.repeat_interleave(X[..., [-1]], tau * d, -1)], dim=-1
        )
    elif pad == "first":
        X_padded = torch.cat(
            [torch.repeat_interleave(X[..., [0]], tau * d, -1), X], dim=-1
        )
    else:
        X_padded = X
    return torch.stack(
        [
            X_padded[
                ...,
                i * tau : -(d - i - 1) * tau if d - i - 1 != 0 else X_padded.shape[-1],
            ]
            for i in range(d)
        ],
        dim=-1,
    ).float()


# TODO: undo td?


def time_aug(X, t_range=(0, 1)):
    # assume (batch, time, channel)
    t_len = X.shape[1]
    time_axis = torch.linspace(*t_range, t_len)
    time_component = torch.repeat_interleave(time_axis[None, :, None], len(X), dim=0)
    return torch.cat([time_component, X], dim=-1)


# TODO: support n-D data augmentation
# TODO: use scipy spl only
# def to_diff_aug(X, alpha=1.0):
#     X_tensor = torch.tensor(X[..., None]).float()
#     coeffs = torchcde.natural_cubic_coeffs(X_tensor)
#     spl = torchcde.CubicSpline(coeffs)
#     diff = spl.derivative(torch.arange(X.shape[-1]))
#     diff = (diff - diff.mean()) / diff.std()
#     return torch.cat([X_tensor, alpha * diff], dim=2)


def znorm(x, dim=-1):
    if torch.is_tensor(x):
        return (x - x.mean(dim, keepdim=True)) / (1e-8 + x.std(dim, keepdim=True))
    else:
        return (x - x.mean(dim, keepdims=True)) / (1e-8 + x.std(dim, keepdims=True))


def unorm(x, dim=-1, eps=1e-8):
    return x / (eps + torch.norm(x, dim=dim, keepdim=True))


# def envelope(x, window):
#     x_df = pd.DataFrame(x)
#     ue = x_df.rolling(window, 1, True, axis=1).max().values
#     le = x_df.rolling(window, 1, True, axis=1).min().values
#     return np.stack([ue, le], axis=-1)


def smoothing(x, window):
    x_df = pd.DataFrame(x)
    return x_df.rolling(window, 1, True, axis=1).mean().values


def to_td_pca(
    X,
    dims=2,
    tau=1,
    d=10,
    pretrained_pca=None,
    pad=None,
    backend="sklearn",
    random_state=None,
    inc_pca_batch_size=None,
):
    # (batch, time, channel) -> (batch, channel, time, delays)
    # -> (batch, time, channel * delays)
    if d == 1:
        X_td = X
    else:
        X_td = to_td(X, tau, d, pad, time_last=False).transpose(1, 2).flatten(2, 3)
    batch_dims = X_td.shape[:-1]
    if dims is not None and dims != 0:
        if pretrained_pca:
            pca = pretrained_pca
            if backend in ["cuml", "cuml_inc"] and not NO_CUML:
                X_in = X_td.reshape(-1, X_td.shape[-1])
                X_in = cp.asarray(X_in)
                Z = pca.transform(X_in)
            elif backend == "sklearn":
                Z = pca.transform(X_td.reshape(-1, X_td.shape[-1]))
            else:
                raise ValueError("Invalid backend option!")
        else:
            if backend == "cuml" and not NO_CUML:
                pca = cuml.PCA(
                    n_components=dims,
                    output_type="numpy",
                    random_state=random_state,
                )
                X_in = X_td.reshape(-1, X_td.shape[-1])
                X_in = cp.asarray(X_in)
                Z = pca.fit_transform(X_in)
            elif backend == "cuml_inc" and not NO_CUML:
                pca = cuml.IncrementalPCA(
                    n_components=dims,
                    output_type="numpy",
                    batch_size=inc_pca_batch_size,
                )
                X_in = X_td.reshape(-1, X_td.shape[-1])
                X_in = cp.asarray(X_in)
                Z = pca.fit_transform(X_in)
            elif backend == "sklearn":
                pca = PCA(n_components=dims, random_state=random_state)
                Z = pca.fit_transform(X_td.reshape(-1, X_td.shape[-1]))
            else:
                raise ValueError("Invalid backend option!")
        return torch.tensor(Z).float().view(*batch_dims, -1), pca
    else:
        # dims is None or 0 -> do not perform pca
        return X_td, None


def to_steps(X, bins, cp=None):
    cuts = np.linspace(0, 1, bins)
    if cp is None:
        cut_points = np.quantile(X.reshape(-1), cuts, interpolation="linear")
        # cut_points = np.linspace(0, 1, bins)
    else:
        cut_points = cp
    steps = np.digitize(X, cut_points)
    ext_cut_points = np.concatenate([cut_points[[0]], cut_points, cut_points[[-1]]])
    return ext_cut_points[steps], cut_points


def to_leadlag(X):
    if not torch.is_tensor(X):
        X = torch.tensor(X).float()
    # [batch, time, (channel)]
    X_repeat = X.repeat_interleave(2, dim=1)

    # Split out lead and lag
    lead = X_repeat[:, 1:, :]
    lag = X_repeat[:, :-1, :]

    # Combine
    X_leadlag = torch.cat((lead, lag), 2)

    return X_leadlag


def to_nd_leadlag(X):
    if not torch.is_tensor(X):
        X = torch.tensor(X).float()
    # [batch, time, channel]
    n_chs = X.shape[-1]
    X_repeat = X.repeat_interleave(n_chs * 2, dim=1)
    streams = []
    for i in range(2):
        for j in range(n_chs):
            streams += [
                X_repeat[
                    :, 2 * n_chs - 2 * i - j - 1 : (X_repeat.shape[1] - 2 * i - j), j
                ]
            ]
    X_leadlag = torch.stack(streams, 2)
    return X_leadlag


def resample(X, resample, smooth=0, k=3):
    # TODO: batch optimisation?
    spls = [
        splrep(np.linspace(0, 1, X.shape[-1]), X[i, :], k=k, s=smooth)
        for i in range(X.shape[0])
    ]
    X_rs = [splev(np.linspace(0, 1, resample), spl) for spl in spls]
    return np.stack(X_rs, axis=0)


# NOTE: borrowed
def rescale_path(path, depth):
    coeff = math.factorial(depth) ** (1 / depth)
    return coeff * path


def estimate_depth_limit(dims, data_size, strict=True, sig_mode="logsig"):
    if not HAS_SIGNATORY:
        raise ImportError(
            "signatory is required for estimate_depth_limit. "
            "Install it with: uv pip install signatory"
        )
    sigch = (
        signatory.logsignature_channels
        if sig_mode == "logsig"
        else signatory.signature_channels
    )
    for i in range(2, 10):
        if sigch(dims, i) > data_size:
            return i - 1 if strict and i > 2 else i
    return 9 if strict else 10


# def rescale_signature(signature, channels, depth):
#     sigtensor_channels = signature.size(-1)
#     if signatory.signature_channels(channels, depth) != sigtensor_channels:
#         raise ValueError(
#             "Given a sigtensor with {} channels, a path with {} channels and a depth of {}, which are "
#             "not consistent.".format(sigtensor_channels, channels, depth)
#         )

#     end = 0
#     term_length = 1
#     val = 1
#     terms = []
#     for d in range(1, depth + 1):
#         start = end
#         term_length *= channels
#         end = start + term_length

#         val *= d

#         terms.append(signature[..., start:end] * val)

#     return torch.cat(terms, dim=-1)


# def rectify_signature(signature, channels, depth):
#     sigtensor_channels = signature.size(-1)
#     if signatory.signature_channels(channels, depth) != sigtensor_channels:
#         raise ValueError(
#             "Given a sigtensor with {} channels, a path with {} channels and a depth of {}, which are "
#             "not consistent.".format(sigtensor_channels, channels, depth)
#         )

#     end = 0
#     term_length = 1
#     terms = []
#     for d in range(1, depth + 1):
#         start = end
#         term_length *= channels
#         end = start + term_length
#         this = signature[..., start:end]
#         terms.append(torch.sign(this) * torch.abs(this) ** (1 / d))

#     return torch.cat(terms, dim=-1)
