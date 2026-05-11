import ast

import numpy as np
import torch


def parse_features(features_data):
    """Parse tabular features stored as lists, arrays, or stringified vectors."""
    if isinstance(features_data, (list, np.ndarray)):
        return np.asarray(features_data, dtype=np.float32)

    if isinstance(features_data, str):
        value = features_data.strip()
        try:
            return np.asarray(ast.literal_eval(value), dtype=np.float32)
        except (SyntaxError, ValueError):
            value = value.strip("[] \n\t")
            return np.fromstring(value, sep=" ", dtype=np.float32)

    raise TypeError(f"Unexpected feature type: {type(features_data)}")


def replace_frac_with_uniform_noise(vec, frac, rng):
    x = np.asarray(vec, dtype=np.float32)
    out = x.copy()
    flat = out.ravel()
    if flat.size == 0 or frac <= 0:
        return out

    k = max(1, min(int(np.floor(frac * flat.size)), flat.size))
    idx = rng.choice(flat.size, size=k, replace=False)
    lo = np.nanmin(flat)
    hi = np.nanmax(flat)
    if not np.isfinite(lo) or not np.isfinite(hi) or hi == lo:
        return out

    flat[idx] = rng.uniform(lo, hi, size=k).astype(flat.dtype, copy=False)
    return out


def replace_frac_with_uniform_noise_torch(x, frac, generator):
    out = torch.as_tensor(x, dtype=torch.float32).clone()
    flat = out.view(-1)
    if flat.numel() == 0 or frac <= 0:
        return out

    k = max(1, min(int(frac * flat.numel()), flat.numel()))
    idx = torch.randperm(flat.numel(), generator=generator, device=flat.device)[:k]
    lo = flat.min()
    hi = flat.max()
    if hi == lo:
        return out

    flat[idx] = lo + (hi - lo) * torch.rand(
        k, generator=generator, device=flat.device, dtype=flat.dtype
    )
    return out
