import random
from collections import OrderedDict

import numpy as np
import scipy.stats as st
import torch


def prepare_batch_inputs(batch, img_model_name, foundation_model, device):
    if img_model_name != "MII" or foundation_model is not None:
        raise ValueError("The public release expects precomputed 2D MII embeddings.")
    return batch["image"].to(device), batch["demo_features"].to(device)


def save_checkpoint(model, optimizer, epoch, loss, save_dir, filename):
    save_path = save_dir / filename
    torch.save(
        {
            "epoch": epoch,
            "state_dict": model.state_dict(),
            "optim": optimizer.state_dict(),
            "loss": loss,
        },
        save_path,
    )


def load_model_with_flex(
    model,
    checkpoint_path,
    checkpoint_key="state_dict",
    map_location="cpu",
    use_dataparallel=False,
    device_ids=None,
):
    checkpoint = torch.load(checkpoint_path, map_location=map_location)
    state_dict = checkpoint[checkpoint_key]
    has_module_prefix = next(iter(state_dict)).startswith("module.")

    if use_dataparallel:
        model = torch.nn.DataParallel(model, device_ids=device_ids)
        if not has_module_prefix:
            state_dict = OrderedDict((f"module.{k}", v) for k, v in state_dict.items())
    elif has_module_prefix:
        state_dict = OrderedDict((k.replace("module.", "", 1), v) for k, v in state_dict.items())

    model.load_state_dict(state_dict)
    model.to(map_location)
    model.eval()
    return model


def bootstrap_ci(cindex_func, estimates_tensor, events_tensor, times_tensor, num_bootstrap=100):
    estimates = estimates_tensor.cpu().numpy()
    events = events_tensor.cpu().numpy()
    times = times_tensor.cpu().numpy()
    n = len(estimates)
    cindex_list = []

    if n < 2:
        raise ValueError("Need at least two samples for bootstrap CI.")

    for _ in range(num_bootstrap):
        sample_size = n if n < 20 else random.randrange(20, n + 1)
        indices = np.random.choice(n, size=sample_size, replace=True)
        est_sample = torch.tensor(estimates[indices], dtype=torch.float32)
        evt_sample = torch.tensor(events[indices], dtype=torch.bool)
        time_sample = torch.tensor(times[indices], dtype=torch.float32)

        try:
            value = cindex_func(est_sample, evt_sample, time_sample)
            if not np.isnan(value):
                cindex_list.append(float(value))
        except Exception:
            continue

    if len(cindex_list) < 2:
        raise ValueError("Too few valid bootstrap samples to compute confidence interval.")

    c_mean = np.mean(cindex_list)
    c_sem = st.sem(cindex_list)
    if c_sem == 0:
        return c_mean, (c_mean, c_mean)
    return c_mean, st.t.interval(confidence=0.95, df=len(cindex_list) - 1, loc=c_mean, scale=c_sem)
