"""
Utility functions for getting reconstructions and scores from the models.
"""


import torch
import pandas as pd
from ignite.engine import create_supervised_evaluator
from torch.utils.data import DataLoader


def get_recons(data, model, device):
    recons = model(data.to(device=device))
    if isinstance(recons, tuple):
        recons = recons[0]
    return recons.sigmoid().cpu().numpy()


def model_scores(models, data, model_names, metric, device):
    scores = []
    loader = DataLoader(data, batch_size=120, num_workers=8, pin_memory=True)
    for m in models:
        engine = create_supervised_evaluator(m, metric, device)
        metrics = engine.run(loader).metrics
        metric_score = metrics[list(metric.keys())[0]]
        scores.append(metric_score)

    return pd.Series(scores, index=model_names)


def infer(model, data):
    with torch.no_grad():
        model.eval()
        device = next(model.parameters()).device

        latents, targets = [], []
        for x, t in data:
            x = x.to(device=device).flatten(start_dim=1)
            z = model.embed(x)

            latents.append(z.cpu())
            targets.append(t)

    latents = torch.cat(latents)
    targets = torch.cat(targets)

    return latents, targets
