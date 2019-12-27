import numpy as np
import torch


def greedy_search(raw: torch.Tensor, blank=0):
    _, max_id = raw.max(2)
    max_id = max_id.cpu().numpy().astype(int)

    mask = np.diff(max_id, axis=0)
    mask = np.concatenate((np.ones((1, mask.shape[1])), mask), axis=0)
    mask = mask.astype(bool) & (max_id != blank)

    texts = []
    for i in range(max_id.shape[1]):
        text = max_id[mask[:, i], i]
        texts.append(text)

    return texts
