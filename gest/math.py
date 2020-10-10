import numpy as np


def relative_average_coordinate(heatmap, axis, weight_exponent=4):
    if isinstance(axis, tuple):
        return tuple(
            relative_average_coordinate(heatmap, axis=a, weight_exponent=weight_exponent)
            for a in axis
        )
    other = tuple(i for i in range(len(heatmap.shape)) if i != axis)
    weights = heatmap.sum(axis=other) ** weight_exponent
    value, step = np.linspace(0, 1, weights.shape[0], endpoint=False, retstep=True)
    value += step / 2
    return (value * weights).sum() / weights.sum()


def accumulate(accumulated, current, accumulated_weight=1):
    if accumulated is None:
        return current
    return (accumulated * accumulated_weight + current) / (accumulated_weight + 1)
