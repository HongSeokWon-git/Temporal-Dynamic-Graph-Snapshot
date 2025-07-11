from typing import Dict, List
import random


def _variance(values: List[float]) -> float:
    mean = sum(values) / len(values)
    return sum((x - mean) ** 2 for x in values) / len(values)


def _correlation(xs: List[float], ys: List[float]) -> float:
    mean_x = sum(xs) / len(xs)
    mean_y = sum(ys) / len(ys)
    num = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    den_x = sum((x - mean_x) ** 2 for x in xs) ** 0.5
    den_y = sum((y - mean_y) ** 2 for y in ys) ** 0.5
    if den_x == 0 or den_y == 0:
        return 0.0
    return num / (den_x * den_y)


def select_trigger_features(df: Dict[str, List[float]], label_col: str = 'label', ranking_method: str = 'variance', top_k: int = 3) -> List[str]:
    """Select top_k feature keys from a dictionary-based table.

    Parameters
    ----------
    df : Dict[str, List[float]]
        Mapping of column names to values.
    label_col : str
        Name of the label column.
    ranking_method : str
        Method used for ranking features. Supported methods are ``variance``,
        ``correlation`` and ``random``.
    top_k : int
        Number of features to return.
    """
    features = {k: v for k, v in df.items() if k != label_col}

    if ranking_method == 'variance':
        scores = {k: _variance(v) for k, v in features.items()}
    elif ranking_method == 'correlation':
        labels = df[label_col]
        scores = {k: abs(_correlation(v, labels)) for k, v in features.items()}
    elif ranking_method == 'random':
        scores = {k: random.random() for k in features}
    else:
        raise ValueError(f'Unknown ranking method: {ranking_method}')

    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [k for k, _ in ranked[:top_k]]
