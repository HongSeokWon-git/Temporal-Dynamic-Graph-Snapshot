# Run `pytest` to execute these tests
import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.feature_selection import select_trigger_features


def dummy_df():
    return {
        'f1': [1, 2, 3, 4],
        'f2': [4, 3, 2, 1],
        'f3': [1, 1, 1, 1],
        'label': [0, 1, 0, 1]
    }


def test_select_trigger_features_variance():
    df = dummy_df()
    result = select_trigger_features(df, ranking_method='variance', top_k=2)
    assert isinstance(result, list)


def test_select_trigger_features_correlation():
    df = dummy_df()
    result = select_trigger_features(df, ranking_method='correlation', top_k=2)
    assert isinstance(result, list)


def test_select_trigger_features_random():
    df = dummy_df()
    result = select_trigger_features(df, ranking_method='random', top_k=2)
    assert isinstance(result, list)
