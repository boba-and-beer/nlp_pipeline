"""
Create split data in python.
"""
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
from sklearn.model_selection import GroupShuffleSplit
import pandas as pd


def multilabelstratsplit(
    data, group_col: pd.Series, index=0, num_of_splits=5, **kwargs
):
    """
    Multi-label stratified split.
    """
    gss = MultilabelStratifiedKFold(num_of_splits, **kwargs)
    indices = list(gss.split(X=data, y=data, groups=group_col))
    train_index, test_index = indices[index]
    return data.loc[train_index], data.loc[test_index]


def groupkfold(data: pd.DataFrame, group_col: str, index=0, num_of_splits=5, train_size=0.8, **kwargs):
    """
    GroupKFold Split of data 
    """
    # Create the relevant indexes first for 5-fold cross validation later
    gss = GroupShuffleSplit(
        num_of_splits, train_size=train_size, test_size=1 - train_size, **kwargs
    )
    indices = list(gss.split(X=data, y=data, groups=group_col))
    train_index, test_index = indices[index]
    return data.loc[train_index], data.loc[test_index]


def train_test_split(data, split_method, model_cols, model_labels, group_col, **kwargs):
    """
    Returns the train test split.
    """
    # train, valid = groupkfold(data, **kwargs)
    train, valid = split_method(data, group_col, **kwargs)
    train_x, train_y = train[model_cols], train[model_labels]
    valid_x, valid_y = valid[model_cols], valid[model_labels]
    return train_x, train_y, valid_x, valid_y
