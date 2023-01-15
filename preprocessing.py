# Initialization
import numpy as np
import torch
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from itertools import chain
from datetime import datetime as dt
import random

pd.set_option('display.max_columns', None)


def split_temporal(df, time, train_split=0.8, val_split=None):
    df[time] = pd.to_datetime(df[time])
    df = df.sort_values(time)

    train_ind = int(np.round(len(df)*train_split))
    train_df = df.iloc[:train_ind]

    if val_split == None:
        test_df = df.iloc[train_ind:]
        return train_df, test_df
    else:
        val_ind = int(np.round(len(df)*(val_split + train_split)))
        val_df = df.iloc[train_ind:val_ind]
        test_df = df.iloc[val_ind:]
        
    return train_df, val_df, test_df


def preprocessing(df, column):
    print(df[column].describe())
    remap = {df[column].unique()[i]: i for i in range(len(df[column].unique()))}
    df[f"{column}"] = df[column].replace(remap)
    print(df[f"{column}"].describe())
    print(df.head())
    return df

if __name__ == "__main__":
    df = pd.read_csv("online_retail_processed.csv")
    df = preprocessing(df, "CustomerID")
    df.to_csv("online_retail_processed.csv", index=False)
    