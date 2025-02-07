import random

import pandas as pd
import torch
import yfinance as yf
from sklearn.preprocessing import LabelEncoder, StandardScaler

scaler = StandardScaler()


def split_train_test(x, y, test_ratio=0.3):
    idxs = [i for i in range(len(x))]
    random.shuffle(idxs)
    # delimiter between test and train data
    delim = int(len(x) * test_ratio)
    test_idxs, train_idxs = idxs[:delim], idxs[delim:]
    return x[train_idxs], y[train_idxs], x[test_idxs], y[test_idxs]


def get_hospital_data(standatrization=None):
    # this data comes from: https://www.kaggle.com/datasets/saurabhshahane/patient-treatment-classification/data#
    data = pd.read_csv("data/data-ori.csv")

    data = data.dropna()
    label_encoders = {}
    for column in data.columns:
        if column in ["SEX", "SOURCE"]:
            le = LabelEncoder()
            data[column] = le.fit_transform(data[column])
            label_encoders[column] = le

    # extract labels
    y = torch.tensor(data["SOURCE"].values).float().unsqueeze(1)

    # remove label column before normalization
    data = data.drop("SOURCE", axis=1)

    if standatrization == "z-score":
        data = (data - data.mean()) / data.std()
    elif standatrization == "min-max":
        data = (data - data.min()) / (data.max() - data.min())
    elif standatrization is None:
        pass
    else:
        raise ValueError(f"Unknown standardization method: {standatrization}")

    x = torch.tensor(data.values).float()
    return split_train_test(x, y)


def get_financial_data(lookback_years=1, standardization=None):
    tech_list = ["AAPL", "GOOG", "MSFT", "AMZN"]
    all_data = []
    for ticker in tech_list:
        stock = yf.Ticker(ticker)
        stock_info = stock.history(period="1y")  # Fetching 1 year of historical data
        all_data.append(stock_info)

    # Concatenating all DataFrames
    full_df = pd.concat(all_data, axis=0)

    full_df = full_df.dropna()
    full_df = full_df.reset_index()

    X = full_df.drop(columns=["Date", "Volume", "Stock Splits"], axis=1)

    print(X.head())

    # Apply standardization if specified
    if standardization == "z-score":
        X = (X - X.mean()) / X.std()
    elif standardization == "min-max":
        X = (X - X.min()) / (X.max() - X.min())
    elif standardization is not None:
        raise ValueError(f"Unknown standardization method: {standardization}")

    y = pd.DataFrame(X["Close"])
    X = X.drop(columns=["Close"])

    X = torch.tensor(X.values).float()

    y = torch.tensor(y.values).float().view(-1, 1)
    # print(y)
    # print(X)
    # Split into train and test sets

    return split_train_test(X, y)
