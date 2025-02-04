import pandas as pd 
import torch 
from sklearn.preprocessing import LabelEncoder
import random

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