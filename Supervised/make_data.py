from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from pandas import DataFrame
from sklearn import datasets


__all__ = ['regression_data', 'classification_data']

def regression_data():
    housing = fetch_california_housing()
    X = housing.data
    y = housing.target
    data = DataFrame(X)
    scaler = StandardScaler()
    X = scaler.fit_transform(data)
    return train_test_split(X, y, test_size= 0.2, random_state= 123)

def classification_data():
    X, y = datasets.make_classification(n_samples=1000, n_features=10, n_classes=2, random_state=123)
    return train_test_split(X, y, test_size=0.2, random_state=123)


