from sklearn.cross_validation import train_test_split, KFold
import pandas as pd
import numpy as np
import pickle
import copy


def load_raw_data():
    file_path = 'california/cal_housing.data'
    df = pd.read_csv(file_path, header=None)
    df = df.iloc[np.random.permutation(len(df))]

    return df


def split_train_test_set(df):
    X_matrix = df.ix[:, 0:7].values
    Y_matrix = df.ix[:, 8].values
    return train_test_split(X_matrix, Y_matrix, train_size=0.7)


def calculate_X_train_df_mean_std(X_train):
    X_training_df = pd.DataFrame(data=X_train)
    mean = []
    std = []

    for c in X_training_df.columns:
        mean.append(X_training_df[c].mean())
        std.append(X_training_df[c].std())

    return mean, std


def calculate_Y_train_df_mean_std(Y_train):
    Y_training_df = pd.DataFrame(data=Y_train)
    mean = []
    std = []

    for c in Y_training_df.columns:
        mean.append(Y_training_df[c].mean())
        std.append(Y_training_df[c].std())

    return mean, std


def normalize_X_data(data, mean, std):
    data = copy.deepcopy(data)
    for i in range(len(data)):
        for j in range(len(data[i])):
            data[i][j] = (data[i][j] - mean[j]) / std[j]

    return data


def normalize_Y_data(data, mean, std):
    data = copy.deepcopy(data)
    for i in range(len(data)):
        data[i] = (data[i] - mean[0]) / std[0]

    return data


if __name__ == '__main__':
    df = load_raw_data()
    X_train, X_test, Y_train, Y_test = split_train_test_set(df)
    X_train_kfolds_indices = KFold(len(X_train), n_folds=3)

    X_train_mean, X_train_std = calculate_X_train_df_mean_std(X_train)
    X_train_norm = normalize_X_data(X_train, X_train_mean, X_train_std)
    X_test_norm = normalize_X_data(X_test, X_train_mean, X_train_std)

    Y_train_mean, Y_train_std = calculate_Y_train_df_mean_std(Y_train)
    Y_train_norm = normalize_Y_data(Y_train, Y_train_mean, Y_train_std)
    Y_test_norm = normalize_Y_data(Y_test, Y_train_mean, Y_train_std)

    X_validation_norm, X_test_norm, Y_validation_norm, Y_test_norm = train_test_split(
        X_test_norm,
        Y_test_norm,
        train_size=0.5
    )


    f = open('data_split.p', 'wb')
    data_split = {
        'X_train': X_train,
        'Y_train': Y_train,
        'X_train_kfolds_indices': X_train_kfolds_indices,
        'X_test': X_test,
        'Y_test': Y_test,
        'X_train_norm': X_train_norm,
        'Y_train_norm': Y_train_norm,
        'X_validation_norm': X_validation_norm,
        'Y_validation_norm': Y_validation_norm,
        'X_test_norm': X_test_norm,
        'Y_test_norm': Y_test_norm,
        'X_train_mean': X_train_mean,
        'X_train_std': X_train_std,
        'Y_train_mean': Y_train_mean,
        'Y_train_std': Y_train_std
    }
    pickle.dump(data_split, f)
    f.close()