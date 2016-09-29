from sklearn.cross_validation import train_test_split, KFold
import pandas as pd
import numpy as np
import pickle
import copy


def load_raw_data():
    file_path = 'spambase/spambase.data'
    df = pd.read_csv(file_path, header=None)
    df = df.iloc[np.random.permutation(len(df))]

    return df


def split_train_test_set(df):
    X_matrix = df.ix[:, 0:56].values
    Y_matrix = df.ix[:, 57].values
    return train_test_split(X_matrix, Y_matrix, train_size=0.7)


def calculate_X_train_df_mean_std(X_train):
    X_training_df = pd.DataFrame(data=X_train)
    mean = []
    std = []

    for c in X_training_df.columns:
        mean.append(X_training_df[c].mean())
        std.append(X_training_df[c].std())

    return mean, std


def normalize_data(data, mean, std):
    data = copy.deepcopy(data)
    for i in range(len(data)):
        for j in range(len(data[i])):
            data[i][j] = (data[i][j] - mean[j]) / std[j]

    return data


if __name__ == '__main__':
    df = load_raw_data()
    X_train, X_test, Y_train, Y_test = split_train_test_set(df)
    X_train_kfolds_indices = KFold(len(X_train), n_folds=3)

    X_train_mean, X_train_std = calculate_X_train_df_mean_std(X_train)
    X_train_norm = normalize_data(X_train, X_train_mean, X_train_std)
    X_test_norm = normalize_data(X_test, X_train_mean, X_train_std)

    data_split = {
        'X_train': X_train,
        'Y_train': Y_train,
        'X_train_kfolds_indices': X_train_kfolds_indices,
        'X_test': X_test,
        'Y_test': Y_test,
        'X_train_norm': X_train_norm,
        'X_test_norm': X_test_norm
    }

    f = open('data_split_2.p', 'wb')
    pickle.dump(data_split, f)
    f.close()