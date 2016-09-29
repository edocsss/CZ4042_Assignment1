from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import RMSprop
from sklearn.cross_validation import train_test_split
import pandas as pd
import pickle
import copy
import time


def load_split_data():
    file_path = 'data_split_2.p'
    f = open(file_path, 'rb')
    data = pickle.load(f)
    f.close()

    return data


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


def train_model(X, Y, n_hidden_layer=1, n_neuron_hidden_layer=100, max_iter=1000, alpha=0.01):
    start = time.time()
    batch_size = 256

    print('\nDesigning NN model with n_hidden_layer = {}, n_neuron_hidden_layer = {}, max_iter = {}, alpha = {}, batch_size = {}'.format(n_hidden_layer, n_neuron_hidden_layer, max_iter, alpha, batch_size))
    model = Sequential()

    # First hidden layer and input layer
    model.add(Dense(n_neuron_hidden_layer, input_dim=len(X[0]), activation='linear', init='uniform'))

    # N - 1 hidden layer
    for n in range(1, n_hidden_layer):
        model.add(Dense(n_neuron_hidden_layer, activation='linear', init='uniform'))

    # Output layer
    model.add(Dense(1, activation='linear', init='uniform'))

    # Model compilation
    rmsprop = RMSprop(lr=alpha)
    model.compile(
        optimizer=rmsprop,
        loss='mean_squared_error',
        metrics=['accuracy'],
    )

    print('Model is ready for training!')
    history = model.fit(
        X,
        Y,
        nb_epoch=max_iter,
        verbose=0,
        batch_size=batch_size
    )

    print('Elapsed time: {}'.format(time.time() - start))
    return model, history


if __name__ == '__main__':
    data = load_split_data()
    X_train = data['X_train']
    Y_train = data['Y_train']
    X_train_kfolds_indices = data['X_train_kfolds_indices']
    X_test = data['X_test']
    Y_test = data['Y_test']
    X_train_norm = data['X_train_norm']
    X_test_norm = data['X_test_norm']
    Y_train_norm = data['Y_train_norm']
    Y_test_norm = data['Y_test_norm']

    N_NEURON_HIDDEN_LAYER = [1, 2, 3, 5, 10, 15, 20, 30]
    N_HIDDEN_LAYER = [2]
    MAX_ITER = [3000]
    ALPHA = [0.01]

    print('Settings:')
    print('N_NEURON_HIDDEN_LAYER = {}'.format(N_NEURON_HIDDEN_LAYER))
    print('N_HIDDEN_LAYER = {}'.format(N_HIDDEN_LAYER))
    print('MAX_ITER = {}'.format(MAX_ITER))
    print('ALPHA = {}'.format(ALPHA))

    final_result = []
    for n_neuron_hidden_layer in N_NEURON_HIDDEN_LAYER:
        for n_hidden_layer in N_HIDDEN_LAYER:
            for max_iter in MAX_ITER:
                for alpha in ALPHA:
                    kfold_results = []
                    for train_index, test_index in X_train_kfolds_indices:
                        model, history = train_model(
                            X_train_norm[train_index],
                            Y_train[train_index],
                            n_neuron_hidden_layer=n_neuron_hidden_layer,
                            n_hidden_layer=n_hidden_layer,
                            max_iter=max_iter,
                            alpha=alpha
                        )

                        metric = model.evaluate(X_train_norm[test_index], Y_train[test_index])
                        kfold_result = {
                            'metrics': metric,
                            'history': history,
                            'model': model
                        }

                        kfold_results.append(kfold_result)

                    result = {
                        'n_neuron_hidden_layer': n_neuron_hidden_layer,
                        'n_hidden_layer': n_hidden_layer,
                        'max_iter': max_iter,
                        'alpha': alpha,
                        'kfolds': kfold_results
                    }

                    final_result.append(result)

    summary = {
        'X_train': X_train,
        'Y_train': Y_train,
        'X_train_kfolds_indices': X_train_kfolds_indices,
        'X_test': X_test,
        'Y_test': Y_test,
        'X_train_norm': X_train_norm,
        'X_test_norm': X_test_norm,
        'Y_train_norm': Y_train_norm,
        'Y_test_norm': Y_test_norm,
        'configuration_results': final_result
    }

    f = open('non_normalized_results/california_result_{}_{}_{}_{}.p'.format(N_NEURON_HIDDEN_LAYER, N_HIDDEN_LAYER, MAX_ITER, ALPHA), 'wb')
    pickle.dump(summary, f)
    f.close()