import pickle
import time

from keras.layers import Dense
from keras.models import Sequential
from keras.optimizers import SGD


def load_split_data():
    file_path = 'data_split.p'
    f = open(file_path, 'rb')
    data = pickle.load(f)
    f.close()

    return data


def train_model(X, Y, n_hidden_layer=1, n_neuron_hidden_layer=100, max_iter=1000, alpha=0.01):
    start = time.time()
    batch_size = 32

    print('\nDesigning NN model with n_hidden_layer = {}, n_neuron_hidden_layer = {}, max_iter = {}, alpha = {}, batch_size = {}'.format(n_hidden_layer, n_neuron_hidden_layer, max_iter, alpha, batch_size))
    model = Sequential()

    # First hidden layer and input layer
    model.add(Dense(n_neuron_hidden_layer, input_dim=len(X[0]), activation='sigmoid', init='uniform'))

    # N - 1 hidden layer
    for n in range (1, n_hidden_layer):
        model.add(Dense(n_neuron_hidden_layer, activation='sigmoid', init='uniform'))

    # Output layer
    model.add(Dense(1, activation='sigmoid', init='uniform'))

    # Model compilation
    sgd = SGD(lr=alpha)
    model.compile(
        optimizer=sgd,
        loss='mean_squared_error',
        metrics=['accuracy']
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
    # LOAD THE SAME DATA SPLIT SO EVERY INDEPENDENT EXPERIMENT IS USING THE SAME TRAINING AND TESTING DATA
    data = load_split_data()
    X_train = data['X_train']
    Y_train = data['Y_train']
    X_train_kfolds_indices = data['X_train_kfolds_indices']
    X_test = data['X_test']
    Y_test = data['Y_test']
    X_train_norm = data['X_train_norm']
    X_test_norm = data['X_test_norm']


    N_NEURON_HIDDEN_LAYER = [50]
    N_HIDDEN_LAYER = [1, 2, 3, 4]
    MAX_ITER = [3000]
    ALPHA = [0.1]

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
        'configuration_results': final_result
    }

    f = open('main_results/spambase_result_same_split_{}_{}_{}_{}.p'.format(N_NEURON_HIDDEN_LAYER, N_HIDDEN_LAYER, MAX_ITER, ALPHA), 'wb')
    pickle.dump(summary, f)
    f.close()