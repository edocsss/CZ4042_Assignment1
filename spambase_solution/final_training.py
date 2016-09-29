import pickle
import pprint
import time
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import SGD


def load_result_data():
    file_path = 'data_split.p'
    f = open(file_path, 'rb')
    data = pickle.load(f)
    f.close()

    return data


def train_model(X, Y, n_hidden_layer=1, n_neuron_hidden_layer=100, max_iter=1000, alpha=0.01):
    start = time.time()
    print('\nDesigning NN model with n_hidden_layer = {}, n_neuron_hidden_layer = {}, max_iter = {}, alpha = {}'.format(n_hidden_layer, n_neuron_hidden_layer, max_iter, alpha))
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
    history = model.fit(X, Y, nb_epoch=max_iter, verbose=1, batch_size=len(X))
    print('Elapsed time: {}'.format(time.time() - start))
    return model, history


if __name__ == '__main__':
    data = load_result_data()
    X_train_norm = data['X_train_norm']
    Y_train = data['Y_train']

    X_test_norm = data['X_test_norm']
    Y_test = data['Y_test']

    target_n_hidden_layer = 1
    target_n_neuron_hidden_layer = 50
    target_max_iter = 3000
    target_alpha = 0.1

    model, history = train_model(
        X_train_norm,
        Y_train,
        target_n_hidden_layer,
        target_n_neuron_hidden_layer,
        target_max_iter,
        target_alpha
    )

    final_error = model.evaluate(X_test_norm, Y_test)
    predictions = model.predict_classes(X_test_norm)

    for i in range(len(predictions)):
        print(predictions[i], Y_test[i])

    pprint.pprint(final_error)