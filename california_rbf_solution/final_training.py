import time
import numpy as np
import pickle
import math
from sklearn.cluster import KMeans


def load_split_data():
    file_path = 'data_split.p'
    f = open(file_path, 'rb')
    data = pickle.load(f)
    f.close()

    return data


def find_miu_using_knn(X, n):
    kmeans = KMeans(n_clusters=n)
    kmeans.fit(X)
    return kmeans


def calculate_euclidean_distance(x, y):
    total = 0
    for i in range(len(x)):
        total += math.pow(x[i] - y[i], 2)

    return math.sqrt(total)


def calculate_cubic_fi(x, centroid):
    return math.pow(calculate_euclidean_distance(x, centroid), 3)


def train_rbf_model(X, Y, n_neuron_hidden_layer=10):
    start = time.time()
    kmeans = find_miu_using_knn(X, n_neuron_hidden_layer)
    miu = kmeans.cluster_centers_

    x_clusters = kmeans.labels_
    Z = []

    counter = 0
    for x in X:
        print(counter)
        counter += 1

        z = []
        for i in range(len(miu)):
            fi = calculate_cubic_fi(x, miu[i])
            z.append(fi)

        Z.append([1] + z) # the first one is the bias

    print(time.time() - start)
    Z = np.matrix(Z)
    ZT = Z.transpose()

    ZT_Z = np.dot(ZT, Z)
    ZT_Z_inverse = np.linalg.inv(ZT_Z)

    Y = np.array(Y).reshape(-1, 1)
    ZT_Y = np.dot(ZT, Y)
    W = np.dot(ZT_Z_inverse, ZT_Y)
    return W, miu


def predict_house_price(X, weights, miu):
    predictions = []
    for x in X:
        z = [1] # bias
        for i in range(len(miu)):
            fi = calculate_cubic_fi(x, miu[i])
            z.append(fi)

        prediction = sum([z[i] * weights[i] for i in range(len(z))])
        predictions.append(prediction[0, 0])

    return predictions


def calculate_mse(predictions, Y):
    squared_error = [math.pow(predictions[i] - Y[i], 2) for i in range(len(predictions))]
    total_squared_error = sum(squared_error)
    return total_squared_error / len(predictions)


if __name__:
    data = load_split_data()
    X_model_train = data['X_train_norm']
    Y_model_train = data['Y_train_norm']
    X_model_validation = data['X_validation_norm']
    Y_model_validation = data['Y_validation_norm']

    X_train_full = np.append(X_model_train, X_model_validation, axis=0)
    Y_train_full = np.append(Y_model_train, Y_model_validation, axis=0)

    X_test = data['X_test_norm']
    Y_test = data['Y_test_norm']

    N_NEURON_HIDDEN_LAYER = [300]
    for n_neuron_hidden_layer in N_NEURON_HIDDEN_LAYER:
        weights, miu = train_rbf_model(
            X_train_full,
            Y_train_full,
            n_neuron_hidden_layer=n_neuron_hidden_layer
        )

        test_predictions = predict_house_price(X_test, weights, miu)
        test_mse = calculate_mse(test_predictions, Y_test)
        print('Test MSE: {}'.format(test_mse))

        training_predictions = predict_house_price(X_train_full, weights, miu)
        training_mse = calculate_mse(training_predictions, Y_train_full)
        print('Training MSE: {}'.format(training_mse))