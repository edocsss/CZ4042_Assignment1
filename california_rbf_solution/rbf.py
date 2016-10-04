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


def calculate_piecewise_fi(x, centroid):
    return calculate_euclidean_distance(x, centroid)


def calculate_cubic_fi(x, centroid):
    return math.pow(calculate_euclidean_distance(x, centroid), 3)


def calculate_thin_plate_fi(x, centroid):
    dist = calculate_euclidean_distance(x, centroid)
    try:
        return math.pow(dist, 2) * math.log(dist, 2)
    except:
        return 0.0


def calculate_gaussian_fi(x, centroid, covariance):
    covariance_inverse = np.linalg.inv(covariance)
    sub_T = np.subtract(x, centroid)
    sub = sub_T.reshape(-1, 1)

    cov_sub = np.dot(covariance_inverse, sub)
    subT_cov_sub = np.dot(sub_T, cov_sub)

    power = (-0.5) * subT_cov_sub
    return math.pow(math.e, power)


def calculate_covariance(X, miu, x_clusters):
    result_list = []
    for i in range(len(miu)):
        sigma = []
        centroid = miu[i]
        x_for_this_centroid = [X[index] for index, x in enumerate(x_clusters) if x == i]

        for x in x_for_this_centroid:
            sub_T = np.subtract(x, centroid)
            sub = sub_T.reshape(-1, 1)
            multiply = np.multiply(sub, sub_T)
            sigma.append(multiply)

        sigma_total = np.zeros((8, 8))
        for s in sigma:
            sigma_total = np.add(sigma_total, s)

        if np.count_nonzero(sigma_total) > 0:
            result = sigma_total / (len(x_for_this_centroid) - 1)
        else:
            result = sigma_total

        result_list.append(result)

    return result_list


def remove_singular_centroid_covariance(miu, covariances):
    new_miu = []
    new_covariances = []

    for i in range(len(miu)):
        if np.count_nonzero(covariances[i]) > 0:
            new_miu.append(miu[i])
            new_covariances.append(covariances[i])

    return new_miu, new_covariances


def train_rbf_model(X, Y, n_neuron_hidden_layer=10):
    start = time.time()
    kmeans = find_miu_using_knn(X, n_neuron_hidden_layer)
    miu = kmeans.cluster_centers_

    x_clusters = kmeans.labels_
    covariances = calculate_covariance(X, miu, x_clusters)
    # miu, covariances = remove_singular_centroid_covariance(miu, covariances)
    Z = []

    counter = 0
    for x in X:
        print(counter)
        counter += 1

        z = []
        for i in range(len(miu)):
            # fi = calculate_gaussian_fi(x, miu[i], covariances[i]) # scalar results
            # fi = calculate_piecewise_fi(x, miu[i])
            fi = calculate_cubic_fi(x, miu[i])
            # fi = calculate_thin_plate_fi(x, miu[i])
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
    return W, miu, covariances


def predict_house_price(X, weights, miu, covariances):
    predictions = []
    for x in X:
        z = [1] # bias
        for i in range(len(miu)):
            # fi = calculate_gaussian_fi(x, miu[i], covariances[i])
            # fi = calculate_piecewise_fi(x, miu[i])
            fi = calculate_cubic_fi(x, miu[i])
            # fi = calculate_thin_plate_fi(x, miu[i])
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
    X_train = data['X_train']
    Y_train = data['Y_train']
    X_model_train = data['X_train_norm']
    Y_model_train = data['Y_train_norm']
    X_model_validation = data['X_validation_norm']
    Y_model_validation = data['Y_validation_norm']

    N_NEURON_HIDDEN_LAYER = [1, 2, 3, 4, 5, 10, 15]
    print('Settings:')
    print('N_NEURON_HIDDEN_LAYER: {}'.format(N_NEURON_HIDDEN_LAYER))

    final_result = []
    for n_neuron_hidden_layer in N_NEURON_HIDDEN_LAYER:
        print()
        print()
        print('N_NEURON_HIDDEN_LAYER: {}'.format(n_neuron_hidden_layer))
        weights, miu, covariances = train_rbf_model(
            X_model_train,
            Y_model_train,
            n_neuron_hidden_layer=n_neuron_hidden_layer
        )

        validation_predictions = predict_house_price(X_model_validation, weights, miu, covariances)
        validation_mse = calculate_mse(validation_predictions, Y_model_validation)
        print('Validation MSE: {}'.format(validation_mse))

        training_predictions = predict_house_price(X_model_train, weights, miu, covariances)
        training_mse = calculate_mse(training_predictions, Y_model_train)
        print('Training MSE: {}'.format(training_mse))

        result = {
            'n_neuron_hidden_layer': n_neuron_hidden_layer,
            'validation_predictions': validation_predictions,
            'validation_mse': validation_mse,
            'training_predictions': training_predictions,
            'training_mse': training_mse,
            'weights': weights,
            'miu': miu,
            'covariances': covariances
        }

        final_result.append(result)

    f = open('results/rbf_result_thin_plate_{}.p'.format(N_NEURON_HIDDEN_LAYER),'wb')
    pickle.dump(final_result, f)
    f.close()