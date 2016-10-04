import os
import pickle
import pprint
import matplotlib.pyplot as plt


BASIS_FUNCTION = 'gaussian'


def load_result_data():
    file_path = os.path.join('results', 'rbf_result_{}_[1, 2, 3, 4, 5, 10, 15, 20, 25, 30, 50, 75, 100, 300, 500, 1000].p'.format(BASIS_FUNCTION))
    f = open(file_path, 'rb')
    data = pickle.load(f)
    f.close()

    return data


def build_combined_error_graph(results):
    training_x = []
    training_error = []

    validation_x = []
    validation_error = []

    for result in results:
        training_x.append(result[0])
        training_error.append(result[1]['training_mse'])

        validation_x.append(result[0])
        validation_error.append(result[1]['validation_mse'])

    plt.plot(training_x, training_error, label='Training')
    plt.plot(validation_x, validation_error, label='Validation')

    plt.xlabel('n_neuron_hidden_layer')
    plt.ylabel('Error')
    plt.legend()
    plt.title('Training and Validation Error')
    plt.show()


if __name__ == '__main__':
    raw_results = load_result_data()
    d = []

    for result in raw_results:
        training_mse = result['training_mse']
        validation_mse = result['validation_mse']
        n_neuron_hidden_layer = result['n_neuron_hidden_layer']

        d.append((n_neuron_hidden_layer, {
            'training_mse': training_mse,
            'validation_mse': validation_mse
        }))

    build_combined_error_graph(d)
