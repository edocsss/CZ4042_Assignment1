import pickle
import pprint
import matplotlib.pyplot as plt
import pandas as pd


N_NEURON_HIDDEN_LAYER = [1, 2, 3, 4, 5, 10, 20, 30, 50, 75, 100]
N_HIDDEN_LAYER = [1]
MAX_ITER = [10000]
ALPHA = [0.01]

FOLDER_NAME = 'results'
GRAPH_X_AXIS = 'n_neuron_hidden_layer'


def load_result_data():
    file_path = '{}/spambase_result_{}_{}_{}_{}.p'.format(FOLDER_NAME, N_NEURON_HIDDEN_LAYER, N_HIDDEN_LAYER, MAX_ITER, ALPHA)
    f = open(file_path, 'rb')
    results = pickle.load(f)
    f.close()

    pprint.pprint(results)
    return results


def calculate_average_loss_and_accuracy_per_config(results, k=3):
    for config in results['configuration_results']:
        total_loss = 0
        total_accuracy = 0

        for kfold_result in config['kfolds']:
            total_loss += kfold_result['metrics'][0]
            total_accuracy += kfold_result['metrics'][1]

        average_loss = total_loss / len(config['kfolds'])
        average_accuracy = total_accuracy / len(config['kfolds'])

        config['average_loss'] = average_loss
        config['average_accuracy'] = average_accuracy

    return results


def find_best_loss_and_accuracy(results):
    best_accuracy_index = 0
    best_accuracy = 0
    best_loss_index = 0
    best_loss = 100

    for i, config in enumerate(results['configuration_results']):
        if config['average_loss'] < best_loss:
            best_loss = config['average_loss']
            best_loss_index = i

        if config['average_accuracy'] > best_accuracy:
            best_accuracy = config['average_accuracy']
            best_accuracy_index = i

    print('Configuration index with best accuracy: {}, accuracy: {}'.format(best_accuracy_index, best_accuracy))
    print('Configuration index with best loss: {}, loss: {}'.format(best_loss_index, best_loss))

    print('Configuration with best accuracy:')
    pprint.pprint(results['configuration_results'][best_accuracy_index])

    print()
    print()

    print('Configuration with best loss:')
    pprint.pprint(results['configuration_results'][best_loss_index])

    return results['configuration_results'][best_accuracy_index], results['configuration_results'][best_loss_index]


def draw_loss_history_graph_per_config(config_result):
    x_axis = GRAPH_X_AXIS
    if GRAPH_X_AXIS == 'max_iter':
        x_axis = 'max_epoch'

    for i in range(0, len(config_result['kfolds'])):
        loss = config_result['kfolds'][i]['history'].history['loss']
        epoch = [x for x in range(0, len(loss))]

        plt.plot(epoch, loss)

    plt.xlabel('Epoch')
    plt.ylabel('Error')
    plt.title('Error Convergence ({} = {})'.format(
        x_axis,
        config_result[GRAPH_X_AXIS]
    ))

    plt.show()


def draw_accuracy_history_graph_per_config(config_result):
    x_axis = GRAPH_X_AXIS
    if GRAPH_X_AXIS == 'max_iter':
        x_axis = 'max_epoch'

    for i in range(0, len(config_result['kfolds'])):
        accuracy = config_result['kfolds'][i]['history'].history['acc']
        epoch = [x for x in range(0, len(accuracy))]
        plt.plot(epoch, accuracy)

    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Accuracy Convergence ({} = {})'.format(
        x_axis,
        config_result[GRAPH_X_AXIS]
    ))

    plt.show()


def draw_validation_set_loss_graph_overall(config_results, x_axis):
    X = []
    loss = []

    for c in config_results:
        x = c[x_axis]
        avg_loss = c['average_loss']

        X.append(x)
        loss.append(avg_loss)

    if x_axis == 'max_iter':
        x_axis = 'max_epoch'

    plt.plot(X, loss)
    plt.xlabel(x_axis)
    plt.ylabel('Error')
    plt.title('Validation Error')
    plt.show()


def draw_validation_set_accuracy_graph_overall(config_results, x_axis):
    X = []
    accuracy = []

    for c in config_results:
        x = c[x_axis]
        avg_accuracy = c['average_accuracy']

        X.append(x)
        accuracy.append(avg_accuracy)

    if x_axis == 'max_iter':
        x_axis = 'max_epoch'

    plt.plot(X, accuracy)
    plt.xlabel(x_axis)
    plt.ylabel('Accuracy')
    plt.title('Validation Accuracy')
    plt.show()


def draw_training_set_loss_graph_overall(config_results, x_axis):
    X = []
    loss = []

    for c in config_results:
        x = c[x_axis]
        kfold_results = c['kfolds']
        training_loss = [r['history'].history['loss'][-1] for r in kfold_results]
        pprint.pprint([r['history'].history['loss'][-1] for r in kfold_results])
        average_training_loss = sum(training_loss) / len(kfold_results)

        X.append(x)
        loss.append(average_training_loss)

    if x_axis == 'max_iter':
        x_axis = 'max_epoch'

    plt.plot(X, loss)
    plt.xlabel(x_axis)
    plt.ylabel('Error')
    plt.title('Training Error')
    plt.show()


def draw_training_set_accuracy_graph_overall(config_results, x_axis):
    X = []
    accuracy = []

    for c in config_results:
        x = c[x_axis]
        kfold_results = c['kfolds']
        training_accuracy = [r['history'].history['acc'][-1] for r in kfold_results]
        average_training_loss = sum(training_accuracy) / len(kfold_results)

        X.append(x)
        accuracy.append(average_training_loss)

    if x_axis == 'max_iter':
        x_axis = 'max_epoch'

    plt.plot(X, accuracy)
    plt.xlabel(x_axis)
    plt.ylabel('Accuracy')
    plt.title('Training Accuracy')
    plt.show()


def draw_combined_loss_graph_overall(config_results, x_axis):
    X_validation = []
    loss_validation = []
    for c in config_results:
        x_validation = c[x_axis]
        avg_loss = c['average_loss']

        X_validation.append(x_validation)
        loss_validation.append(avg_loss)

    X_training = []
    loss_training = []
    for c in config_results:
        x_training = c[x_axis]
        kfold_results = c['kfolds']
        training_loss = [r['history'].history['loss'][-1] for r in kfold_results]
        pprint.pprint([r['history'].history['loss'][-1] for r in kfold_results])
        average_training_loss = sum(training_loss) / len(kfold_results)

        X_training.append(x_training)
        loss_training.append(average_training_loss)

    if x_axis == 'max_iter':
        x_axis = 'max_epoch'

    plt.plot(X_validation, loss_validation, label='Validation')
    plt.plot(X_training, loss_training, label='Training')
    plt.xlabel(x_axis)
    plt.ylabel('Error')
    plt.legend()
    plt.title('Training and Validation Error')
    plt.show()


def draw_combined_accuracy_graph_overall(config_results, x_axis):
    X_validation = []
    accuracy_validation = []
    for c in config_results:
        x_validation = c[x_axis]
        avg_accuracy = c['average_accuracy']

        X_validation.append(x_validation)
        accuracy_validation.append(avg_accuracy)

    X_training = []
    accuracy_training = []
    for c in config_results:
        x_training = c[x_axis]
        kfold_results = c['kfolds']
        training_accuracy = [r['history'].history['acc'][-1] for r in kfold_results]
        average_training_loss = sum(training_accuracy) / len(kfold_results)

        X_training.append(x_training)
        accuracy_training.append(average_training_loss)

    if x_axis == 'max_iter':
        x_axis = 'max_epoch'

    plt.plot(X_validation, accuracy_validation, label='Validation')
    plt.plot(X_training, accuracy_training, label='Training')
    plt.xlabel(x_axis)
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title('Training and Validation Accuracy')
    plt.show()


if __name__ == '__main__':
    results = load_result_data()
    results = calculate_average_loss_and_accuracy_per_config(results)
    best_accuracy_config, best_loss_config = find_best_loss_and_accuracy(results)

    draw_combined_loss_graph_overall(results['configuration_results'], GRAPH_X_AXIS)
    draw_combined_accuracy_graph_overall(results['configuration_results'], GRAPH_X_AXIS)

    draw_validation_set_loss_graph_overall(results['configuration_results'], GRAPH_X_AXIS)
    draw_training_set_loss_graph_overall(results['configuration_results'], GRAPH_X_AXIS)

    draw_validation_set_accuracy_graph_overall(results['configuration_results'], GRAPH_X_AXIS)
    draw_training_set_accuracy_graph_overall(results['configuration_results'], GRAPH_X_AXIS)

    for config_result in results['configuration_results']:
        draw_loss_history_graph_per_config(config_result)
        draw_accuracy_history_graph_per_config(config_result)