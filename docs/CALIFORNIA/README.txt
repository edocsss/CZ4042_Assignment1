Configurations:
Data split —> 70% training, 30% testing
3-folds on the 70% training data to find the best combination for a given single variable

Batch size = 256
Accuracy is not shown because it will always be 0 by definition.


Methods:
ALPHA:
Hidden Neurons: 15
Hidden Layers: 2
Epoch: 3000
Alpha: 0.00001, 0.0001, 0.001, 0.01, 0.1, 0.5

BEST —> 0.0001
ANALYSIS:



MAX_EPOCH:
Hidden Neurons: 15
Hidden Layers: 1
Epoch: 100, 500, 1000, 2000, 3000, 5000, 7500, 10000
Alpha: 0.0001

BEST —> 3000
ANALYSIS:



HIDDEN_LAYERS:
Hidden Neurons: 15
Hidden Layers: 1, 2, 3, 4
Epoch: 3000
Alpha: 0.0001

BEST —> 2
ANALYSIS:



HIDDEN_NEURONS:
Hidden Neurons: 1, 3, 5, 10, 15, 30, 50
Hidden Layers: 2
Epoch: 3000
Alpha: 0.0001

BEST —> 15
ANALYSIS:




The best configuration is:
Hidden Neurons: 15
Hidden Layer: 2
Epoch: 3000
Alpha: 0.0001

This configuration is found by combining all the best value for each variable. This configuration is balancing the tradeoff between the accuracy and the speed and performance needed for training a model.

By training a model configuration with this configuration on the whole 70% training data, the loss and accuracy on the testing set is:
Loss: 0.35415110497795743