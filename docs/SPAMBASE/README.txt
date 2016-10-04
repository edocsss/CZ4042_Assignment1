Configurations:
Data split —> 70% training, 30% testing
3-folds on the 70% training data to find the best combination for a given single variable


Methods:
ALPHA:
Hidden Neurons: 50
Hidden Layers: 1
Epoch: 3000
Alpha: 0.0001, 0.001, 0.01, 0.1, 0.5, 1

BEST —> 0.1
ANALYSIS:
- When the alpha is too small, the error is going down very slowly. This is because of the gradient descent algorithm (need to show the equation). Since the alpha is small, the change in the gradient in each epoch is smaller and hence, the error converges slowly.

- There is no significant difference when the learning rate is >= 0.1. When the learning rate is near to 1.0, the network may miss the global minima of the function and hence there are spikes in the error convergence graph (see Figure X).

- The learning rate 0.1 seems to be the best as it has the lowest loss for the validation set. The training set also has a good enough error rate with 0.1 learning rate.



MAX_EPOCH:
Hidden Neurons: 50
Hidden Layers: 1
Epoch: 100, 500, 1000, 2000, 3000, 5000, 7500, 10000
Alpha: 0.1

BEST —> 3000
ANALYSIS:
- With a bigger max epoch and a constant learning rate that is small enough, the network may overfit the training data. This can be seen from the graph. When max epoch is approaching 10000, the error on the training data is decreasing while the error on the validation data is increasing. The overfitting may be because when the learning rate is small, the weights of the network change by a small value in every epoch and so the network may find the optimal weights for the “training data” but not generalised for the testing data.

- The epoch 3000 should be chosen as it is not overfitting the training data and the validation set has a relatively lower error than other max epoch value.



HIDDEN_LAYERS:
Hidden Neurons: 50
Hidden Layers: 1, 2, 3, 4
Epoch: 3000
Alpha: 0.1

BEST —> 1
ANALYSIS:
- A network with higher number of hidden layers is able to solve a more complex problem with higher degree of non-linearity. In this spam problem, the classes may not need a complex solution with high degree of non-linearity. As can be seen in Figure X, when the number of hidden layer is increased, the training set error rate increases. The validation set error goes down probably because the weight is more right for the validation set. This happens because the network may not find the best weights for a problem with lower degree of non-linearity using multiple hidden layers which usually is for higher degree non-linearity problem.

- The network should use 1 hidden layer as it has the lowest error rate for validation set and it is less demanding in terms of computation power.



HIDDEN_NEURONS:
Hidden Neurons: 1, 5, 10, 30, 50, 100, 200, 300
Hidden Layers: 1
Epoch: 3000
Alpha: 0.1

BEST —> 50
ANALYSIS:
- The number of hidden neurons depends on a number of factors:
	- Number of input nodes
	- Number of output nodes
	- Function complexity
	- Training algorithm

- Generally, if the network is using too few nodes, the network may not be able to learn the training data well. As a result, the network does not learn the predictive factors of the classification.

- On the other hand, if the network is using too many nodes, the network may learn the training data too well resulting in overfitting problem. As a result, the validation set error will be high as the network cannot generalise good enough.

- In this case, it is found that 50 is quite optimal for the number of hidden neutrons for this particular problem. In this problem, it seems having more hidden neurons does not really reduce the loss.




The best configuration is:
Hidden Neurons: 50
Hidden Layer: 1
Epoch: 3000
Alpha: 0.1

This configuration is found by combining all the best value for each variable. This configuration is balancing the tradeoff between the accuracy and the speed and performance needed for training a model.

By training a model configuration with this configuration on the whole 70% training data, the loss and accuracy on the testing set is:
Loss: 0.050789242791823834
Accuracy: 0.93989862418537296