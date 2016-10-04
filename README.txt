REQUIREMENTS:
- keras
- theano
- numpy
- scikit-learn
- pandas
- matplotlib


SOLUTION STRUCTURE:
- Each problem has its own folder with its own model training Python file and any other required files (data, result reader, etc.)
- The raw data is initially split into multiple sets before the first experiment is executed
    - This ensures that each experiment is using the same data split
    - This is done by using data_splitter.py initially
    - It will generate a data_split.p file which contains anything related to the data splitting, including the original data, the normalized data, the mean and standard deviation used for normalization, etc.

- When experimenting with a set of configurations, it will generate a pickle file that contains everything related to that experiment:
    - Data used
    - Configuration used
    - The training and validation set accuracy and loss value (for each fold, if using k-folds CV)

- This result file is serialized and stored as a pickle file (.p extension) and can be read in the form of graphs using the result_reader.py file in each solution folder
- After deciding the best configuration, this configuration will be used in final_training.py
    - This final basically trains a new model that uses the full training and validation data
    - Then, we evaluate this new model using the testing data set that we have put aside from the beginning