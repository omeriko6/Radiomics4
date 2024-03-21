
import pandas as pd
import numpy as np


def normalize_columns(arr):
    """
    Normalize each column of a NumPy ndarray based on the maximum absolute value in that column.

    Parameters:
        arr (ndarray): Input NumPy array

    Returns:
        ndarray: Normalized array
    """
    # Compute the maximum absolute value in each column
    max_abs_values = np.max(np.abs(arr), axis=0)

    # Normalize each column based on its maximum absolute value
    normalized_arr = arr / max_abs_values

    return normalized_arr

def binomial_test(k, n, p):
    """
    Perform a binomial test.

    Parameters:
        k (int): Number of successes.
        n (int): Total number of trials.
        p (float): Hypothesized probability of success.

    Returns:
        float: Two-tailed p-value.
    """
    from math import comb

    # Calculate the probability of obtaining k or more successes
    p_value = sum(comb(n, i) * (p ** i) * ((1 - p) ** (n - i)) for i in range(k, n + 1))

    # For a two-tailed test, double the p-value if k is greater than n/2
    if k > n / 2:
        p_value *= 2

    return p_value

def relu(x):
    return np.maximum(0, x)

# Define the sigmoid activation function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Define the derivative of the sigmoid function
def sigmoid_derivative(x):
    return x * (1 - x)

# Define the neural network class
class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size):
        # Randomly initialize weights and biases
        self.weights_input_hidden = np.random.rand(input_size, hidden_size)
        self.bias_hidden = np.zeros((1, hidden_size))
        self.weights_hidden_output = np.random.rand(hidden_size, output_size)
        self.bias_output = np.zeros((1, output_size))

    def forward(self, inputs):
        # Input to hidden layer
        hidden_input = np.dot(inputs, self.weights_input_hidden) + self.bias_hidden
        hidden_output = normalize_columns(hidden_input)

        # Hidden to output layer
        output_input = np.dot(hidden_output, self.weights_hidden_output) + self.bias_output
        final_output = sigmoid(output_input)

        return final_output, hidden_output

    def train(self, inputs, targets, learning_rate, epochs):
        #threshold = 0.8
        for epoch in range(epochs):
            
            # Forward pass

            output, hidden = self.forward(inputs)
            error = np.array([0, 0, 0, 0, 0, 0])

            # Compute the error
            error = targets - output
            error1 = np.transpose(np.array([error[0],]))
            # Backpropagation
            output_error = error1*(sigmoid_derivative(output))
            hidden_error = output_error.dot(self.weights_hidden_output.T) * sigmoid_derivative(hidden)

            # Update weights and biases
            self.weights_hidden_output += hidden.T.dot(output_error) * learning_rate
            self.bias_output += np.sum(output_error, axis=0, keepdims=True) * learning_rate
            self.weights_input_hidden += inputs.T.dot(hidden_error) * learning_rate
            self.bias_hidden += np.sum(hidden_error, axis=0, keepdims=True) * learning_rate

            # Print the mean squared error at each epoch
            mse = np.mean(error**2)
           # print(f'Epoch {epoch + 1}/{epochs}, Mean Squared Error: {mse}')
        # print(self.weights_hidden_output)





    def get_most_important_features(self, threshold):
            # Compute the average of the weights for each input feature

            avg_weight = np.abs(self.weights_input_hidden).mean(axis=1)


            # Identify features above the threshold

            important_features = np.where(avg_weight > threshold)[0]

            print(important_features)
            return important_features



def inference(input_data, neural_network):
    # Forward pass through the neural network
    output, _ = neural_network.forward(input_data)

    # Make system decision based on output (e.g., thresholding for binary classification)
    # Example: If output > 0.5, classify as class 1, otherwise classify as class 0
    decision = 1 if output > 0.5 else 0

    return decision



if __name__ == "__main__":
    # Load the CSV file into a DataFrame
    df = pd.read_csv('/Users/omergeffen/Desktop/DataBaseRadiomics/results_training.csv')
    input_test = pd.read_csv('/Users/omergeffen/Desktop/DataBaseRadiomics/results_testing.csv')
    features = pd.read_csv('/Users/omergeffen/Desktop/DataBaseRadiomics/features.csv')

    # Drop columns containing string values
    df = df.select_dtypes(exclude=['object'])
    input_test = input_test.select_dtypes(exclude=['object'])

    # Perform additional preprocessing if needed
    # Example: Handle missing values, scale features, etc.
    # Replace NaN values with zero
    df.fillna(0, inplace=True)
    input_test.fillna(0, inplace=True)

    input_data = input_test.drop('targets', axis=1).values
    input_data = input_data / input_data.max(axis=0)
    input_data = input_data[:, 5:]

    input_results = input_test['targets'].values

    # Convert DataFrame to NumPy array and ensure data type is float64
   # input_data = input_test.values.astype(np.float64)





    # Assuming the target variable is in the 'targets' column
    # Extract input features and target labels
    X = df.drop('targets', axis=1).values

    # Make every negative value its absolute value
    X = np.abs(X)

    # Normalize each column in X with respect to the largest value in each column
    X_normalized = X / X.max(axis=0)
    X_normalized = X_normalized[:, 5:]
    y = df['targets'].values  # Reshape to ensure it's a column vector

    # Normalize input features (optional but recommended)
    # Example:
    # from sklearn.preprocessing import StandardScaler
    # scaler = StandardScaler()
    # X = scaler.fit_transform(X)

    # Assuming you have defined the NeuralNetwork class as shown above
    # Create and train the neural network
   # neural_network = NeuralNetwork(input_size=df.shape[1]-1, hidden_size=64, output_size=1)
    #neural_network.train(X, y, learning_rate=0.01, epochs=10)

   # for row in df.iterrows():
    #    X_row = df.drop('targets', axis=1).values
     #   y_row = df['targets'].values  # Reshape to ensure it's a column vector
      #  neural_network.train(X_row, y_row, learning_rate=0.01, epochs=100)

   # print_weights_greater_than(neural_network, 0.8)
    # Make inference
  #  importance_threshold = 0.55


    # Get the most important features
  #  important_features = neural_network.get_most_important_features(importance_threshold)

   # print("Most important features (indices) with importance above", importance_threshold, ":")
    #print(important_features)

  #  system_decision = inference(input, neural_network)

   # print("System Decision:", system_decision)

 # Define the number of runs and importance threshold

# Create the neural network
neural_network = NeuralNetwork(input_size=X_normalized.shape[1], hidden_size=50, output_size=1)
neural_network.train(X_normalized, y, learning_rate=0.01, epochs=100)

# Initialize an empty dictionary to keep track of selected features and their counts
feature_counts = {}

# Assuming you have defined the NeuralNetwork class as shown above
# Create and train the neural network
# Run the script 10 times
"""
for _ in range(10):

    #neural_network = NeuralNetwork(input_size=df.shape[1] - 1, hidden_size=64, output_size=1)
    #neural_network.train(X_normalized, y, learning_rate=0.01, epochs=100)

    importance_threshold = 0.56
    important_features = neural_network.get_most_important_features(importance_threshold)

    # Update the feature counts
    for feature in important_features:
        if feature in feature_counts:
            feature_counts[feature] += 1
        else:
            feature_counts[feature] = 1


# Get the 3 most common features

print("\n")
most_common_feature = max(feature_counts, key=feature_counts.get)
print("Most  important features:")
print(most_common_feature)
feature_counts[most_common_feature] = 0

most_common_feature = max(feature_counts, key=feature_counts.get)
print(most_common_feature)
feature_counts[most_common_feature] = 0

most_common_feature = max(feature_counts, key=feature_counts.get)
print( most_common_feature)

"""
importance_threshold = 0.5
print("Low importance:")
neural_network.get_most_important_features(importance_threshold)

print("\nMedium importance:")
importance_threshold = 0.58
neural_network.get_most_important_features(importance_threshold)
print("\nHigh importance:")
importance_threshold = 0.60
important_features = neural_network.get_most_important_features(importance_threshold)


X_selected = X[:, important_features]


neural_network = NeuralNetwork(input_size=len(important_features), hidden_size=30, output_size=1)
neural_network.train(X_selected, y, learning_rate=0.01, epochs=100)

importance_threshold = 0.53
important_features = neural_network.get_most_important_features(importance_threshold)

 # Update the feature counts
for feature in important_features:
     if feature in feature_counts:
          feature_counts[feature] += 1
     else:
         feature_counts[feature] = 1

# Get the most common feature
most_common_feature = max(feature_counts, key=feature_counts.get)

print("Most selected important feature:", most_common_feature)
counter = 0
k = 0  # Replace with the actual number of successful predictions

for row in input_data:
    if row[0] != 0:
         system_decision = inference(row, neural_network)
         if system_decision == 1:
            print("System has Identified BRCA attributes")
         if system_decision == 0:
            print("System has not Identified BRCA attributes")
         if system_decision == input_results[counter]:
            k = k+1
    counter = counter + 1

# p-value calculation:


n = counter  # Replace with the total number of predictions
p_value = binomial_test(k, n, 0.5)  # Assuming random chance (p=0.5)
print("P-value:", p_value)
