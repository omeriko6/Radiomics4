import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestClassifier
import pickle


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


# Sigmoid activation function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


# Derivative of sigmoid function
def sigmoid_derivative(x):
    return sigmoid(x) * (1 - sigmoid(x))


def absolute_and_normalize(data):
    # Compute absolute value of the dataset
    abs_data = np.abs(data)

    # Normalize each column by dividing by the largest number in that column
    max_values = np.max(abs_data, axis=0)  # Get the largest number in each column
    normalized_data = abs_data / max_values

    return normalized_data

def clean_data(data):
    # Drop the first 26 columns
    data = data.iloc[:, 24:]

    # Drop rows containing NaN values
    cleaned_data = data.dropna()
    return cleaned_data




# Neural network class
class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        # Initialize weights and biases
        self.weights_input_hidden = np.random.randn(self.input_size, self.hidden_size)
        self.bias_hidden = np.random.randn(self.hidden_size)
        self.weights_hidden_output = np.random.randn(self.hidden_size, self.output_size)
        self.bias_output = np.random.randn(self.output_size)

    def forward(self, X):
        # Forward pass
        self.hidden_input = np.dot(X, self.weights_input_hidden) + self.bias_hidden
        self.hidden_output = sigmoid(self.hidden_input)
        self.output = sigmoid(np.dot(self.hidden_output, self.weights_hidden_output) + self.bias_output)
        return self.output

    def backward(self, X, y, output):
        # Backward pass
        j=0
        error = np.zeros((X.shape[0], 1))
        d_output= np.zeros((X.shape[0], 1))
        for i in y:
            error[j] = i - output[j]
            j=j+1

        d_output = error * sigmoid_derivative(output)


        error_hidden = np.dot(d_output, self.weights_hidden_output.T)
        d_hidden = error_hidden * sigmoid_derivative(self.hidden_output)

        # Update weights and biases
        self.weights_hidden_output += np.dot(self.hidden_output.T, d_output)
        self.bias_output += np.sum(d_output, axis=0)
        self.weights_input_hidden += np.dot(X.T, d_hidden)
        self.bias_hidden += np.sum(d_hidden, axis=0)


    def train(self, X, y, epochs):
        for epoch in range(epochs):
            output = self.forward(X)
            self.backward(X, y, output)
           # if epoch % 100 == 0:
            #    loss = np.mean(np.square(y - output))
             #   print(f'Epoch {epoch}, Loss: {loss}')

    def predict(self, X):
        # Make predictions using the trained neural network
        output = self.forward(X)
        return output


def load_data_from_csv(filename):
    # Load data
    data = pd.read_csv(filename)
    X = data.iloc[:, :-1].values
    y = data.iloc[:, -1].values.reshape(-1, 1)
    return X, y


def preprocess_data(filename):
    # Load data
    data = pd.read_csv(filename)

    # Handle missing values
    data.dropna(inplace=True)  # Drop rows with missing values

    # Handle string data
    string_columns = data.select_dtypes(include=['object']).columns
    for col in string_columns:
        le = LabelEncoder()
        data[col] = le.fit_transform(data[col])

    # Split data into features (X) and target variable (y)
    X = data.drop(columns=['targets'])
    y = data['targets']

    return X, y


def feature_importance_analysis(model, X, y):
    """
    Perform feature importance analysis for a neural network model.

    Parameters:
        model: The trained neural network model.
        X: Input features.
        y: Target variable.

    Returns:
        importance_scores: A dictionary mapping feature names to their importance scores.
    """
    importance_scores = {}

    # Iterate over each feature
    for feature_name in X.columns:
        # Make a copy of the data with the feature shuffled
        X_shuffled = X.copy()
        X_shuffled[feature_name] = np.random.permutation(X_shuffled[feature_name])

        # Calculate the difference in predictions
        predictions_original = model.predict(X)
        predictions_shuffled = model.predict(X_shuffled)
        diff = predictions_original - predictions_shuffled

        # Compute the importance score as the mean absolute difference
        importance_scores[feature_name] = np.mean(np.abs(diff))

    return importance_scores



def feature_selection(X_train, y_train, k):
    """
    Perform feature selection using Recursive Feature Elimination (RFE) with a Random Forest classifier.

    Parameters:
        X_train (DataFrame): Training features.
        y_train (Series): Training target variable.
        k (int): Number of top features to select (default is 10).

    Returns:
        X_selected (DataFrame): Selected features.
    """
    estimator = RandomForestClassifier(n_estimators=100)
    selector = RFE(estimator, n_features_to_select=k, step=1)
    X_selected = selector.fit_transform(X_train, y_train)
    selected_feature_indices = selector.get_support(indices=True)
    selected_feature_names = X_train.columns[selected_feature_indices]

    print("Selected Features:")
    for feature in selected_feature_names:
        print(feature)
    print("\n")
    X_selected = X_train.iloc[:, selected_feature_indices]

    return X_selected

def feature_engineering(X):
    """
    Perform feature engineering on input features.

    Parameters:
        X (DataFrame): Input features.

    Returns:
        X_engineered (DataFrame): Engineered features.
    """
    # Example feature engineering steps:
    # 1. Logarithmic transformation
    X_log = X.apply(lambda x: np.log(x + 1))

    # 2. Standardization
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

    # 3. Interaction features
    interaction_columns = []
    interaction_values = []
    for i in range(len(X.columns)):
        for j in range(i + 1, len(X.columns)):
            interaction_column_name = f"{X.columns[i]}_{X.columns[j]}_interaction"
            interaction_columns.append(interaction_column_name)
            interaction_values.append(X.iloc[:, i] * X.iloc[:, j])

    X_interaction = pd.DataFrame(np.column_stack(interaction_values), columns=interaction_columns)

    # Concatenate all engineered features
    X_engineered = pd.concat([X_log, X_scaled, X_interaction], axis=1)

    return X_engineered


if __name__ == '__main__':

    # Preprocess data

    X, y = preprocess_data('/Users/omergeffen/Desktop/DataBaseRadiomics/results_training.csv')
    X = absolute_and_normalize(X)
    X = clean_data(X)
    # Feature engineering
    #X_engineered = feature_engineering(X)

    # Feature selection

    X_selected = feature_selection(X, y, 10)




    # Split data into training and testing sets

    X_train, X_test, y_train, y_test = train_test_split(X_selected, y, test_size=0.2, random_state=None)

    input_size = X_train.shape[1]
    hidden_size = 64
    output_size = 1

    # Create neural network
    Radiomics4 = NeuralNetwork(input_size, hidden_size, output_size)

    # Train the model
    Radiomics4.train(X_train, y_train, epochs=100)


    file_path = '/Users/omergeffen/Desktop/DataBaseRadiomics/trained_model.pkl'

    # Save the trained model to a file with a specific path
    """
    with open(file_path, 'wb') as file:
        pickle.dump(Radiomics4, file)


    # Save X_test, y_test to CSV file

    X_test.to_csv('/Users/omergeffen/Desktop/DataBaseRadiomics/X_test.csv', index=False)  # Specify index=False to exclude row indices from the output

    print("X_test saved successfully.")

    y_test.to_csv('/Users/omergeffen/Desktop/DataBaseRadiomics/y_test.csv', index=False)  # Specify index=False to exclude row indices from the output

    print("y_test saved successfully.")
    """

    # Load the saved model from the file

    with open(file_path, 'rb') as file:
        loaded_model = pickle.load(file)

    # Load X_test, y_test into DataFrame

    X_testing = pd.read_csv('/Users/omergeffen/Desktop/DataBaseRadiomics/X_test.csv')
    y_test_loaded = pd.read_csv('/Users/omergeffen/Desktop/DataBaseRadiomics/y_test.csv')
    y_testing = y_test_loaded['targets']


    """
    
    #Perform feature importance analysis
    importance_scores = feature_importance_analysis(Radiomics4, X_train, y_train)

     #Print importance scores
    print("Feature Importance:")
    for feature_name, importance_score in importance_scores.items():
        if importance_score > 0.03:
            print(f"{feature_name}: {importance_score}")
    print("\n")
    """


    #Test the model
    # Print results
    #model result
    #predictions = Radiomics4.forward(X_test)

    #saved module result
    predictions = loaded_model.forward(X_testing)

    for i in range(len(predictions)):
        if predictions[i] > 0.5:
            predictions[i] = 1
        if predictions[i] < 0.5:
            predictions[i] = 0
    k = 0
    for i, value_ndarray in enumerate(predictions):
        value_series = y_testing.iloc[i]  # Get the value at the same index in the Series

        if value_ndarray == value_series:
            print(f"Values at index {i} are equal: {value_ndarray}")
            k=k+1
        else:
            print(f"Values at index {i} are not equal: ndarray={value_ndarray}, Series={value_series}")



    n = i+1  # Replace with the total number of predictions
    p_value = binomial_test(k, n, 0.5)  # Assuming random chance (p=0.5)
    print(f"for {k} successes")
    print("P-value:", p_value)


