import numpy as np
from sklearn.feature_extraction.text import CountVectorizer


messages = [
    "Earn cash fast, limited time offer",
    "Are we going to the library later?",
    "Claim your free reward before midnight",
    "Your payment receipt is attached here",
    "Congratulations! You have been selected to win",
    "Please share the assignment details with me"
]

# 1 = Spam, 0 = Ham
targets = np.array([[1, 0, 1, 0, 1, 0]]).T


text_vectorizer = CountVectorizer()
input_data = text_vectorizer.fit_transform(messages).toarray()


np.random.seed(42)

input_size = input_data.shape[1]
hidden_size = 5
output_size = 1

w_input_hidden = np.random.randn(input_size, hidden_size) * 0.1
w_hidden_output = np.random.randn(hidden_size, output_size) * 0.1


def sigmoid_fn(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_grad(x):
    return x * (1 - x)


learning_rate = 0.5
epochs = 1000


for _ in range(epochs):

    # Forward pass
    input_layer = input_data
    hidden_layer = sigmoid_fn(np.dot(input_layer, w_input_hidden))
    output_layer = sigmoid_fn(np.dot(hidden_layer, w_hidden_output))

    # Error calculation
    output_error = targets - output_layer
    output_delta = output_error * sigmoid_grad(output_layer)

    hidden_error = output_delta.dot(w_hidden_output.T)
    hidden_delta = hidden_error * sigmoid_grad(hidden_layer)

    # Weight update
    w_hidden_output += hidden_layer.T.dot(output_delta) * learning_rate
    w_input_hidden += input_layer.T.dot(hidden_delta) * learning_rate


test_message = ["Free money waiting for you"]
test_vector = text_vectorizer.transform(test_message).toarray()

hidden_test = sigmoid_fn(np.dot(test_vector, w_input_hidden))
final_prediction = sigmoid_fn(np.dot(hidden_test, w_hidden_output))


print("Test Message:", test_message[0])
print(f"Spam Probability: {final_prediction[0][0]:.4f}")
print("Result:", "Spam" if final_prediction > 0.5 else "Ham")

