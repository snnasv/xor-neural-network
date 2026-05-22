import numpy as np


X = np.array([
    [0,0],
    [0,1],
    [1,0],
    [1,1]
])


y = np.array([
    [0],
    [1],
    [1],
    [0]
])


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_grad(x): # inatçılık seviyesi verilen cevap ne kadar kesi ise türevi de o kadar düşüktür yani kararından kolay kolay vaz geçmez
    return x * (1 - x)



np.random.seed(1)

W1 = np.random.uniform(-1, 1, (2, 2))

W2 = np.random.uniform(-1, 1, (2, 1))

b1 = np.zeros((1, 2))
b2 = np.zeros((1, 1))

for epoch in range (10000):

    hidden_layer_raw = np.dot(X, W1) + b1
    hidden_layer_output = sigmoid(hidden_layer_raw)


    output_layer_raw = np.dot(hidden_layer_output, W2) + b2
    guess = sigmoid(output_layer_raw)



    error = y - guess
    output_delta = error * sigmoid_grad(guess) #ceza verme formulü
    

    hidden_error = np.dot(output_delta, W2.T) #W2 de hangi nöronun bu hataya ne kadar suçu varsa bununla otomatik belirliyoruz ki hidden layera ceza keselim


    hidden_delta = hidden_error * sigmoid_grad(hidden_layer_output)

    W2 += np.dot(hidden_layer_output.T, output_delta)
    W1 += np.dot(X.T, hidden_delta)


    b2 += np.sum(output_delta, axis=0, keepdims=True)
    b1 += np.sum(hidden_delta, axis=0, keepdims=True)

    print("Eğitim Sonucu (Ağın Tahminleri):")
    print(guess)


