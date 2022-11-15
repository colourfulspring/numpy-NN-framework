import numpy as np
import matplotlib.pyplot as plt
from NN import *

# dataset
x1 = np.array([[1.58, 2.32, -5.8], [0.67, 1.58, -4.78], [1.04, 1.01, -3.63],
                   [-1.49, 2.18, -3.39], [-0.41, 1.21, -4.73], [1.39, 3.16, 2.87],
                   [1.20, 1.40, -1.89], [-0.92, 1.44, -3.22], [0.45, 1.33, -4.38],
                   [-0.76, 0.84, -1.96]])
x2 = np.array([[0.21, 0.03, -2.21], [0.37, 0.28, -1.8], [0.18, 1.22, 0.16],
                   [-0.24, 0.93, -1.01], [-1.18, 0.39, -0.39], [0.74, 0.96, -1.16],
                   [-0.38, 1.94, -0.48], [0.02, 0.72, -0.17], [0.44, 1.31, -0.14],
                   [0.46, 1.49, 0.68]])
x3 = np.array([[-1.54, 1.17, 0.64], [5.41, 3.45, -1.33], [1.55, 0.99, 2.69],
                   [1.86, 3.19, 1.51], [1.68, 1.79, -0.87], [3.51, -0.22, -1.39],
                   [1.40, -0.44, -0.92], [0.44, 0.83, 1.97], [0.25, 0.68, -0.99],
                   [0.66, -0.45, 0.08]])
one = np.ones((10, 1))
data1 = np.concatenate((x1, one), axis=1)
data2 = np.concatenate((x2, one), axis=1)
data3 = np.concatenate((x3, one), axis=1)

label1 = np.zeros((10, 3))
class1 = np.full((10, 3), 1)
np.put_along_axis(label1, class1 - 1, 1.0, 1)
label2 = np.zeros((10, 3))
class2 = np.full((10, 3), 2)
np.put_along_axis(label2, class2 - 1, 1.0, 1)
label3 = np.zeros((10, 3))
class3 = np.full((10, 3), 3)
np.put_along_axis(label3, class3 - 1, 1.0, 1)

data = np.concatenate((data1, data2, data3))
label = np.concatenate((label1, label2, label3))

single_point = np.arange(1, data.shape[-2])
single_data = np.split(data, single_point, axis=-2)
single_label = np.split(label, single_point, axis=-2)

# module
class Module(object):
    def __init__(self, input_dim, hidden_dim, output_dim):
        self.__W1 = Tensor(np.random.normal(0, 2, (input_dim, hidden_dim)), requires_grad=True)
        self.__W2 = Tensor(np.random.normal(0, 2, (hidden_dim, output_dim)), requires_grad=True)

    def forward(self, data):
        net1 = data @ self.__W1
        h1 = net1.tanh()
        net2 = h1 @ self.__W2
        y = net2.sigmoid()
        return y
    
    def step(self, lr):
        self.__W1.val = self.__W1.val - lr * self.__W1.grad
        self.__W2.val = self.__W2.val - lr * self.__W2.grad

    def zero_grad(self):
        self.__W1.zero_grad()
        self.__W2.zero_grad()

# training
lr = 0.1
epochs = 1000
fig, ax = plt.subplots()
hidden_dims = [5, 10, 20, 30, 50]

for hidden in hidden_dims:
    loss_list = []
    net = Module(4, hidden, 3)

    for epoch in range(epochs):
        epoch_loss = []
        for i in range(len(single_data)):
            y = net.forward(Tensor(single_data[i]))
            loss = (y - Tensor(single_label[i])) ** 2.0
            loss.backward()
            net.step(lr)
            net.zero_grad()
            epoch_loss.append(np.sum(loss.val))
            
        loss_list.append(np.mean(epoch_loss))
    
    print(f"Training with hidden_dim = {hidden} finished.")

    x = np.arange(0, epochs)
    loss = np.array(loss_list)
    plt.plot(x, loss_list)

plt.legend([f"Nodes={hidden}" for hidden in hidden_dims], loc="best")
ax.set(xlabel='epochs', ylabel='loss', title='Single Sample Method Loss With Different Hidden Nodes')
plt.show()