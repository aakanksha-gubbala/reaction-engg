import numpy as np
import scipy.optimize as opt
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline

# layers of neurons
'''
inputs1 = [[1, 2, 3, 2.5],
          [2.0, 5.0, -1.0, 2.0],
          [-1.5, 2.7, 3.3, -0.8]]

weights1 = [[0.2, 0.8, -0.5, 1.0],
           [0.5, -0.91, 0.26, -0.5],
           [-0.26, -0.27, 0.17, 0.87]]

biases1 = [2, 3, 0.5]

layer1_outputs = np.dot(inputs1, np.array(weights1).T) + biases1

weights2 = [[0.1, -0.14, 0.5],
            [-0.5, 0.12, -0.33],
            [-0.44, 0.73, -0.13]]

biases2 = [-1, 2, 0.5]

layer2_outputs = np.dot(layer1_outputs, np.array(weights2).T) + biases2
'''
np.random.seed(0)

X = [[1, 2, 3, 2.5],
     [2.0, 5.0, -1.0, 2.0],
     [-1.5, 2.7, 3.3, -0.8]]


class Layer_Dense:
    def __init__(self, n_inputs, n_neurons):
        self.weights = 0.10 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))

    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases


layer1 = Layer_Dense(4, 5)
layer2 = Layer_Dense(5, 2)

layer1.forward(X)
layer2.forward(layer1.output)
# print(layer2.output)

inputs = [0, 2, -1, 3.3, -2.7, 1.1, 2.2, -100]


def forward(inputs):
    return np.maximum(0, inputs)


outputs = forward(inputs)
# print(outputs)


a = np.array([[1., 5.],
              [5., 1.]])
w, v = np.linalg.eig(a)

n_tanks = 2


def get_tau(tau):
    c0 = 0.016
    for i in range(n_tanks):
        c1 = opt.fsolve(lambda c: c - c0 + tau * (0.033 * c ** 2 + 0.006 * c), c0)
        c0 = c1
        # print("%d\t%0.4f\n" % (i+1, c1))
    return c1 - 0.0008


tau = opt.fsolve(get_tau, 0.01)
print(tau)
# print(tau*0.278)

'''c0 = 0.08
x = np.linspace(0, 0.1, 100)
y = 9.92 * x ** 2 + 0.1984 * x

plt.title(r"$\tau$ = %0.3f, number of CSTRs = %d" % (tau, n_tanks))
plt.plot(x, y)
plt.xlim(0, 0.1)
plt.ylim(0, 0.1)
plt.xlabel(r"$C_B\, (kmol/m^3)$")
plt.ylabel(r"$-r_B\, (kmol/m^3.ks)$")
for i in range(n_tanks):
    plt.plot([c0, c0], [9.92 * c0 ** 2 + 0.1984 * c0, 0], color="green")
    c1 = opt.fsolve(lambda c: c - c0 + tau * (9.92 * c ** 2 + 0.1984 * c), c0)
    plt.plot([c1, c0], [9.92 * c1 ** 2 + 0.1984 * c1, 0], color="green")
    c0 = c1
    print("%d\t%0.4f\n" % (i + 1, c1))

plt.plot([c1, c0], [9.92 * c1 ** 2 + 0.1984 * c1, 0], color="green")

# plt.savefig("%dtanks.png" % (n_tanks), dpi=300)
t = [23.522, 4.989, 2.527, 0.964, 0.519]
n = [1, 2, 3, 6, 10]
n_x = np.linspace(1, 10, 30)
t_y = make_interp_spline(n, t, k=1)(n_x)

plt.plot(n_x, t_y)
plt.xlabel("Number of tanks")
plt.ylabel(r"$\tau,\, (ks)}$")
# plt.savefig("ntanks-tau.png", dpi=300)

plt.show()'''

x = [5, 2, 1]
y = [0.167, 0.067, 0.036]

from scipy.optimize import curve_fit
from sklearn.metrics import accuracy_score


def func(x, max, ki):
    return max * x / (ki + x)


[max, ki], params = curve_fit(func, x, y)
print(max, ki)
print(func(1, 4.9, 142))
