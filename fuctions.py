# -*- coding: utf-8 -*-
import numpy as np
import math
from abc import ABC, abstractmethod


# def ReLU(b):
#     # ReLu 函数
#     b = b.astype(np.float)
#     r = ((np.abs(b) + b) / 2.0)
#     return r
#
#
# def ReLU_dev(m):
#     # ReLu 导数
#     m = m.astype(np.float)
#     dev = m
#     for i, element in enumerate(m):
#         if element[0] > 0:
#             dev[i][0] = 1
#         else:
#             dev[i][0] = 0
#     return dev
#
#
def quadratic_cost(a, y):
    return 0.5 * (a - y) * (a - y)


def quadratic_cost_dev(output, y):
    return output - y


#
#
# def sigmoid(a):
#     a = a.astype(np.float)
#     si = a
#     for i, element in enumerate(a):
#         si[i][0] = 1.0 / (1.0 + math.exp(-element[0]))
#     return si
#
#
# def sigmoid_dev(a):
#     ones = np.ones(a.shape)
#     return np.multiply(sigmoid(a), ones - sigmoid(a))
#
#
# def sigmoid_(z):
#     """The sigmoid function."""
#     return 1.0 / (1.0 + np.exp(-z))
#
#
# def sigmoid_prime(z):
#     """Derivative of the sigmoid function."""
#     return sigmoid_(z) * (1 - sigmoid_(z))


class activate_fuc(ABC):

    @abstractmethod
    def activate(self, b):
        pass

    @abstractmethod
    def activate_dev(self, m):
        pass


class leakyReLU(activate_fuc):
    def activate(self, b):
        b = b.astype(np.float)
        r = np.maximum(b, 0.01*b)
        return r

    def activate_dev(self, m):
        m = m.astype(np.float)
        dev = m
        for i, element in enumerate(m):
            if element[0] >= 0:
                dev[i][0] = 1
            else:
                dev[i][0] = 0.01
        return dev

class ReLU(activate_fuc):
    def activate(self, b):
        b = b.astype(np.float)
        r = ((np.abs(b) + b) / 2.0)
        # b = b.astype(np.float)
        # r = np.maximum(b, 0.01*b)
        return r

    def activate_dev(self, m):
        m = m.astype(np.float)
        dev = m
        for i, element in enumerate(m):
            if element[0] >= 0:
                dev[i][0] = 1
            else:
                dev[i][0] = 0
        return dev

class sigmoid(activate_fuc):
    def activate(self, b):
        """The sigmoid function."""
        return 1.0 / (1.0 + np.exp(-b))

    def activate_dev(self, m):
        """Derivative of the sigmoid function."""
        return self.activate(m) * (1 - self.activate(m))


class tanh(activate_fuc):
    def activate(self, b):
        return (np.exp(b) - np.exp(-b)) / (np.exp(b) + np.exp(-b))

    def activate_dev(self, m):
        return 1 - self.activate(m) * self.activate(m)


"""调试用代码："""
if __name__ == '__main__':
    x = [[-5.1], [-1.1], [0], [3], [4]]
    y = np.array(x)
    ac = ReLU()
    print('y = ', y, '\n')
    print('ReLU(y) = ', ac.activate(y), '\n')
    print('ReLU_dev(y) = ', ac.activate_dev(y), '\n')
