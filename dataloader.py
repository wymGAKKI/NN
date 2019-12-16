# -*- coding: utf-8 -*-
"""
y=a*sin(bx)+c*cos(dx)
x:[-2*pi,2*pi]
"""
from options import options
import numpy as np
from math import sin, cos, pi


class data_loader(object):
    def __init__(self, opt):
        a = opt.a
        b = opt.b
        c = opt.c
        d = opt.d
        self.x = []
        scalar_x = np.random.rand(opt.data_scale)
        scalar_x = scalar_x * 4 * pi - 2 * pi
        for s in scalar_x:
            self.x.append(to_vector(s, opt.input_dim))
        self.y = [a * sin(b * element) + c * cos(d * element) for element in scalar_x]
        self.data = [(x, y) for x, y in zip(self.x, self.y)]


def to_vector(x, dim):
    v = np.zeros((dim, 1))
    for i in range(dim):
        v[i][0] = x ** ((i + 1) % 3)
    return v

# 调试用：：：
# def to_vector(x, dim):
#     v = np.zeros((4, 1))
#     v[0][0] = x
#     v[1][0] = x * x
#     v[2][0] = -x
#     v[3][0] = -x * x
#     return v

"""调试用代码："""
if __name__ == '__main__':
    # class fake_option(object):
    #     def __init__(self):
    #         self.a = 1
    #         self.b = 1
    #         self.c = 0
    #         self.d = 0
    #         self.data_scale = 10
    #
    #
    # opt = fake_option()
    # test_data = data_loader(opt)
    # print(test_data.data)
    x = 2
    print('x = ', x, '\n')
    print('vector_x = ', to_vector(x, 2), '\n')
    print('type of x_vector:', type(to_vector(x, 2)))
