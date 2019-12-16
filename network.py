# -*- coding: utf-8 -*-
import numpy as np
from fuctions import ReLU, sigmoid, leakyReLU, tanh
import random
from fuctions import quadratic_cost_dev
from fuctions import quadratic_cost
import matplotlib.pyplot as plt
from dataloader import data_loader
import math
from dataloader import to_vector
import time


class network(object):
    def __init__(self, opt):
        self.weights = []  # list of array
        self.bias = []  # list of array
        self.deltas = []  # list of array
        self.dims = [opt.input_dim, ]  # list
        self.learn_rate = opt.learn_rate
        self.batch_size = opt.batch_size
        self.epoch = opt.epoch
        self.costs = []
        if opt.activate == 'ReLU':
            self.activate = ReLU()
        if opt.activate == 'sigmoid':
            self.activate = sigmoid()
        if opt.activate == 'leakyReLU':
            self.activate = leakyReLU()
        if opt.activate == 'tanh':
            self.activate = tanh()
        # 初始化维数
        for d in range(opt.depth - 2):
            self.dims.append(opt.width)
        self.dims.append(1)
        # 初始化权重
        for index in range(opt.depth - 1):
            self.weights.append(np.random.randn(self.dims[index + 1], self.dims[index]))
        # 初始化偏移量, delta
        for index in range(opt.depth - 1):
            self.bias.append(np.random.randn(self.dims[index + 1], 1))
            self.deltas.append(np.zeros((self.dims[index + 1], 1)))

    def forward(self, a):
        a = to_vector(a, self.dims[0])
        for b, w in zip(self.bias, self.weights):
            z = np.dot(w, a) + b
            if z.shape == (1, 1):
                a = z
            else:
                a = self.activate.activate(z)
        return a

    def BP(self, x, y):
        dev_w = [np.zeros(w.shape) for w in self.weights]
        dev_b = [np.zeros(b.shape) for b in self.bias]
        activation = x
        activations = [x]
        zs = []
        for b, w in zip(self.bias, self.weights):
            z = w.dot(activation) + b
            if z.shape == (1, 1):
                activation = z
            else:
                activation = self.activate.activate(z)
            zs.append(z)
            activations.append(activation)
        delta = quadratic_cost_dev(zs[-1], y)
        dev_b[-1] = delta
        dev_w[-1] = np.dot(delta, activations[-2].transpose())
        for l in range(2, len(self.dims)):
            delta = np.dot(self.weights[-l + 1].transpose(), delta) * self.activate.activate_dev(zs[-l])
            dev_b[-l] = delta
            dev_w[-l] = np.dot(delta, activations[-l - 1].transpose())
        return dev_w, dev_b, quadratic_cost(activations[-1], y)

    def update_weights(self, train_data):
        cost_sum_epoch = 0
        time_s = time.time()
        time_e = time_s
        flag = 1
        for e in range(self.epoch):
            cost_sum_epoch = 0
            random.shuffle(train_data)
            batches = [train_data[i:i + self.batch_size] for i in range(0, len(train_data), self.batch_size)]
            for batch in batches:
                cost_sum_batch = 0
                sum_dev_w = [np.zeros(w.shape) for w in self.weights]
                sum_dev_b = [np.zeros(b.shape) for b in self.bias]
                for x, y in batch:
                    dev_w, dev_b, cost = self.BP(x, y)
                    cost_sum_batch += cost
                    sum_dev_w = [sw + dw for sw, dw in zip(sum_dev_w, dev_w)]
                    sum_dev_b = [sb + db for sb, db in zip(sum_dev_b, dev_b)]

                self.weights = [w - self.learn_rate * sw / len(batch) for w, sw in zip(self.weights, sum_dev_w)]
                self.bias = [b - self.learn_rate * sb / len(batch) for b, sb in zip(self.bias, sum_dev_b)]
                cost_sum_epoch += cost_sum_batch
            if cost_sum_epoch[0][0] < 500 and flag:
                time_e = time.time()
                flag = 0
                """"""
            self.costs.append(cost_sum_epoch[0][0])
            print(e, ' epoch cost:', cost_sum_epoch, '\n')
        return cost_sum_epoch, (time_e - time_s)



"""调试用代码："""
if __name__ == '__main__':
    class fake_option(object):
        def __init__(self):
            self.depth = 5
            self.width = 5
            self.learn_rate = 0.0015
            self.batch_size = 100
            self.epoch = 50
            self.activate = 'sigmoid'
            self.data_scale = 100
            self.a = 1
            self.b = 1
            self.c = 0
            self.d = 0
            self.input_dim = 4


    path = "NN_HW/normal_sample/sin/%s/w=%s,d=%s,r=%s"
    fake_opt = fake_option()
    net = network(fake_opt)
    # ___查看weights___:
    # print("weights' type:", type(net.weights), '\n')
    # print("weights[0] type:", type(net.weights[0]), '\n')
    # print("weights[0]", net.weights[0], '\n')
    # print("weights[0][0] type:", type(net.weights[0][0]), '\n')
    # print("weights[0][0]", net.weights[0][0], '\n')
    # print('weights: ', net.weights)
    # ___查看dims___:
    # print('dims: ', net.dims)
    # print('dims type:', type(net.dims))
    # 查看delta:
    # print("deltas: ", net.deltas, '\n')
    # print("deltas type: ", type(net.deltas), '\n')
    # print('delta[0]: ', net.deltas[0])
    # print('delta[0]: ', type(net.deltas[0]))
    # print('delta[0][0]: ', net.deltas[0][0])
    # print('delta[0][0]: ', type(net.deltas[0][0]))
    # 查看bias:
    # print("bias: ", net.bias, '\n')
    # print("bias type: ", type(net.bias), '\n')
    # print('bias[0]: ', net.bias[0])
    # print('bias[0] type: ', type(net.bias[0]))
    # print('bias[0][0]: ', net.bias[0][0])
    # print('bias[0][0] type: ', type(net.bias[0][0]))

    # 调试forward函数：

    # x = np.random.randint(1, 2, (1, 1))
    # y = net.forward(x)
    # print('weights: \n', net.weights, '\n')
    # print('bias: \n', net.bias, '\n')
    # print('x: ', x)
    # print('x type: ', type(x))
    # print('y: ', y)
    # print('y type:', type(y))
    # __测试激活函数__:
    # x = [[-1], [-1], [0], [3], [4]]
    # y = np.array(x)
    # print('ReLU(y) = ', net.activate.activate(y))
    # print('ReLU_dev(y) = ', net.activate.activate_dev(y))
    data = data_loader(fake_opt)
    time_start = time.time()
    cost, time_three_hud = net.update_weights(data.data)
    time_end = time.time()
    x = np.linspace(-5, 5, 1000)
    y = [net.forward(m)[0][0] for m in x]
    sin_x = [math.sin(e) for e in x]
    plt.figure(1)
    plt.plot(x, sin_x, '--g')
    plt.plot(x, y, '--r')
    # 图片保存：
    png_path = path + '.png'
    png_value = (fake_opt.activate, fake_opt.width, fake_opt.depth, fake_opt.learn_rate)
    plt.savefig(png_path % png_value)
    plt.clf()
    plt.figure(1)
    plt.plot(range(len(net.costs)), net.costs)
    cost_path = path + 'cost.png'
    plt.savefig(cost_path % png_value)
    # plt.show()
    # 生成文本文件：
    txt_path = path + '.txt'
    txt_value = (fake_opt.activate, fake_opt.width, fake_opt.depth, fake_opt.learn_rate)
    info_file = open(txt_path % txt_value, 'w')
    info_str = 'width: %s\n depth: %s\n learn_rate: %s\n activate_fuction: %s \n\n\ntime: %s(s)\n300_time=%s \navg_cost: %s'
    info_values = (fake_opt.width, fake_opt.depth, fake_opt.learn_rate, fake_opt.activate, time_end - time_start, time_three_hud, cost / fake_opt.data_scale)
    info_file.write(info_str % info_values)
