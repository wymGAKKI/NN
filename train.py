# -*- coding: utf-8 -*-
from dataloader import data_loader
from options import options
from network import network
import numpy as np
import math
from math import pi
import matplotlib.pyplot as plt
import time

# python train.py --datascale 100 --a 1 --b 1 --c 1 --d 1 --width 3 --depth 3 --learn_rate 0.01 --batch_size 10 --epoch 1 --activate ReLU --inputdim 1 --no 1

path = "NN_HW/normal_sample/sin/%s/w=%s,d=%s,r=%s"

opt = options()
data = data_loader(opt)
net = network(opt)
time_start = time.time()

cost, time_three_hud = net.update_weights(data.data)
time_end = time.time()
time = time_end - time_start

x = np.linspace(-5, 5, 1000)
y = [net.forward(m)[0][0] for m in x]
sin_x = [math.sin(e) for e in x]

plt.figure(1)
plt.plot(x, sin_x, '--g')
plt.plot(x, y, '--r')
# 图片保存：
png_path = path + '.png'
png_value = (opt.activate, opt.width, opt.depth, opt.learn_rate)
plt.savefig(png_path % png_value)
plt.clf()
plt.figure(1)
plt.plot(range(len(net.costs)), net.costs)
cost_path = path + 'cost.png'
plt.savefig(cost_path % png_value)
#plt.show()
# 生成文本文件：
txt_path = path + '.txt'
txt_value = (opt.activate, opt.width, opt.depth, opt.learn_rate)
info_file = open(txt_path % txt_value, 'w')
info_str = 'width: %s\n depth: %s\n learn_rate: %s\n activate_fuction: %s \n\n\ntime: %s(s)\n500_time=%s \navg_cost: %s'
info_values = (opt.width, opt.depth, opt.learn_rate, opt.activate, time, time_three_hud, cost / opt.data_scale)
info_file.write(info_str % info_values)
