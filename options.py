# -*- coding: utf-8 -*-
import argparse


class options(object):
    def __init__(self):
        parser = argparse.ArgumentParser()
        parser.description = '输入训练集的规模datascale,函数的四个参数a,b,c,d'
        parser.add_argument('--datascale', help="训练数据的规模", type=int)
        parser.add_argument('--a', help="第一个参数", type=int)
        parser.add_argument('--b', help="第二个参数", type=int)
        parser.add_argument('--c', help="第三个参数", type=int)
        parser.add_argument('--d', help="第四个参数", type=int)
        parser.add_argument('--width', help="每层神经元的个数", type=int)
        parser.add_argument('--depth', help="神经网络的深度", type=int)
        parser.add_argument('--learn_rate', help="学习率", type=float)
        parser.add_argument('--batch_size', help="batch的大小", type=int)
        parser.add_argument('--epoch', help="epoch次数", type=int)
        parser.add_argument('--activate', help="激活函数", type=str)
        parser.add_argument('--input_dim', help="输入维度", type=int)
        args = parser.parse_args()
        self.data_scale = args.datascale
        self.a = args.a
        self.b = args.b
        self.c = args.c
        self.d = args.d
        self.width = args.width
        self.depth = args.depth
        self.learn_rate = args.learn_rate
        self.batch_size = args.batch_size
        self.epoch = args.epoch
        self.activate = args.activate
        self.input_dim = args.input_dim

