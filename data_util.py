#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import numpy as np
#  import torch
import pickle

import pdb


def load_data(data_path):

    dataset = dict()
    voca = None
    pad_index = 0
    index2word = {}

    if (not os.path.exists(data_path)):
        print("data not found")

    else:
        print("loading dataset")
        # num of train samples : 71420
        # each sample length : x=2000, title=25
        dataset['train_x'] = np.load(
            os.path.join(data_path, "train/train_body.npy"))
        dataset['train_title'] = np.load(
            os.path.join(data_path, "train/train_title.npy"))
        dataset['train_y'] = np.load(
            os.path.join(data_path, "train/train_label.npy"))

        # num of dev samples : 6302
        dataset['dev_x'] = np.load(os.path.join(data_path, "dev/dev_body.npy"))
        dataset['dev_title'] = np.load(
            os.path.join(data_path, "dev/dev_title.npy"))
        dataset['dev_y'] = np.load(os.path.join(data_path,
                                                "dev/dev_label.npy"))

        # num of test samples : 6302
        dataset['test_x'] = np.load(
            os.path.join(data_path, "test/test_body.npy"))
        dataset['test_title'] = np.load(
            os.path.join(data_path, "test/test_title.npy"))
        dataset['test_y'] = np.load(
            os.path.join(data_path, "test/test_label.npy"))

        with open(os.path.join(data_path, "dic_mincutN.txt"), "rb") as reader:
            dataset['dic_mincutN'] = reader.read()

        voca = pickle.load(open(data_path + "/dic_mincutN.pkl", "rb"),
                           encoding="utf-8")
        dataset['voca'] = voca
        dataset['pad_index'] = voca['']
        for w in voca:
            index2word[voca[w]] = w
        dataset['index2word'] = index2word

    return dataset
