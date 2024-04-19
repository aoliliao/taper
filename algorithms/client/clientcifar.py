import copy
import os
import random

import torch
import torch.nn as nn
import numpy as np
import time

from sklearn import metrics
from sklearn.preprocessing import label_binarize

from algorithms.client.client import Client
from models.MLP import MLP, ConvNet_100
import clip
import torch.autograd as autograd


class clientcifar(Client):
    def __init__(self, args, id, train_samples, test_samples, **kwargs):
        super().__init__(args, id, train_samples, test_samples, **kwargs)
        self.id = id
        self.loss = nn.CrossEntropyLoss()

        if self.dataset == "cifar100":
            self.trainset = kwargs['train_loader']
            self.testset = kwargs['test_loader']
            self.labels = kwargs['task_label']
        print(self.labels)
        self.new_label_mapping = {}
        for i, key in enumerate(self.labels.keys()):
            self.new_label_mapping[key] = i
        self.trainset = [(img, self.new_label_mapping[original_label]) for img, original_label in self.trainset]
        self.testset = [(img, self.new_label_mapping[original_label]) for img, original_label in self.testset]

        self.train_loader = torch.utils.data.DataLoader(self.trainset, batch_size=args.batch_size, shuffle=True,)
        self.test_loader = torch.utils.data.DataLoader(self.testset, batch_size=args.batch_size, shuffle=False,)
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay, momentum=self.momentum)


    def set_base_model(self):
        global_model = ConvNet_100().to(self.device)
        model_path = os.path.join(self.save_folder_name, "conv_stageone" + ".pth")
        assert (os.path.exists(model_path))
        state_dict = torch.load(model_path, map_location='cuda')
        global_model.load_state_dict(state_dict['model'])
        for key in global_model.state_dict().keys():
            if 'fc' not in key:
                # print(key)
                self.model.state_dict()[key].data.copy_(global_model.state_dict()[key])

        new_weight = torch.zeros_like(self.model.fc.weight.data)
        new_bias = torch.zeros_like(self.model.fc.bias.data)
        for key, index in self.new_label_mapping.items():
            vector = index
            new_weight[vector:vector + 1] = global_model.fc.weight.data[key:key + 1]
            new_bias[vector:vector + 1] = global_model.fc.bias.data[key:key + 1]
        self.model.fc.weight.data = new_weight
        self.model.fc.bias.data = new_bias


    def train_stage_two(self):
        self.set_base_model()
        if self.dataset == "cifar100":
            trainloader = self.train_loader
        else:
            trainloader = self.load_train_data()

        max_local_steps = self.local_steps

        for step in range(max_local_steps):
            # print('epoch',step)
            stats = self.test_metrics()
            stats_train = self.train_metrics()

            test_acc = stats[0] * 1.0 / stats[1]
            test_auc = stats[2] * 1.0 / stats[1]
            test_loss = stats[3] * 1.0 / stats[1]
            train_loss = stats_train[0] * 1.0 / stats_train[1]
            train_acc = stats_train[2] * 1.0 / stats_train[1]
            # print('train_loss:',train_loss,'  train_acc:',train_acc)
            # print('test_loss:', test_loss, '  test_acc:', test_acc)
            self.model.train()
            # self.task_model.make_per_model()
            for i, (x, y) in enumerate(trainloader):
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)
                self.optimizer.zero_grad()
                output = self.model(x)
                loss = self.loss(output, y)
                loss.backward()
                self.optimizer.step()
        print('final test time')
        stats = self.test_metrics()
        test_acc = stats[0] * 1.0 / stats[1]
        test_loss = stats[3] * 1.0 / stats[1]
        print('test_loss:', test_loss, '  test_acc:', test_acc)
        class_str = ''
        for key, index in self.labels.items():
            class_str += index.replace('_', '') + '_'
        # print(class_str)
        model_path = os.path.join(self.save_folder_name, )
        # model_path = './save'
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        model_path = os.path.join(model_path, "conv_" + class_str + f"acc{int(test_acc*100)}_1" + ".pth")
        torch.save({'model': self.model.state_dict(),
                    'acc': test_acc, }, model_path)

    def load_model(self):
        model_path = os.path.join(self.save_folder_name, )
        model_path = os.path.join(model_path, "conv_" + 'bear_cup_baby_rose_bee_plain_mountain_road_otter_seal_' + f"acc67_2" + ".pth")
        assert (os.path.exists(model_path))
        state_dict = torch.load(model_path, map_location='cuda')
        self.model.load_state_dict(state_dict['model'])
        acc = state_dict['acc']
        print(acc)

    def test_metrics(self):
        if self.dataset == "cifar100":
            test_loader_full = self.test_loader
        else:
            test_loader_full = self.load_test_data()

        self.model.eval()

        test_acc = 0
        test_num = 0
        loss = 0
        y_prob = []
        y_true = []

        with torch.no_grad():
            for x, y in test_loader_full:
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)
                output = self.model(x)

                test_acc += (torch.sum(torch.argmax(output, dim=1) == y)).item()
                test_num += y.shape[0]
                loss += self.loss(output, y).item() * y.shape[0]
                y_prob.append(output.detach().cpu().numpy())
                nc = self.num_classes
                if self.num_classes == 2:
                    nc += 1
                lb = label_binarize(y.detach().cpu().numpy(), classes=np.arange(nc))
                if self.num_classes == 2:
                    lb = lb[:, :2]
                y_true.append(lb)

        y_prob = np.concatenate(y_prob, axis=0)
        y_true = np.concatenate(y_true, axis=0)

        auc = metrics.roc_auc_score(y_true, y_prob, average='micro')
        # print(self.id, "_testAcc:", test_acc * 1.0 / test_num)
        self.test_acc.append(test_acc / test_num)
        self.test_loss.append(loss / test_num)

        return test_acc, test_num, auc, loss

    def train_metrics(self):
        if self.dataset == "cifar100":
            train_loader = self.train_loader
        else:
            train_loader = self.load_train_data()

        self.model.eval()
        train_acc = 0
        train_num = 0
        loss = 0
        for x, y in train_loader:
            if type(x) == type([]):
                x[0] = x[0].to(self.device)
            else:
                x = x.to(self.device)
            y = y.to(self.device)
            output = self.model(x)
            train_acc += (torch.sum(torch.argmax(output, dim=1) == y)).item()
            train_num += y.shape[0]
            loss += self.loss(output, y).item() * y.shape[0]

        return loss, train_num, train_acc
