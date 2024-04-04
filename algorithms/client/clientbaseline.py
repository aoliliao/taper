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
from models.MLP import MLP
import clip
import torch.autograd as autograd

class clientTAPER_baseline(Client):
    def __init__(self, args, id, train_samples, test_samples, **kwargs):
        super().__init__(args, id, train_samples, test_samples, **kwargs)
        self.id = id
        # self.label_dict = {'bird': 0, 'feather': 1, 'headphones': 2, 'ice_cream': 3, 'teapot': 4, 'tiger': 5, 'whale': 6,
        #               'windmill': 7, 'wine_glass': 8, 'zebra': 9}
        self.domain = {'clipart':0, 'infograph':1, 'painting':2, 'quickdraw':3, 'real':4, 'sketch':5}
        self.loss = nn.CrossEntropyLoss()

        if self.dataset == "domainnet" or self.dataset == "minidomainnet":
            self.data_name = kwargs['data_name']
            self.train_loader = kwargs['train_loader']
            self.test_loader = kwargs['test_loader']
            self.labels = kwargs['task_label']
            # print(self.data_name, len(self.train_loader), len(self.test_loader))
        # self.model = self.set_model()

        # print(self.data_name, self.labels)
        # print(self.model.head.weight)
        # print(self.model.head.bias)

        self.task_optimizer = torch.optim.SGD(self.model.parameters(), lr=self.learning_rate,
                                              weight_decay=self.weight_decay, momentum=self.momentum)


    def set_model(self):
        new_weight = torch.zeros_like(self.model.head.weight.data)
        new_bias = torch.zeros_like(self.model.head.bias.data)
        for key, index in self.labels.items():
            vector = index
            new_weight[vector:vector+1] = self.model.head.weight.data[vector:vector+1]
            new_bias[vector:vector+1] = self.model.head.bias.data[vector:vector+1]

        # self.head = copy.deepcopy(self.model.head)
        self.model.head.weight.data = new_weight
        self.model.head.bias.data = new_bias


    def load_model(self, model, domain):
        model_path = os.path.join(self.save_folder_name,)
        model_path = os.path.join(model_path, f"basis_{self.data_name}" + ".pth")
        assert (os.path.exists(model_path))
        state_dict = torch.load(model_path)
        head = nn.Linear(512, self.num_classes)
        # print(model.head)
        model.base.load_state_dict(state_dict['base'])
        head.load_state_dict(state_dict['head'])
        return model, head


    def test_baseline(self):
        # self.set_model()
        stats = self.test_metrics()
        test_acc = stats[0] * 1.0 / stats[1]
        test_auc = stats[2] * 1.0 / stats[1]
        test_loss = stats[3] * 1.0 / stats[1]

        print('domain:', self.data_name, 'test_loss:', test_loss, '  test_acc:', test_acc * 100)
        return {self.data_name:test_acc}


    def test_metrics(self):
        if self.dataset == "domainnet" or self.dataset == "minidomainnet":
            test_loader_full = self.test_loader
        else:
            test_loader_full = self.load_test_data()
        # print(self.model.head.weight)
        # print(self.model.head.bias)
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


