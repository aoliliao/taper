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

class clientTAPER(Client):
    def __init__(self, args, id, train_samples, test_samples, **kwargs):
        super().__init__(args, id, train_samples, test_samples, **kwargs)
        self.id = id

        self.loss = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay, momentum=self.momentum)

        if self.dataset == "domainnet":
            self.data_name = kwargs['data_name']
            self.train_loader = kwargs['train_loader']
            self.test_loader = kwargs['test_loader']
            # self.labels = kwargs['task_label']
            # print(self.data_name, len(self.train_loader), len(self.test_loader))

    def set_base_model(self):
        self.base_model = []
        model_tmp = copy.deepcopy(self.model)
        datasets = ['clipart', 'infograph', 'painting', 'quickdraw', 'real', 'sketch']
        for index in range(len(datasets)):
            model_tmp = self.load_model(model_tmp, datasets[index])
            self.base_model.append(model_tmp)

    def load_model(self, model, domain):
        model_path = os.path.join(self.save_folder_name,)
        model_path = os.path.join(model_path, f"basis_{self.data_name}" + ".pth")
        assert (os.path.exists(model_path))
        state_dict = torch.load(model_path)
        model.load_state_dict(state_dict['model'])
        return model

    def save_client_model(self):
        model_path = os.path.join(self.save_folder_name,)
        # model_path = './save'
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        model_path = os.path.join(model_path,  f"basis_{self.data_name}" + ".pth")
        torch.save({'model': self.model.state_dict(),
                    'base': self.model.base.state_dict(),
                    'head': self.model.head.state_dict(), }, model_path)

    def train_stage_two(self):
        # self.load_model()
        if self.dataset == "domainnet":
            trainloader = self.train_loader
        else:
            trainloader = self.load_train_data()

        # self.model.to(self.device)

        max_local_steps = self.local_steps
        for step in range(max_local_steps):
            stats = self.test_metrics()
            stats_train = self.train_metrics()
            test_acc = stats[0] * 1.0 / stats[1]
            test_auc = stats[2] * 1.0 / stats[1]
            test_loss = stats[3] * 1.0 / stats[1]
            train_loss = stats_train[0] * 1.0 / stats_train[1]
            train_acc = stats_train[2] * 1.0 / stats_train[1]
            print('domain:',self.data_name,'train_loss:',train_loss,'  train_acc:',train_acc*100)
            print('domain:', self.data_name, 'test_loss:', test_loss, '  test_acc:', test_acc * 100)

            self.model.train()
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

            self.save_client_model()

    def test_metrics(self):
        if self.dataset == "domainnet" or self.dataset == "minidomainnet":
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
        if self.dataset == "domainnet" or self.dataset == "minidomainnet":
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
