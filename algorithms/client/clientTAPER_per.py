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

class task_model(nn.Module):
    def __init__(self, text_embedding, base_model, classifer=None):
        super(task_model, self).__init__()
        # self.block_key = ['layer2', 'layer3', 'layer4']
        # self.block_key = [['layer_2', 'layer_3'], ['layer_4', 'layer_5'], ['layer_6', 'layer_7']]
        self.mixer1 = MLP(out_features=6, )
        self.text_embedding = text_embedding
        self.base_model = base_model
        self.per_model = copy.deepcopy(base_model[0])

        self.classifer = classifer

    def make_per_model(self, ):
        uploaded_weights = self.mixer1(self.text_embedding)
        # print(uploaded_weights[-1])
        for i in range(len(uploaded_weights[-1])):
            if i == 0:
                model_param = uploaded_weights[-1][i] * self.base_model[i].base.get_param(clone=False)
            else:
                model_param = model_param.add(uploaded_weights[-1][i] * self.base_model[i].base.get_param(clone=False))

        return model_param


    def forward(self, x,):
        model_param = self.make_per_model()
        output = self.per_model.base.forward_with_param(x, model_param)
        # print(output.shape)
        output = self.classifer(output)
        # print(output.shape)
        return output

class clientTAPER_per(Client):
    def __init__(self, args, id, train_samples, test_samples, **kwargs):
        super().__init__(args, id, train_samples, test_samples, **kwargs)
        self.id = id
        # self.label_dict = ['bird', 'feather', 'headphones', 'ice_cream', 'teapot', 'tiger', 'whale', 'windmill', 'wine_glass', 'zebra']
        # self.label_dict = {'bird': 0, 'feather': 1, 'headphones': 2, 'ice_cream': 3, 'teapot': 4, 'tiger': 5,
        #                    'whale': 6, 'windmill': 7, 'wine_glass': 8, 'zebra': 9}
        self.domain = {'clipart': 0, 'infograph': 1, 'painting': 2, 'quickdraw': 3, 'real': 4, 'sketch': 5}
        self.loss = nn.CrossEntropyLoss()

        if self.dataset == "domainnet" or self.dataset == "minidomainnet":
            self.data_name = kwargs['data_name']
            self.train_loader = kwargs['train_loader']
            self.test_loader = kwargs['test_loader']
            self.labels = kwargs['task_label']
            # print(self.data_name, len(self.train_loader), len(self.test_loader))

        task_descriptions = [f"This is a {self.data_name} {label}" for label in self.labels]
        print(self.data_name,'--',task_descriptions)
        self.text_model, preprocess = clip.load("RN50", device=self.device)
        text = clip.tokenize(task_descriptions).to(self.device)
        with torch.no_grad():
            text_features = self.text_model.encode_text(text)
            self.text_features = text_features.mean(0).to(torch.float32).unsqueeze(0)
        self.set_base_model()
        # print(self.base_model[1])

        # print(self.head.weight)
        # print(self.head.bias)
        self.task_model = task_model(self.text_features, self.base_model, self.head).to(self.device)
        self.task_optimizer = torch.optim.SGD(self.task_model.parameters(), lr=self.learning_rate,
                                              weight_decay=self.weight_decay, momentum=self.momentum)


    def set_base_model(self):
        base_model = []
        datasets = ['clipart', 'infograph', 'painting', 'quickdraw', 'real', 'sketch']
        for index in range(len(datasets)):
            model_tmp = copy.deepcopy(self.model)
            model_tmp, head = self.load_model(model_tmp, datasets[index])

            if datasets[index] == self.data_name:
                self.head = copy.deepcopy(head)

                new_weight = torch.zeros_like(self.head.weight.data)
                new_bias = torch.zeros_like(self.head.bias.data)
                for key, index in self.labels.items():
                    vector = index
                    new_weight[vector:vector + 1] = self.head.weight.data[vector:vector + 1]
                    new_bias[vector:vector + 1] = self.head.bias.data[vector:vector + 1]

                # self.head = copy.deepcopy(self.model.head)
                self.head.weight.data = new_weight
                self.head.bias.data = new_bias
                # print(self.head.weight)
            model_tmp.fc = nn.Identity()
            base_model.append(model_tmp)
        self.base_model = nn.ModuleList([base_model[i] for i in range(len(base_model))])

    def load_model(self, model, domain):
        model_path = os.path.join(self.save_folder_name,)
        model_path = os.path.join(model_path, f"basis_{self.data_name}" + ".pth")
        assert (os.path.exists(model_path))
        state_dict = torch.load(model_path, map_location='cuda')
        head = nn.Linear(512, self.num_classes)
        # print(model.head)
        model.base.load_state_dict(state_dict['base'])
        head.load_state_dict(state_dict['head'])
        return model, head


    def train_stage_three(self):
        # self.load_model()
        if self.dataset == "domainnet" or self.dataset == "minidomainnet":
            trainloader = self.train_loader
        else:
            trainloader = self.load_train_data()

        # for param in self.task_model.classifer.parameters():
        #     param.requires_grad = False

        max_local_steps = self.local_steps

        for step in range(max_local_steps):
            print('epoch',step)
            stats = self.test_metrics()
            stats_train = self.train_metrics()

            test_acc = stats[0] * 1.0 / stats[1]
            test_auc = stats[2] * 1.0 / stats[1]
            test_loss = stats[3] * 1.0 / stats[1]
            train_loss = stats_train[0] * 1.0 / stats_train[1]
            train_acc = stats_train[2] * 1.0 / stats_train[1]
            print('domain:',self.data_name,'train_loss:',train_loss,'  train_acc:',train_acc*100)
            print('domain:', self.data_name, 'test_loss:', test_loss, '  test_acc:', test_acc * 100)
            self.task_model.train()
            # self.task_model.make_per_model()
            for i, (x, y) in enumerate(trainloader):
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)
                self.task_optimizer.zero_grad()
                output = self.task_model(x)
                loss = self.loss(output, y)
                loss.backward()
                # print(self.task_model.classifer.weight)
                # print(self.task_model.classifer.bias)
                # for k, v in self.task_model.mixer1.named_parameters():
                #     print('key:', k, v.grad)
                # for k, v in self.task_model.base_model[1].named_parameters():
                #     print('key:', k, v.grad)
                self.task_optimizer.step()
        print('final test time')
        stats = self.test_metrics()
        test_acc = stats[0] * 1.0 / stats[1]
        test_loss = stats[3] * 1.0 / stats[1]
        print('domain:', self.data_name, 'test_loss:', test_loss, '  test_acc:', test_acc * 100)
        return {self.data_name: test_acc}


    def test_metrics(self):
        if self.dataset == "domainnet" or self.dataset == "minidomainnet":
            test_loader_full = self.test_loader
        else:
            test_loader_full = self.load_test_data()

        self.task_model.eval()

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
                output = self.task_model(x)

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

        self.task_model.eval()
        train_acc = 0
        train_num = 0
        loss = 0
        for x, y in train_loader:
            if type(x) == type([]):
                x[0] = x[0].to(self.device)
            else:
                x = x.to(self.device)
            y = y.to(self.device)
            output = self.task_model(x)
            train_acc += (torch.sum(torch.argmax(output, dim=1) == y)).item()
            train_num += y.shape[0]
            loss += self.loss(output, y).item() * y.shape[0]

        return loss, train_num, train_acc
