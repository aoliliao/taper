import os
import time
import random

import torch
import torch.nn as nn
import numpy as np
from threading import Thread

from algorithms.client.clientTAPER import clientTAPER
from algorithms.client.clientTAPER_per import clientTAPER_per
from algorithms.client.clientbaseline import clientTAPER_baseline
from algorithms.client.clientcifar import clientcifar
from algorithms.server.server import Server
from dataset.all_domainnet import prepare_data_all_domainnet, prepare_data_all_domainnet_person
from dataset.cifar100 import random_groups_cifar100, superclass_groups_cifar100, superclass_diff_groups_cifar100
from dataset.domainnet import prepare_data_domainnet_customer, prepare_data_domainnet_per
import wandb


class servercifar(Server):
    def __init__(self, args, times):
        super().__init__(args, times)
        self.per_node = args.per_node
        self.weight_decay = args.weight_decay
        self.momentum = args.momentum
        self.loss = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.SGD(self.global_model.parameters(), lr=self.learning_rate,
                                         weight_decay=self.weight_decay, momentum=self.momentum)
        self.learning_rate_scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer=self.optimizer,
            step_size=30,
            gamma=0.1
        )

        self.Budget = []
        self.args = args
        # if args.dataset == 'cifar100':
        #     self.train_sets, self.test_sets, task_label, self.concat_train_dataloader, self.concat_test_dataloader,\
        #         = random_groups_cifar100(args)

    def save_global_model(self):
        model_path = os.path.join(self.save_folder_name,)
        # model_path = './save'
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        model_path = os.path.join(model_path,  "conv_stageone" + ".pth")
        torch.save({'model': self.global_model.state_dict(),}, model_path)

    def load_global_model(self):
        model_path = os.path.join(self.save_folder_name,)
        model_path = os.path.join(model_path, "conv_stageone" + ".pth")
        assert (os.path.exists(model_path))
        state_dict = torch.load(model_path, map_location='cuda')
        self.global_model.load_state_dict(state_dict['model'])

    def set_clients_per(self, args, clientObj,):
        if args.dataset == 'cifar100':
            # self.train_sets, self.test_sets, task_label, self.concat_train_dataloader, self.concat_test_dataloader, \
            #     = random_groups_cifar100(args)
            # self.train_sets, self.test_sets, task_label, self.concat_train_dataloader, self.concat_test_dataloader, \
            #     = superclass_groups_cifar100(args)
            self.train_sets, self.test_sets, task_label, self.concat_train_dataloader, self.concat_test_dataloader, \
                = superclass_diff_groups_cifar100(args)
        client_num = len(self.train_sets)
        for i in range(client_num):
            train_data_loader = self.train_sets[i]
            test_data_loader = self.test_sets[i]
            client = clientObj(args,
                               id=i,
                               train_samples=len(train_data_loader.dataset),
                               test_samples=len(test_data_loader.dataset),
                               train_loader=train_data_loader,
                               test_loader=test_data_loader,
                               task_label=task_label[i],
                              )
            self.clients.append(client)

    def train_stage_two(self):
        self.set_clients_per(self.args, clientObj=clientcifar,)

        for client in self.clients:
            client.train_stage_two()

    def train_stage_one(self):
        # self.model.to(self.device)
        # wandb.init(project='TAPER',
        #            name='baseline_345',
        #            config={
        #                "learning_rate": 0.1,
        #                "architecture": "resnet18",
        #                "dataset": "domainnet_345",
        #                "epochs": 125,
        #                "seed": 42,
        #            }
        #            )

        global_steps = self.global_rounds

        for epoch in range(global_steps):
            print(epoch)
            # print(self.optimizer.state_dict()['param_groups'][0]['lr'])
            # wandb.log({f"lr": self.optimizer.state_dict()['param_groups'][0]['lr']})

            stats_train = self.train_metrics()
            train_loss = stats_train[0] * 1.0 / stats_train[1]
            train_acc = stats_train[2] * 1.0 / stats_train[1]
                # wandb.log({f"train_loss": train_loss})
                # wandb.log({f"train_acc": train_acc})
            print('loss:',train_loss,'  acc:',train_acc)
            # if epoch % 10 == 0 or epoch == global_steps - 1:
            stats_test = self.test_metrics()
            test_loss = stats_test[0] * 1.0 / stats_test[1]
            test_acc = stats_test[2] * 1.0 / stats_test[1]
                # wandb.log({f"test_loss": test_loss})
                # wandb.log({f"test_acc": test_acc})
            print('test_loss:', test_loss, '  test_acc:', test_acc)
            if epoch % 20 == 0 or epoch == global_steps - 1:
                self.save_global_model()
            self.global_model.train()
            for i, (x, y) in enumerate(self.concat_train_dataloader):
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)
                self.optimizer.zero_grad()
                output = self.global_model(x)
                loss = self.loss(output, y)
                loss.backward()
                self.optimizer.step()

            # self.learning_rate_scheduler.step()
        # wandb.finish()

    def train_metrics(self):
        self.global_model.eval()
        train_acc = 0
        train_num = 0
        loss = 0
        # print(len(self.concat_train_dataloader))
        with torch.no_grad():
            for x, y in self.concat_train_dataloader:
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)
                # print(y)
                output = self.global_model(x)
                train_acc += (torch.sum(torch.argmax(output, dim=1) == y)).item()
                train_num += y.shape[0]
                loss += self.loss(output, y).item() * y.shape[0]
        # print(train_acc,'--',train_num)
        return loss, train_num, train_acc

    def test_metrics(self):
        self.global_model.eval()
        train_acc = 0
        train_num = 0
        loss = 0
        with torch.no_grad():
            for x, y in self.concat_test_dataloader:
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)

                output = self.global_model(x)
                train_acc += (torch.sum(torch.argmax(output, dim=1) == y)).item()
                train_num += y.shape[0]
                loss += self.loss(output, y).item() * y.shape[0]

        return loss, train_num, train_acc



