import copy
import random

import torch
import torch.nn as nn
import numpy as np
import os
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.preprocessing import label_binarize
from sklearn import metrics

from util.data_util import read_client_data



class Client(object):
    """
    Base class for clients in federated learning.
    """

    def __init__(self, args, id, train_samples, test_samples, **kwargs):
        self.model = copy.deepcopy(args.model)
        self.dataset = args.dataset
        self.device = args.device
        self.id = id
        self.save_folder_name = args.save_folder_name
        self.algorithm = args.algorithm
        self.num_classes = args.num_classes
        self.train_samples = train_samples
        self.test_samples = test_samples
        self.batch_size = args.batch_size
        self.learning_rate = args.local_learning_rate
        self.local_steps = args.local_steps
        self.momentum = args.momentum
        self.weight_decay = args.weight_decay

        self.train_time_cost = {'num_rounds': 0, 'total_cost': 0.0}
        self.send_time_cost = {'num_rounds': 0, 'total_cost': 0.0}

        self.test_acc = []
        self.test_loss = []

        self.data_name = None
        self.train_loader = None
        self.test_loader = None
        self.save_folder_name = './save/all_domainnet'
        # self.save_folder_name = './save/resnet18-base'


    def load_train_data(self, batch_size=None):
        if batch_size == None:
            batch_size = self.batch_size
        train_data = read_client_data(self.dataset, self.id, is_train=True)
        return DataLoader(train_data, batch_size, drop_last=False, shuffle=True)

    def load_test_data(self, batch_size=None):
        if batch_size == None:
            batch_size = self.batch_size
        test_data = read_client_data(self.dataset, self.id, is_train=False)
        return DataLoader(test_data, batch_size, drop_last=False, shuffle=False)

    def set_parameters(self, model):
        for key in model.state_dict().keys():
            self.model.state_dict()[key].data.copy_(model.state_dict()[key])
        # for new_param, old_param in zip(model.parameters(), self.model.parameters()):
        #     old_param.data = new_param.data.clone()

    def clone_model(self, model, target):
        for param, target_param in zip(model.parameters(), target.parameters()):
            target_param.data = param.data.clone()

    def update_parameters(self, model, new_params):
        for param, new_param in zip(model.parameters(), new_params):
            param.data = new_param.data.clone()

    def test_metrics(self):
        if self.dataset == "digits" or self.dataset == "office" or self.dataset == "domainnet":
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
        if self.dataset == "digits" or self.dataset == "office" or self.dataset == "domainnet":
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


    def save_item(self, item, item_name, item_path=None):
        model_path = os.path.join(self.save_folder_name, self.dataset, self.algorithm)
        if item_path == None:
            item_path = model_path
        if not os.path.exists(item_path):
            os.makedirs(item_path)
        torch.save(item, os.path.join(item_path, "client_" + str(self.id) + "_" + item_name + ".pth"))

    def load_item(self, item_name, item_path=None):
        if item_path == None:
            item_path = self.save_folder_name
        return torch.load(os.path.join(item_path, "client_" + str(self.id) + "_" + item_name + ".pt"))

    def report_process_client(self, domain, algorithm, test_acc, test_loss, comment=''):
        from torch.utils.tensorboard import SummaryWriter
        from datetime import datetime
        import os

        TIMES = "{0:%Y-%m-%d--%H-%M-%S/}".format(datetime.now())
        ori_logs = str(self.dataset) + "-" + str(domain) + str(self.id) + TIMES + str(algorithm) + str(comment)
        writer = SummaryWriter(log_dir=os.path.join('./logclients', ori_logs))
        for epoch in range(len(test_acc)):
            writer.add_scalar("acc/avg_p_acc", test_acc[epoch], epoch)
            writer.add_scalar("loss/avg_test_loss", test_loss[epoch], epoch)
        writer.close()

    # @staticmethod
    # def model_exists():
    #     return os.path.exists(os.path.join("models", "server" + ".pt"))
