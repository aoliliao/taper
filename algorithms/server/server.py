import torch
import os
import numpy as np

import copy
import time
import random

from sklearn import metrics
from sklearn.preprocessing import label_binarize
from torch.utils.data import DataLoader


# from dataset.domainnet import prepare_data_domainnet_customer, prepare_data_domain1

from util.data_util import read_client_data


class Server(object):
    def __init__(self, args, times):
        # Set up the main attributes
        self.device = args.device
        self.dataset = args.dataset
        self.num_class = args.num_classes
        self.global_rounds = args.global_rounds
        self.local_steps = args.local_steps
        self.batch_size = args.batch_size
        self.learning_rate = args.local_learning_rate
        self.global_model = copy.deepcopy(args.model)
        self.num_clients = args.num_clients
        self.join_ratio = args.join_ratio
        self.random_join_ratio = args.random_join_ratio
        self.join_clients = int(self.num_clients * self.join_ratio)
        self.algorithm = args.algorithm
        self.time_select = args.time_select
        self.goal = args.goal
        self.time_threthold = args.time_threthold
        self.top_cnt = 100
        # self.save_folder_name = './save/all_domainnet'
        # self.save_folder_name = './save/resnet18-base'
        self.save_folder_name = './save/cifar'
        self.clients = []
        self.selected_clients = []
        # self.train_slow_clients = []
        # self.send_slow_clients = []

        self.uploaded_weights = []
        self.uploaded_ids = []
        self.uploaded_models = []

        self.rs_test_acc = []
        self.rs_test_auc = []
        self.rs_train_loss = []

        self.times = times
        self.eval_gap = args.eval_gap
        self.client_drop_rate = args.client_drop_rate
        # self.train_slow_rate = args.train_slow_rate
        # self.send_slow_rate = args.send_slow_rate
        self.comment = args.comment

        self.train_loaders = None
        self.test_loaders = None
        self.concat_train_dataloader = None
        self.concat_test_dataloader = None


    def set_clients(self, args, clientObj):
        for i in range(self.num_clients):
            train_data = read_client_data(self.dataset, i, is_train=True)
            test_data = read_client_data(self.dataset, i, is_train=False)
            client = clientObj(args,
                               id=i,
                               train_samples=len(train_data),
                               test_samples=len(test_data),
                              )
            self.clients.append(client)

    # def set_clients_bn(self, args, clientObj):
    #     if args.dataset == "office":
    #         self.train_loaders, self.test_loaders, self.concat_test_dataloader, datasets = prepare_data_office1(args)
    #         # self.train_loaders, self.test_loaders, self.concat_test_dataloader, = prepare_data_office(args)
    #         # name of each dataset
    #         # datasets = ['Amazon', 'Caltech', 'DSLR', 'Webcam']
    #     elif args.dataset == "digits":
    #         self.train_loaders, self.test_loaders = prepare_data_digits(args)
    #         datasets = ['MNIST', 'SVHN', 'USPS', 'SynthDigits', 'MNIST-M']
    #     elif args.dataset == "domainnet":
    #         # train_loaders, test_loaders = prepare_data_domainnet(args)
    #         # datasets = ['clipart', 'infograph', 'painting', 'quickdraw', 'real', 'sketch']
    #         # self.train_loaders, self.test_loaders, self.concat_test_dataloader = prepare_data_domainnet_customer(args, datasets)
    #         self.train_loaders, self.test_loaders, self.concat_test_dataloader, datasets = prepare_data_domain1(args)
    #         # print(datasets)
    #     # federated setting
    #     client_num = args.num_clients
    #     client_weights = [1 / client_num for i in range(client_num)]
    #     for i in range(client_num):
    #         train_data_loader = self.train_loaders[i]
    #         test_data_loader = self.test_loaders[i]
    #         client = clientObj(args,
    #                            id=i,
    #                            train_samples=len(train_data_loader.dataset),
    #                            test_samples=len(test_data_loader.dataset),
    #                            train_loader=train_data_loader,
    #                            test_loader=test_data_loader,
    #                            data_name=datasets[i]
    #                           )
    #         self.clients.append(client)

    # random select slow clients
    # def select_slow_clients(self, slow_rate):
    #     slow_clients = [False for i in range(self.num_clients)]
    #     idx = [i for i in range(self.num_clients)]
    #     idx_ = np.random.choice(idx, int(slow_rate * self.num_clients))
    #     for i in idx_:
    #         slow_clients[i] = True
    #
    #     return slow_clients

    # def set_slow_clients(self):
    #     self.train_slow_clients = self.select_slow_clients(
    #         self.train_slow_rate)
    #     self.send_slow_clients = self.select_slow_clients(
    #         self.send_slow_rate)

    def select_clients(self):
        if self.random_join_ratio:
            join_clients = np.random.choice(range(self.join_clients, self.num_clients + 1), 1, replace=False)[0]
        else:
            join_clients = self.join_clients
        selected_clients = list(np.random.choice(self.clients, join_clients, replace=False))
        # print(selected_clients)
        # selected_clients = self.clients
        return selected_clients

    def send_models(self):
        assert (len(self.clients) > 0)

        for client in self.clients:
            start_time = time.time()

            client.set_parameters(self.global_model)

            client.send_time_cost['num_rounds'] += 1
            client.send_time_cost['total_cost'] += 2 * (time.time() - start_time)

    def receive_models(self):
        assert (len(self.selected_clients) > 0)

        active_clients = random.sample(
            self.selected_clients, int((1 - self.client_drop_rate) * self.join_clients))
        # active_clients = self.selected_clients
        # print(active_clients)
        self.uploaded_weights = []
        self.uploaded_models = []
        tot_samples = 0
        for client in active_clients:
            # client_time_cost = client.train_time_cost['total_cost'] / client.train_time_cost['num_rounds'] + \
            #                    client.send_time_cost['total_cost'] / client.send_time_cost['num_rounds']

            tot_samples += client.train_samples
            self.uploaded_weights.append(client.train_samples)
            self.uploaded_models.append(client.model)
        for i, w in enumerate(self.uploaded_weights):
            self.uploaded_weights[i] = w / tot_samples


    def aggregate_parameters(self):
        for key in self.global_model.state_dict().keys():
            # print('key:', key)
            tmp = torch.zeros_like(self.global_model.state_dict()[key]).float()
            for client_idx in range(len(self.uploaded_weights)):
                tmp += self.uploaded_weights[client_idx] * self.uploaded_models[client_idx].state_dict()[key]
            self.global_model.state_dict()[key].data.copy_(tmp)
        # assert (len(self.uploaded_models) > 0)
        #
        # self.global_model = copy.deepcopy(self.uploaded_models[0])
        #
        # for param in self.global_model.parameters():
        #     param.data.zero_()
        # for w, client_model in zip(self.uploaded_weights, self.uploaded_models):
        #     self.add_parameters(w, client_model)

    def add_parameters(self, w, client_model):
        for server_param, client_param in zip(self.global_model.parameters(), client_model.parameters()):
            server_param.data += client_param.data.clone() * w

    def save_global_model(self):
        model_path = os.path.join(self.save_folder_name, self.dataset, self.algorithm)
        # model_path = './save'
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        model_path = os.path.join(model_path, self.algorithm + "_server" + ".pth")
        torch.save(self.global_model, model_path)

    def load_model(self):
        model_path = os.path.join("models", self.dataset)
        model_path = os.path.join(model_path, self.algorithm + "_server" + ".pt")
        assert (os.path.exists(model_path))
        self.global_model = torch.load(model_path)

    def model_exists(self):
        model_path = os.path.join("models", self.dataset)
        model_path = os.path.join(model_path, self.algorithm + "_server" + ".pt")
        return os.path.exists(model_path)

    def save_item(self, item, item_name):
        model_path = os.path.join(self.save_folder_name, self.dataset, self.algorithm)
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        torch.save(item, os.path.join(model_path, "server_" + item_name + ".pth"))

    def load_item(self, item_name):
        return torch.load(os.path.join(self.save_folder_name, "server_" + item_name + ".pt"))

    def test_metrics(self):
        num_samples = []
        tot_correct = []
        tot_auc = []
        tot_loss = []
        for c in self.clients:
            ct, ns, auc, loss = c.test_metrics()
            tot_correct.append(ct * 1.0)
            tot_auc.append(auc * ns)
            num_samples.append(ns)
            tot_loss.append(loss * 1.0)

        ids = [c.id for c in self.clients]

        return ids, num_samples, tot_correct, tot_auc, tot_loss
    def load_global_test_data(self, batch_size=None, dataset=None):
        if batch_size == None:
            batch_size = self.batch_size
        if self.dataset == "cifar":
            batch_size = 500
        # print(batch_size)
        if dataset == 'cifar':
            dataset = 'cifar100FL/cifar20/nc5R0/'
        elif dataset == "10label":
            dataset = 'cifar10/nc10_0_0.1/'
        test_data_dir = os.path.join('./dataset', dataset, 'test/')
        test_file = test_data_dir + 'global' + '.npz'
        with open(test_file, 'rb') as f:
            data = np.load(f, allow_pickle=True)['data'].tolist()

        X_test = torch.Tensor(data['x']).type(torch.float32)
        y_test = torch.Tensor(data['y']).type(torch.int64)
        test_data = [(x, y) for x, y in zip(X_test, y_test)]
        return DataLoader(test_data, batch_size, drop_last=False, shuffle=False)

    def test_generic_metric(self, num_classes, device, model, test_data=None):
        if test_data == None:
            test_loader_global = self.load_global_test_data(dataset=self.dataset)
        else:
            test_loader_global = test_data
        model.eval()

        test_acc = 0
        test_num = 0
        loss = 0
        y_prob = []
        y_true = []
        with torch.no_grad():
            for x, y in test_loader_global:
                if type(x) == type([]):
                    x[0] = x[0].to(device)
                else:
                    x = x.to(device)
                y = y.to(device)
                output = model(x)

                test_acc += (torch.sum(torch.argmax(output, dim=1) == y)).item()
                test_num += y.shape[0]

                y_prob.append(output.detach().cpu().numpy())
                nc = num_classes
                if num_classes == 2:
                    nc += 1
                lb = label_binarize(y.detach().cpu().numpy(), classes=np.arange(nc))
                if num_classes == 2:
                    lb = lb[:, :2]
                y_true.append(lb)

        y_prob = np.concatenate(y_prob, axis=0)
        y_true = np.concatenate(y_true, axis=0)

        auc = metrics.roc_auc_score(y_true, y_prob, average='micro')

        return test_acc, test_num, auc

    def train_metrics(self):
        num_samples = []
        losses = []
        accs = []
        for c in self.clients:
            cl, ns, acc = c.train_metrics()
            num_samples.append(ns)
            losses.append(cl * 1.0)
            accs.append(acc * 1.0)

        ids = [c.id for c in self.clients]

        return ids, num_samples, losses, accs

    # evaluate selected clients
    def evaluate(self, acc=None, loss=None):
        stats = self.test_metrics()
        stats_train = self.train_metrics()

        test_acc = sum(stats[2]) * 1.0 / sum(stats[1])
        test_auc = sum(stats[3]) * 1.0 / sum(stats[1])
        test_loss = sum(stats[4]) * 1.0 / sum(stats[1])
        train_loss = sum(stats_train[2]) * 1.0 / sum(stats_train[1])
        train_acc = sum(stats_train[3]) * 1.0 / sum(stats_train[1])
        accs = [a / n for a, n in zip(stats[2], stats[1])]
        train_accs = [a / n for a, n in zip(stats_train[3], stats_train[1])]
        aucs = [a / n for a, n in zip(stats[3], stats[1])]

        mean_testacc = np.mean(accs)
        mean_trainacc = np.mean(train_accs)

        if acc == None:
            self.rs_test_acc.append(test_acc)
        else:
            acc.append(test_acc)

        if loss == None:
            self.rs_train_loss.append(train_loss)
        else:
            loss.append(train_loss)

        print("Averaged Train Loss: {:.4f}".format(train_loss))
        print("Averaged Train Accurancy: {:.4f}".format(train_acc))
        print("Averaged Test Loss: {:.4f}".format(test_loss))
        print("Averaged Test Accurancy: {:.4f}".format(test_acc))
        print("Averaged Test AUC: {:.4f}".format(test_auc))
        print("mean Test Accurancy: {:.4f}".format(np.mean(accs)))
        print("Std Test Accurancy: {:.4f}".format(np.std(accs)))
        print("Std Test AUC: {:.4f}".format(np.std(aucs)))

        return train_loss, test_acc, test_loss, train_acc, mean_testacc, mean_trainacc

    def report_process(self, avg_acc, avg_train_loss, glo_acc, avg_test_loss, avg_train_acc, mean_testaccs, mean_trainaccs, comment=''):
        from torch.utils.tensorboard import SummaryWriter
        from datetime import datetime
        import os

        TIMES = "{0:%Y-%m-%d--%H-%M-%S/}".format(datetime.now())
        ori_logs = str(self.dataset) + "-" + str(self.algorithm) + TIMES + self.comment
        writer = SummaryWriter(log_dir=os.path.join('./log', ori_logs))
        for epoch in range(len(avg_acc)):
            writer.add_scalar("acc/testp_avg_acc", avg_acc[epoch], epoch)
            writer.add_scalar("loss/train_avg_loss", avg_train_loss[epoch], epoch)
            writer.add_scalar("loss/test_avg_loss", avg_test_loss[epoch], epoch)
            writer.add_scalar("acc/testg_acc", glo_acc[epoch], epoch)
            writer.add_scalar("acc/train_acc", avg_train_acc[epoch], epoch)
            writer.add_scalar("acc/testp_mean__acc", mean_testaccs[epoch], epoch)
            # writer.add_scalar("acc/train_mean_acc", mean_trainaccs[epoch], epoch)
        writer.close()

