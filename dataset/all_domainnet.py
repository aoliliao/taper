import os
from PIL import Image
import warnings

from torch.utils.data import Dataset, ConcatDataset
from torch.utils.data import DataLoader
import numpy as np
import bisect
import json
from torchvision import transforms
import torch


class DomainNetImageList(Dataset):
    def __init__(self, image_root, image_list_root, domain, domain_label, split='train', transform=None,):
        self.image_root = image_root
        self.image_list_root = image_list_root
        self.domain = domain  # name of the domain

        self.transform = transform

        self.loader = self._rgb_loader

        imgs = self._make_dataset(os.path.join(self.image_list_root, domain + '_' + split + '.txt'), domain_label)

        self.imgs = imgs
        self.tgts = [s[1] for s in imgs]

    def _rgb_loader(self, path):
        with open(path, 'rb') as f:
            with Image.open(f) as img:
                return img.convert('RGB')

    def _make_dataset(self, image_list_path, domain):
        image_list = open(image_list_path).readlines()
        images = [(val.split()[0], int(val.split()[1]), int(domain)) for val in image_list]
        return images

    def __getitem__(self, index):
        output = {}
        path, target, domain = self.imgs[index]

        raw_img = self.loader(os.path.join(self.image_list_root, path))

        if self.transform is not None:
            img = self.transform(raw_img)
        output['img'] = img
        output['target'] = torch.squeeze(torch.LongTensor([np.int64(target).item()]))
        output['domain'] = domain
        output['idx'] = index

        label = torch.squeeze(torch.LongTensor([np.int64(target).item()]))
        return img, label

    def __len__(self):
        return len(self.imgs)


def prepare_data_all_domainnet(args, datasets=[]):
    data_base_path = './dataset/domainNet'
    image_list_root = './dataset/domainNet/DomainNet'
    transform_train = transforms.Compose([
        transforms.Resize([224, 224]),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation((-30, 30)),
        transforms.ToTensor(),
    ])

    transform_test = transforms.Compose([
        transforms.Resize([224, 224]),
        transforms.ToTensor(),
    ])
    all_domains = {
        'domainnet':
            {
                'path': '/',
                'list_root': '/',
                'sub_domains': ['clipart', 'infograph', 'painting', 'quickdraw', 'real', 'sketch'],
                'numbers': [14604, 15582, 21850, 51750, 52041, 20916],
                'classes': 345,
                'neurons': 128
            },
    }

    train_sets = []
    test_sets = []
    train_loaders = []
    test_loaders = []
    min_data_len = 1e8
    num_workers = 4
    if len(datasets) <= 6:
        for i in range(len(datasets)):
            trainset = DomainNetImageList(image_root=data_base_path, image_list_root=image_list_root,
                      domain=datasets[i], transform=transform_train, domain_label=i, split='train',)

            testset = DomainNetImageList(image_root=data_base_path, image_list_root=image_list_root,
                      domain=datasets[i], transform=transform_test, domain_label=i, split='test',)
            train_sets.append(trainset)
            test_sets.append(testset)
        concat_train_dataset = ConcatDataset(train_sets)
        concat_test_dataset = ConcatDataset(test_sets)
        print(len(concat_train_dataset), len(concat_test_dataset))
        concat_train_dataloader = torch.utils.data.DataLoader(concat_train_dataset, batch_size=args.batch_size,
                                                              shuffle=True, num_workers=num_workers)
        concat_test_dataloader = torch.utils.data.DataLoader(concat_test_dataset, batch_size=args.batch_size,
                                                             shuffle=False, num_workers=num_workers)
        print(len(concat_train_dataloader), len(concat_test_dataloader))
        # min_data_len = int(min_data_len * 0.05)
        # print(min_data_len)
        for i in range(len(train_sets)):
            # train_sets[i] = torch.utils.data.Subset(train_sets[i], list(range(min_data_len)))
            # print(len(train_sets[i]))
            train_loaders.append(torch.utils.data.DataLoader(train_sets[i], batch_size=args.batch_size, shuffle=True, num_workers=num_workers))
            test_loaders.append(torch.utils.data.DataLoader(test_sets[i], batch_size=args.batch_size, shuffle=False, num_workers=num_workers))

    print(len(train_loaders[0]), len(test_loaders[0]))
    return train_loaders, test_loaders, concat_train_dataloader, concat_test_dataloader,

def prepare_data_all_domainnet_person(args, datasets=[], node_num=1):
    data_base_path = './dataset/domainNet'
    image_list_root = './dataset/domainNet/DomainNet'
    transform_train = transforms.Compose([
        transforms.Resize([224, 224]),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation((-30, 30)),
        transforms.ToTensor(),
    ])

    transform_test = transforms.Compose([
        transforms.Resize([224, 224]),
        transforms.ToTensor(),
    ])
    train_sets = []
    test_sets = []
    train_loaders = []
    test_loaders = []
    task_label = []

    with open("./dataset/domainNet/DomainNet/index2label_dict.txt", 'r') as file:
         labels_dict = json.load(file)

    num_workers = 4
    for i in range(len(datasets)):
        trainset = DomainNetImageList(image_root=data_base_path, image_list_root=image_list_root,
                                      domain=datasets[i], transform=transform_train, domain_label=i, split='train', )

        testset = DomainNetImageList(image_root=data_base_path, image_list_root=image_list_root,
                                     domain=datasets[i], transform=transform_test, domain_label=i, split='test', )
        train_sets.append(trainset)
        test_sets.append(testset)
    concat_train_dataset = ConcatDataset(train_sets)
    concat_test_dataset = ConcatDataset(test_sets)
    print(len(concat_train_dataset), len(concat_test_dataset))
    concat_train_dataloader = torch.utils.data.DataLoader(concat_train_dataset, batch_size=args.batch_size,
                                                          shuffle=True, num_workers=num_workers)
    concat_test_dataloader = torch.utils.data.DataLoader(concat_test_dataset, batch_size=args.batch_size,
                                                         shuffle=False, num_workers=num_workers)
    print(len(concat_train_dataloader), len(concat_test_dataloader))
    step = len(labels_dict) // node_num
    for i in range(len(train_sets)):
        st = 0
        for j in range(node_num):
            labels = list(labels_dict.keys())[st:st + step]
            new_dict = {key: labels_dict[key] for key in labels}
            selected_indices = [idx for idx, label in enumerate(train_sets[i].tgts) if
                                label in list(labels_dict.values())[st:st + step]]
            train_data = torch.utils.data.Subset(train_sets[i], selected_indices)
            selected_indices = [idx for idx, label in enumerate(test_sets[i].tgts) if
                                label in list(labels_dict.values())[st:st + step]]
            test_data = torch.utils.data.Subset(test_sets[i], selected_indices)
            task_label.append(new_dict)
            # print(labels[st:st+step])
            train_loaders.append(
                torch.utils.data.DataLoader(train_data, batch_size=args.batch_size, shuffle=True, num_workers=num_workers))
            test_loaders.append(torch.utils.data.DataLoader(test_data, batch_size=args.batch_size, shuffle=False, num_workers=num_workers ))
            st += step
    node_datasets = [x for x in datasets for _ in range(node_num)]

    print(len(train_loaders[0]), len(test_loaders[0]))
    return train_loaders, test_loaders, task_label, concat_train_dataloader, concat_test_dataloader, node_datasets

def index2label():
    # image_list_path = "./domainNet/DomainNet/index2label.txt"
    # image_list = open(image_list_path).readlines()
    # # print(image_list)
    # label = [i.replace('\n', '') for i in image_list]
    # print(label)
    # image_list_path = './domainNet/DomainNet/clipart_test.txt'
    # image_list = open(image_list_path).readlines()
    # images = [val.split("/")[1]  for val in image_list]
    # from collections import OrderedDict
    # res = list(OrderedDict.fromkeys(images))
    # # print(res)
    # result_dict = {}
    # for index, item in enumerate(res):
    #     result_dict[item] = index
    # print(result_dict)
    import json
    # with open("./domainNet/DomainNet/index2label_dict.txt", 'w') as file:
    #     json.dump(result_dict, file)
    with open("./domainNet/DomainNet/index2label_dict.txt", 'r') as file:
         res = json.load(file)
    keys = list(res.keys())[:10]
    new_dict = {key: res[key] for key in keys}
    print(new_dict)
    # f = open("./domainNet/DomainNet/index2label.txt", "w")
    # for line in res:
    #     f.write(line+'\n')
    # f.close()

# if __name__ == '__main__':
#     index2label()

