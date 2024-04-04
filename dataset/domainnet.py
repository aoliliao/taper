import os
import random
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, ConcatDataset
import torchvision.transforms as transforms



class DomainNetDataset(Dataset):
    def __init__(self, base_path, site, train=True, transform=None):
        if train:
            self.paths, self.text_labels = np.load('./dataset/domainNet/DomainNet/{}_train.pkl'.format(site), allow_pickle=True)
        else:
            self.paths, self.text_labels = np.load('./dataset/domainNet/DomainNet/{}_test.pkl'.format(site), allow_pickle=True)

        label_dict = {'bird': 0, 'feather': 1, 'headphones': 2, 'ice_cream': 3, 'teapot': 4, 'tiger': 5, 'whale': 6,
                      'windmill': 7, 'wine_glass': 8, 'zebra': 9}

        self.labels = [label_dict[text] for text in self.text_labels]
        self.transform = transform
        self.base_path = base_path if base_path is not None else '../data'

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.base_path, self.paths[idx])
        label = self.labels[idx]
        image = Image.open(img_path)

        if len(image.split()) != 3:
            image = transforms.Grayscale(num_output_channels=3)(image)

        if self.transform is not None:
            image = self.transform(image)

        return image, label


def prepare_data_domainnet_customer(args, datasets=[]):
    data_base_path = './dataset/domainNet'
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

    min_data_len = 1e8
    if len(datasets) <= 6:
        for i in range(len(datasets)):
            trainset = DomainNetDataset(data_base_path, site=datasets[i], transform=transform_train)
            testset = DomainNetDataset(data_base_path, site=datasets[i], transform=transform_test, train=False)
            train_sets.append(trainset)
            test_sets.append(testset)
            # print(len(trainset), len(testset))
            # if len(trainset) < min_data_len:
            #     min_data_len = len(trainset)
        concat_train_dataset = ConcatDataset(train_sets)
        concat_test_dataset = ConcatDataset(test_sets)
        print(len(concat_test_dataset), len(concat_train_dataset))
        concat_train_dataloader = torch.utils.data.DataLoader(concat_train_dataset, batch_size=args.batch_size, shuffle=True)
        concat_test_dataloader = torch.utils.data.DataLoader(concat_test_dataset, batch_size=args.batch_size, shuffle=False)
        print(len(concat_train_dataloader), len(concat_test_dataloader))
        # min_data_len = int(min_data_len * 0.05)
        # print(min_data_len)
        for i in range(len(train_sets)):
            # train_sets[i] = torch.utils.data.Subset(train_sets[i], list(range(min_data_len)))
            # print(len(train_sets[i]))
            train_loaders.append(torch.utils.data.DataLoader(train_sets[i], batch_size=args.batch_size, shuffle=True, ))
            test_loaders.append(torch.utils.data.DataLoader(test_sets[i], batch_size=args.batch_size, shuffle=False, ))

    # print(len(train_loaders), len(test_loaders))
    print(len(train_loaders[0]), len(test_loaders[0]))
    return train_loaders, test_loaders, concat_train_dataloader, concat_test_dataloader,

def prepare_data_domainnet_per(args, datasets=[], per_node=1):
    data_base_path = './dataset/domainNet'
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
    labels = ['bird', 'feather', 'headphones', 'ice_cream', 'teapot', 'tiger', 'whale', 'windmill', 'wine_glass',
              'zebra']
    labels_dict = {'bird': 0, 'feather': 1, 'headphones': 2, 'ice_cream': 3, 'teapot': 4, 'tiger': 5, 'whale': 6,
                  'windmill': 7, 'wine_glass': 8, 'zebra': 9}

    labels_index = list(range(0, 10))
    train_sets = []
    test_sets = []
    train_loaders = []
    test_loaders = []
    task_label = []
    class_client = 10
    min_data_len = 1e8

    num_workers = 4
    for i in range(len(datasets)):
        trainset = DomainNetDataset(data_base_path, site=datasets[i], transform=transform_train)
        testset = DomainNetDataset(data_base_path, site=datasets[i], transform=transform_test, train=False)

        train_sets.append(trainset)
        test_sets.append(testset)

    concat_train_dataset = ConcatDataset(train_sets)
    concat_test_dataset = ConcatDataset(test_sets)
    concat_train_dataloader = torch.utils.data.DataLoader(concat_train_dataset, batch_size=args.batch_size,
                                                          shuffle=True, num_workers=num_workers)
    concat_test_dataloader = torch.utils.data.DataLoader(concat_test_dataset, batch_size=args.batch_size,
                                                         shuffle=False, num_workers=num_workers)

    step = len(labels_dict) // per_node
    for i in range(len(train_sets)):
        # print(train_sets[i].labels)
        st = 0
        for j in range(per_node):
            labels = list(labels_dict.keys())[st:st + step]
            new_dict = {key: labels_dict[key] for key in labels}
            selected_indices = [idx for idx, label in enumerate(train_sets[i].labels) if
                                label in list(labels_dict.values())[st:st + step]]
            # selected_indices = [idx for idx, label in enumerate(train_sets[i].labels) if label in labels_index[st:st+step]]
            train_data = torch.utils.data.Subset(train_sets[i], selected_indices)
            selected_indices = [idx for idx, label in enumerate(test_sets[i].labels) if
                                label in list(labels_dict.values())[st:st + step]]
            # selected_indices = [idx for idx, label in enumerate(test_sets[i].labels) if label in labels_index[st:st+step]]
            test_data = torch.utils.data.Subset(test_sets[i], selected_indices)
            # task_label.append(labels[st:st+step])
            task_label.append(new_dict)
            train_loaders.append(
                torch.utils.data.DataLoader(train_data, batch_size=args.batch_size, shuffle=True,num_workers=num_workers))
            test_loaders.append(torch.utils.data.DataLoader(test_data, batch_size=args.batch_size, shuffle=False, num_workers=num_workers ))
            st += step
    node_datasets = [x for x in datasets for _ in range(per_node)]
    print(len(train_loaders), len(test_loaders))
    # print(len(train_loaders[0]), len(test_loaders[0]))
    return train_loaders, test_loaders, task_label, concat_train_dataloader, concat_test_dataloader, node_datasets


