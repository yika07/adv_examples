import torch
from torchvision.datasets import MNIST, CIFAR10
from torch.utils.data import DataLoader
import torchvision.transforms as T


class GetData:
    def __init__(self):
        pass

    @staticmethod
    def mnist_dataset(train_batch_size, test_batch_size):
        train_dataset = MNIST("", train=True, download=True,
                              transform=T.Compose([T.ToTensor(),
                                                   T.Lambda(lambda x:  torch.flatten(x))]))

        train_loader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True)

        test_dataset = MNIST("", train=False, download=True,
                             transform=T.Compose([T.ToTensor(),
                                                  T.Lambda(lambda x:  torch.flatten(x))]))

        test_loader = DataLoader(test_dataset, batch_size=test_batch_size, shuffle=True)

        x_train = next(iter(train_loader))[0]
        y_train = next(iter(train_loader))[1]
        x_test = next(iter(test_loader))[0]
        y_test = next(iter(test_loader))[1]
        num_classes, num_features = 10, 784
        dict_data = {"x_train": x_train, "y_train": y_train, "x_test": x_test, "y_test": y_test,
                     "num_classes": num_classes,
                     "num_features": num_features,
                     "train_loader": train_loader,
                     "test_loader": test_loader}

        return dict_data

    @staticmethod
    def cifar10(train_batch_size, test_batch_size):
        train_dataset = CIFAR10(root='./data', train=True, download=True,
                                transform=T.Compose([T.ToTensor(),
                                                     T.Lambda(lambda x: torch.flatten(x))]))

        train_loader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True)

        test_dataset = CIFAR10(root='./data', train=False, download=True,
                               transform=T.Compose([T.ToTensor(),
                                                    T.Lambda(lambda x: torch.flatten(x))]))

        test_loader = DataLoader(test_dataset, batch_size=test_batch_size, shuffle=True)

        x_train = next(iter(train_loader))[0]
        y_train = next(iter(train_loader))[1]
        x_test = next(iter(test_loader))[0]
        y_test = next(iter(test_loader))[1]
        num_classes, num_features = 10, 3072
        dict_data = {"x_train": x_train, "y_train": y_train, "x_test": x_test, "y_test": y_test,
                     "num_classes": num_classes,
                     "num_features": num_features,
                     "train_loader": train_loader,
                     "test_loader": test_loader}

        return dict_data

        # TODO: add cifar100



