import torch
from representation.mlp import MLP
from data import GetData
import tqdm
import torch.nn as nn
import os
import numpy as np
import matplotlib.pyplot as plt
from representation.representation import MlpRepresentation
from torch.utils.data import DataLoader, TensorDataset
from torchattacks import *
from analysis_tools import *


class Experiments:
    def __init__(self, seed, device, hidden_layers, act_function, criterion, learning_rate,
                 momentum, epochs, optimizer, wd, data_name, train_batch_size, test_batch_size):
        self.seed = seed
        self.device = device
        self.hidden_layers = hidden_layers
        self.act_function = act_function
        self.criterion = criterion
        self.lr = learning_rate
        self.momentum = momentum
        self.epochs = epochs
        self.optimizer = optimizer
        self.wd = wd
        self.data_name = data_name
        self.train_batch_size = train_batch_size
        self.test_batch_size = test_batch_size

        if self.data_name.lower() == "mnist":
            self.data = GetData.mnist_dataset(self.train_batch_size, self.test_batch_size)
        elif self.data_name.lower() == "cifar10":
            self.data = GetData.cifar10(self.train_batch_size, self.test_batch_size)
        else:
            ValueError("This data set is not supported, the dataset {} is not recognized".format(self.data_name))

    def training_image(self):

        path = os.getcwd()
        directory = "Saved_model"
        new_path = os.path.join(path, directory)
        if not os.path.exists(directory):
            os.makedirs(new_path)

        train_loader = self.data["train_loader"]
        test_loader = self.data["test_loader"]

        input_shape = (1, 1, self.data["num_features"])
        num_classes = self.data["num_classes"]
        model = MLP(input_shape=input_shape, hidden_sizes=list(self.hidden_layers), num_classes=num_classes,
                    activation=self.act_function, ).to(device=self.device)

        if self.optimizer.lower() == 'adam':
            optimizer = torch.optim.Adam(model.parameters(), self.lr, weight_decay=self.wd)
        else:
            optimizer = torch.optim.SGD(model.parameters(), self.lr, self.momentum, weight_decay=self.wd)

        test_loss_list = []
        test_accuracy_list = []
        training_loss_list = []
        training_accuracy_list = []
        print("training in progress")
        for epoch in tqdm.trange(self.epochs):
            for images, labels in train_loader:
                images = images.view(images.shape[0], -1)
                optimizer.zero_grad()

                output = model(images)
                loss = self.criterion(output, labels)
                loss.backward()
                optimizer.step()

                model_file_name = f'Model{epoch}'
                model_path = os.path.join(new_path, model_file_name)
                torch.save(model, model_path)

                with torch.no_grad():
                    y_prediction_train = model(images)
                    correct_train = (torch.argmax(y_prediction_train, dim=1) == labels).type(torch.FloatTensor)

            training_loss_list.append(loss.item())
            training_accuracy_list.append(correct_train.mean())

            for images_test, labels_test in test_loader:
                images_test.view(images_test.shape[0], -1)
                with torch.no_grad():
                    y_prediction_test = model(images_test)
                    test_loss = self.criterion(y_prediction_test, labels_test)
                    correct_test = (torch.argmax(y_prediction_test, dim=1) == labels_test).type(torch.FloatTensor)

            test_loss_list.append(test_loss.item())
            test_accuracy_list.append(correct_test.mean())
        return training_accuracy_list, training_loss_list, test_accuracy_list, test_loss_list

    def norm_comp(self):
        path = os.getcwd()
        directory = "Saved_model"
        new_path = os.path.join(path, directory)
        model_file = f'Model{self.epochs-1}'
        model_path = os.path.join(new_path, model_file)
        model = torch.load(model_path)
        train_loader = self.data["train_loader"]

        atk = FGSM(model, eps=8 / 255)

        adv_matrices = []
        adv_output_layer = []
        clean_matrices = []
        clean_output_layer = []

        print("creating adversarial examples")
        for images, labels in train_loader:
            adv_image = atk(images, labels)

        for i, x_adv in enumerate(adv_image):
            adv_weights, adv_matrix = representation_matrix(x_adv, model, self.device)
            adv_output_per_layer = layers_output(x_adv, adv_weights)
            adv_matrices.append(adv_matrix)
            adv_output_layer.append(adv_output_per_layer)
        for j, x_clean in enumerate(self.data["x_train"]):
            clean_weights, clean_matrix = representation_matrix(x_clean, model, self.device)
            clean_output_per_layer = layers_output(x_clean, clean_weights)
            clean_matrices.append(clean_matrix)
            clean_output_layer.append(clean_output_per_layer)

        clean_norm_per_layer = []
        noisy_norm_per_layer = []

        std_clean = []
        std_noisy = []

        for i in range(len(self.hidden_layers)+1):
            value = []
            for j in range(len(clean_output_layer)):
                x_data = clean_output_layer[j]
                for k in range(len(x_data)):
                    value.append(norm(x_data[i]))
            clean_norm_per_layer.append(np.mean(value))
            std_clean.append(np.std(value))

        for i in range(len(self.hidden_layers)+1):
            value = []
            for j in range(len(adv_output_layer)):
                x_data = adv_output_layer[j]
                for k in range(len(x_data)):
                    value.append(norm(x_data[i]))
            noisy_norm_per_layer.append(np.mean(value))
            std_noisy.append(np.std(value))

        return clean_norm_per_layer, noisy_norm_per_layer, std_clean, std_noisy

    def plot_norm_per_layer(self, clean_norm, noisy_norm, std_clean, std_noisy, training_loss,
                                        training_accuracy, test_loss, test_accuracy):
        x_epoch = [i for i in range(self.epochs)]
        x = np.arange(1, len(self.hidden_layers) + 2, 1)
        fig, axs = plt.subplots(2)
        axs[0].plot(x, clean_norm, label="norm per layer on normal data", color="blue")
        axs[0].errorbar(x, clean_norm, yerr=std_clean)

        axs[0].plot(x, noisy_norm, label="norm per layer on adversarial example", color="orange")
        axs[0].errorbar(x, noisy_norm, yerr=std_noisy)
        axs[0].legend()

        axs[1].plot(x_epoch, training_loss, label="training loss", linestyle="-", color="r")
        axs[1].plot(x_epoch, training_accuracy, label="training accuracy", linestyle="-.", color="r")

        axs[1].plot(x_epoch, test_loss, label="test loss", linestyle="-", color="g")
        axs[1].plot(x_epoch, test_accuracy, label="test accuracy", linestyle="-.", color="g")
        plt.show()








# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    seed = 78
    device = "cuda" if torch.cuda.is_available() else "cpu"
    criterion = nn.CrossEntropyLoss()
    train_batch_size = 500
    test_batch_size = 150
    act_function = 'relu'
    data_name = 'mnist'
    epochs = 100
    hidden_layers = (100, 100, 100)
    optimizer = "sgd"
    wd = 0.01
    lr = 0.01
    momentum = 0.9
    exp = Experiments(seed, device, hidden_layers, act_function, criterion, lr, momentum, epochs, optimizer,
                          wd, data_name, train_batch_size, test_batch_size)
    training_accuracy, training_loss, test_accuracy, test_loss = exp.training_image()
    clean_norm, noisy_norm, std_clean, std_noisy = exp.norm_comp()
    exp.plot_norm_per_layer(clean_norm, noisy_norm, std_clean, std_noisy, training_loss,
                                        training_accuracy, test_loss, test_accuracy)
