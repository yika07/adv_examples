import torch
import torch.nn as nn


class MLP(nn.Module):
    def __init__(
        self,
        input_shape,
        num_classes,
        hidden_sizes=[10, 10],
        activation="relu",
        bias=False,
    ):
        super(MLP, self).__init__()
        self.num_classes = num_classes
        self.layers = []
        self.activation = activation
        self.input_shape = input_shape
        ch, w, h = self.input_shape
        self.input_size = ch * w * h
        self.layers_size = hidden_sizes
        self.layers_size.insert(0, ch * w * h)
        self.layers_size.append(num_classes)
        self.num_layers = len(self.layers_size) + 2

        for idx in range(len(self.layers_size) - 1):
            self.layers.append(
                nn.Linear(self.layers_size[idx], self.layers_size[idx + 1], bias=bias)
            )

            if idx == len(self.layers_size) - 2:
                break

            self.layers.append(self.get_activation_fn()())
        self.layers.append(nn.Softmax(dim=1))

        self.net = torch.nn.Sequential(*self.layers)
        self.init()

    def forward(self, x):
        return self.net(x)

    def get_weights(self):
        w = []
        for m in self.modules():
            if isinstance(m, nn.Linear):
                w.append(m.weight.data)
        return w

    def get_activation_fn(self):
        act_name = self.activation.lower()
        activation_fn_map = {
            "relu": nn.ReLU,
            "elu": nn.ELU,
            "tanh": nn.Tanh,
            "leakyrelu": nn.LeakyReLU,
            "prelu": nn.PReLU,
            "sigmoid": nn.Sigmoid,
        }
        if act_name not in activation_fn_map.keys():
            raise ValueError("Unknown activation function name : ")
        return activation_fn_map[act_name]

    def init(self):
        def init_func(m):
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight.data)

        self.apply(init_func)
