import torch
from representation.mlp import MLP


class MlpRepresentation:
    def __init__(
        self, model: MLP, device="cpu",
    ):
        super(MlpRepresentation, self).__init__()
        self.model = model
        self.num_classes = self.model.num_classes
        self.input_shape = self.model.input_shape
        self.device = device
        self.act_fn = self.model.get_activation_fn()()
        self.mlp_weights = self.model.get_weights()
        self.input_size = self.model.input_size

    def forward(self, x):
        flat_x = torch.flatten(x)
        weights = [torch.ones(w.shape, device=self.device) for w in self.mlp_weights]
        # First layer
        weights[0] = self.mlp_weights[0] * flat_x

        post_act = flat_x

        for i in range(1, len(weights)):
            pre_act = self.mlp_weights[i - 1].matmul(post_act)
            post_act = self.act_fn(pre_act)
            # TODO check if pre_act has zeroes
            vertices = post_act / pre_act
            weights[i] = self.mlp_weights[i] * vertices

        matrix = self.compute_matrix(weights)
        output = torch.matmul(matrix, torch.ones(self.input_size, device=self.device))
        return output, weights, matrix

    def compute_matrix(self, weights):
        A = torch.matmul(weights[1], weights[0])
        for i in range(2, len(weights)):
            A = torch.matmul(weights[i], A)
        return A









