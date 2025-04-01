import torch
import torch.nn as nn
import numpy as np

class PlanckCoreAI(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(PlanckCoreAI, self).__init__()
        self.hidden = nn.Linear(input_size, hidden_size)
        self.output = nn.Linear(hidden_size, output_size)
        self.activation = nn.ReLU()
        self.instant_convergence()

    def instant_convergence(self):
        with torch.no_grad():
            self.hidden.weight.copy_(
                torch.tensor(self.planck_weight_initialization(self.hidden.weight.shape), dtype=torch.float32)
            )
            self.output.weight.copy_(
                torch.tensor(self.planck_weight_initialization(self.output.weight.shape), dtype=torch.float32)
            )

    def planck_weight_initialization(self, shape):
        phi = (1 + np.sqrt(5)) / 2
        random_weights = np.random.randn(*shape) * phi
        planck_noise = np.random.normal(0, 1e-35, shape)
        return random_weights + planck_noise

    def forward(self, x):
        x = self.activation(self.hidden(x))
        return self.output(x)

def get_planck_model(input_size=10, hidden_size=20, output_size=1):
    model = PlanckCoreAI(input_size, hidden_size, output_size)
    return model