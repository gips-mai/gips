import torch.nn as nn
import torch
class RegressionHead(nn.Module):
    def __init__(
            self,
            initial_dim=512,
            hidden_dim=[128, 32, 2],
            norm=nn.InstanceNorm1d,
            activation=nn.ReLU,
    ):
        """
        Initializes an MLP Classification Head
        Args:
            hidden_dim (list): list of hidden dimensions for the MLP
            norm (nn.Module): normalization layer
            activation (nn.Module): activation layer
        """
        super().__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        final_dim = 2
        dim = [initial_dim] + hidden_dim + [final_dim]
        args = self.init_layers(dim, norm, activation)
        self.reg = nn.Sequential(*args)
        self.coordinate_loss = nn.MSELoss()


    def init_layers(self, dim, norm, activation):
        """Initializes the MLP layers."""
        args = [nn.LayerNorm(dim[0])]
        for i in range(len(dim) - 1):
            args.append(nn.Linear(dim[i], dim[i + 1]))
            if i < len(dim) - 2:
                # args.append(norm(dim[i + 1]))
                args.append(norm(4, dim[i + 1]))
                args.append(activation())
        return args

    # Target cell not used
    def forward(self, x, target_cell=None):
        """Predicts GPS coordinates from an image.
        Args:
            x: torch.Tensor with features
        """
        return self.reg(x)

    # Target cell not used
    def get_loss(self, pred, target_cell, coordinate_target):
        return self.coordinate_loss(pred, coordinate_target)



