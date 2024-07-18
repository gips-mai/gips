import torch.nn as nn
import torch


class RegressionHead(nn.Module):
    """ Creates a regression head for predicting GPS coordinates from an image."""

    def __init__(
            self,
            initial_dim=512,
            hidden_dim=[128, 32, 2],
            norm=nn.InstanceNorm1d,
            activation=nn.ReLU,
    ):
        super().__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Compute the dimensions of the layers
        final_dim = 2
        dim = [initial_dim] + hidden_dim + [final_dim]
        # Initialize the layers with the specified dimensions, normalization and activation functions
        args = self.init_layers(dim, norm, activation)
        self.reg = nn.Sequential(*args)
        self.coordinate_loss = nn.MSELoss()

    def init_layers(self, dim, norm, activation):
        """Initializes the MLP layers."""
        args = [nn.LayerNorm(dim[0])]
        for i in range(len(dim) - 1):
            args.append(nn.Linear(dim[i], dim[i + 1]))
            if i < len(dim) - 2:
                args.append(norm(4, dim[i + 1]))
                args.append(activation())
        return args

    def forward(self, x):
        """Predicts GPS coordinates from an image.
        Args:
            x: torch.Tensor with features """
        return self.reg(x)

    def get_loss(self, pred, coordinate_target):
        """ Computes the loss for the regression head. """
        return self.coordinate_loss(pred, coordinate_target)
