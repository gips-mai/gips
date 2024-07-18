import torch.nn as nn
import pandas as pd
import torch
import numpy as np


# first two classes copied from: https://github.com/gastruc/osv5m/blob/main/models/networks/utils.py
class NormGPS(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        """Normalize latitude longtitude radians to -1, 1."""  # not used currently
        return x / torch.Tensor([np.pi * 0.5, np.pi]).unsqueeze(0).to(x.device)


class UnormGPS(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        """Unormalize latitude longtitude radians to -1, 1."""
        x = torch.clamp(x, -1, 1)
        return x * torch.Tensor([np.pi * 0.5, np.pi]).unsqueeze(0).to(x.device)


class GeoLogHead(nn.Module):
    """ Creates the hybrid geolocation head for predicting GPS coordinates from an image.
    Consists of two modules:
        1. MLP Centroid: Predicts the probability for each cell in the quadtree
        and also predicts the GPS coordinates within each cell.
        2. HybridHeadCentroid: Extracts the most probable cell and respective GPS coordinates
        and forms a final prediction output."""

    # Heavily inspired from https://github.com/gastruc/osv5m/blob/main/models/networks/heads/hybrid.py

    def __init__(self, mid_initial_dim, quad_tree_path, is_training):
        super().__init__()

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.final_dim = 11399  # quadtree length
        use_tanh = True
        scale_tanh = 1.2

        # Module for raw prediction computation
        self.geoloc_head_mid_network = MLPCentroid(initial_dim=mid_initial_dim,
                                                   hidden_dim=[mid_initial_dim, 1024, 512],
                                                   final_dim=self.final_dim,
                                                   activation=torch.nn.GELU,
                                                   norm=torch.nn.GroupNorm)
        # Module for extracting the final prediction from the raw values
        self.hybrid_head_centroid = HybridHeadCentroid(final_dim=self.final_dim,
                                                       quadtree_path=quad_tree_path,
                                                       use_tanh=use_tanh,
                                                       scale_tanh=scale_tanh,
                                                       is_training=is_training)
        # Loss functions
        self.cell_loss = nn.CrossEntropyLoss()
        self.coordinate_loss = nn.MSELoss()

    def forward(self, aggr_input, cell_target=None):
        # Predict the probability of each cell in the quadtree and the regression values
        output_midnetwork = self.geoloc_head_mid_network(aggr_input)
        # Extract the most probable cell and return the GPS coordinates within this cell
        return self.hybrid_head_centroid(output_midnetwork, cell_target)

    def get_loss(self, pred, cell_target, coordinate_target):
        """ Computes the loss for the hybrid head.
        Args:
            pred: dict with the predicted output of the hybrid head
            cell_target: tensor with the target cell
            coordinate_target: tensor with the target GPS coordinates
        Returns:
             The loss for the hybrid head, consisting of the cell loss (classification loss)
             and the coordinate loss (regression loss). """

        # Convert cell target to one-hot encoding because classificator predicts probs for all cells
        cell_target_one_hot = torch.zeros((cell_target.shape[0], self.final_dim)).to(self.device)
        for b in range(cell_target.shape[0]):
            cell_target_one_hot[b][cell_target[b]] = 1

        coords_loss = self.coordinate_loss(pred['gps'].float(), coordinate_target)
        cell_loss = self.cell_loss(pred['label'], cell_target_one_hot)

        return coords_loss + cell_loss


class MLPCentroid(nn.Module):
    """ The MLP module for predicting the probability of each cell in the quadtree and the GPS
    coordinates within each cell."""

    def __init__(
            self,
            initial_dim=512,
            hidden_dim=[128, 32, 2],
            final_dim=2,
            norm=nn.InstanceNorm1d,
            activation=nn.ReLU,
    ):
        super().__init__()

        dim = [initial_dim] + hidden_dim + [final_dim]
        args = self.init_layers(dim, norm, activation)
        self.classif = nn.Sequential(*args)
        dim = [initial_dim] + hidden_dim + [2 * final_dim]
        args = self.init_layers(dim, norm, activation)
        self.reg = nn.Sequential(*args)
        # torch.nn.init.normal_(self.reg.weight, mean=0.0, std=0.01)

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

    def forward(self, x):
        # Predict the probability of each cell in the quadtree
        result_classif = self.classif(x)
        # Predict the GPS coordinates within each cell
        result_regres = self.reg(x)
        # Return the raw values of the classification and regression
        return torch.cat([result_classif, result_regres], dim=1)


class HybridHeadCentroid(nn.Module):
    """ Assisting module to the geolocation head. Extracts the raw values for the cell probabilities and the regression
    values and combines them into a final prediction output. """

    def __init__(self, final_dim, quadtree_path, use_tanh, scale_tanh, is_training):
        super().__init__()
        self.final_dim = final_dim
        self.use_tanh = use_tanh
        self.scale_tanh = scale_tanh

        self.unorm = UnormGPS()
        if quadtree_path is not None:
            quadtree = pd.read_csv(quadtree_path)
            self.init_quadtree(quadtree)

        self.training = is_training

    def init_quadtree(self, quadtree):
        """ Initialize the quadtree with the min, max and mean values for latitude and longitude of each cell. """

        quadtree[["min_lat", "max_lat", "mean_lat"]] /= 90.0
        quadtree[["min_lon", "max_lon", "mean_lon"]] /= 180.0

        self.cell_center = torch.tensor(quadtree[["mean_lat", "mean_lon"]].values)
        self.cell_size_up = torch.tensor(quadtree[["max_lat", "max_lon"]].values) - torch.tensor(
            quadtree[["mean_lat", "mean_lon"]].values)
        self.cell_size_down = torch.tensor(quadtree[["mean_lat", "mean_lon"]].values) - torch.tensor(
            quadtree[["min_lat", "min_lon"]].values)

    def forward(self, x, gt_label):
        # Extract the most probable classified cell in the quadtree
        classification_logits = x[..., : self.final_dim]
        classification = classification_logits.argmax(dim=-1)
        # move the max min and center latt and long values of the classified cell to the correct device
        self.cell_size_up = self.cell_size_up.to(classification.device)
        self.cell_center = self.cell_center.to(classification.device)
        self.cell_size_down = self.cell_size_down.to(classification.device)

        # Extract the regression values
        regression = x[..., self.final_dim:]

        if self.use_tanh:
            regression = self.scale_tanh * torch.tanh(regression)

        # Reshape the regression to batch_size X Num_cells X 2 (lat, long)
        regression = regression.view(regression.shape[0], -1, 2)

        if self.training:
            # Get the predicted regression values for the most probable cell
            regression = torch.gather(
                # batch_size X Num_cells X 2
                regression,
                1,
                # batch_size X 0 -> batch_size X 1 X 1 -> repeat -> batch_size X 1 X 2
                gt_label.unsqueeze(-1).expand(regression.shape[0], 1, 2),
            )[:, 0, :]
            # If the predicted regression values for (x,y)/(long,latt) are greater zero
            # select the ground truth regression values from the upper bound of the respective cell
            # Otherwise use the lower bound
            size = torch.where(
                regression > 0,
                self.cell_size_up[gt_label].squeeze(dim=1),
                self.cell_size_down[gt_label].squeeze(dim=1),
            )
            center = self.cell_center[gt_label].squeeze(dim=1)
            gps = center + regression * size
        else:
            regression = torch.gather(
                regression,
                1,
                classification.unsqueeze(-1)
                .unsqueeze(-1)
                .expand(regression.shape[0], 1, 2),
            )[:, 0, :]
            size = torch.where(
                regression > 0,
                self.cell_size_up[classification],
                self.cell_size_down[classification],
            )
            center = self.cell_center[classification]
            gps = self.cell_center[classification] + regression * size

        gps = self.unorm(gps)

        return {
            "label": classification_logits,
            "gps": gps,
            "size": 1.0 / size,
            "center": center,
            "reg": regression,
        }
