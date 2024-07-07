import torch.nn as nn
import pandas as pd
from utils.osv5m_utils import UnormGPS
import torch


class GeoLogHead(nn.Module):

    def __init__(self, mid_initial_dim, quad_tree_path, is_training):
        super().__init__()  # Call the parent class initializer first

        final_dim = 11399  # quadtree len
        use_tanh = True
        scale_tanh = 1.2

        # Prediction computation
        self.geoloc_head_mid_network = MLPCentroid(initial_dim=mid_initial_dim,
                                                   hidden_dim=[mid_initial_dim, 1024, 512],
                                                   final_dim=final_dim,
                                                   activation=torch.nn.GELU,
                                                   norm=torch.nn.GroupNorm)
        # Loss computation
        self.hybrid_head_centroid = HybridHeadCentroid(final_dim=final_dim,
                                                       quadtree_path=quad_tree_path,
                                                       use_tanh=use_tanh,
                                                       scale_tanh=scale_tanh,
                                                       is_training=is_training)

        self.cell_loss = nn.CrossEntropyLoss()
        self.coordinate_loss = nn.MSELoss()

    def forward(self, aggr_input, cell_target=None):
        return self.hybrid_head_centroid(self.geoloc_head_mid_network(aggr_input), cell_target)

    def get_loss(self, pred, cell_target, coordinate_target):
        return (self.coordinate_loss(pred['gps'].float(), coordinate_target) +
                self.cell_loss(pred['label'], cell_target))

class MLPCentroid(nn.Module):
    def __init__(
            self,
            initial_dim=512,
            hidden_dim=[128, 32, 2],
            final_dim=2,
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
        """Predicts GPS coordinates from an image.
        Args:
            x: torch.Tensor with features
        """
        result_classif = self.classif(x)
        result_regres = self.reg(x)
        return torch.cat([result_classif, result_regres], dim=1)


class HybridHeadCentroid(nn.Module):
    """Classification heads followed by regression heads for the network."""

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

        quadtree[["min_lat", "max_lat", "mean_lat"]] /= 90.0
        quadtree[["min_lon", "max_lon", "mean_lon"]] /= 180.0

        self.cell_center = torch.tensor(quadtree[["mean_lat", "mean_lon"]].values)
        self.cell_size_up = torch.tensor(quadtree[["max_lat", "max_lon"]].values) - torch.tensor(
            quadtree[["mean_lat", "mean_lon"]].values)
        self.cell_size_down = torch.tensor(quadtree[["mean_lat", "mean_lon"]].values) - torch.tensor(
            quadtree[["min_lat", "min_lon"]].values)

    def forward(self, x, gt_label):
        """Forward pass of the network.
        x : Union[torch.Tensor, dict] with the output of the backbone.
        """
        # Extract the most probable classified cell in the quadtree
        classification_logits = x[..., : self.final_dim]
        classification = classification_logits.argmax(dim=-1)
        # move the max min and center latt and long values of the classified cell to the correct device
        self.cell_size_up = self.cell_size_up.to(classification.device)
        self.cell_center = self.cell_center.to(classification.device)
        self.cell_size_down = self.cell_size_down.to(classification.device)

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
            size = torch.where(
                regression > 0,
                self.cell_size_up[gt_label],
                self.cell_size_down[gt_label],
            )
            center = self.cell_center[gt_label]
            gps = self.cell_center[gt_label] + regression * size
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
