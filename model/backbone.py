from torch import nn
from PIL import Image
from transformers import CLIPProcessor, CLIPModel, AutoModel, AutoTokenizer


class StreetCLIP(nn.Module):
    def __init__(self, path='geolocal/StreetCLIP'):
        """Initializes the CLIP model."""
        super().__init__()
        self.clip = CLIPModel.from_pretrained(path)
        self.transform = CLIPProcessor.from_pretrained(path)

    def forward(self, x):
        """Predicts CLIP features from an image.
        Args:
            x (dict that contains "img": torch.Tensor): Input batch
        """
        features = self.clip.get_image_features(
            **self.transform(images=x["img"], return_tensors="pt").to(x["gps"].device)
        ).unsqueeze(1)
        return features

    # TODO: discuss - why not this forward function?
    # taken from https://github.com/gastruc/osv5m/blob/4e6075387ecde4255410785ffb83830c9aa099f6/models/networks/backbones.py
    def forward_new(self, x):
        """Predicts CLIP features from an image.
        Args:
            x (dict that contains "img": torch.Tensor): Input batch
        """
        features = self.clip(pixel_values=x["img"])["last_hidden_state"]
        return features


class ImageDescriptor(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, x):
        pass


class TextEncoder(nn.Module):
    def __init__(self, path):
        """ Initializes the model for encoding the text clues. """
        super().__init__()
        self.textEncoder = AutoModel.from_pretrained(path)
        self.tokenizer = AutoTokenizer.from_pretrained(path)

    def forward(self, x):
        """Encodes the text clues into a dense representation.
        Args:
            x (): Input batch
        """

    pass
    # TODO: jointly tokenize all text clues (guidebook + ocr + scraped information) -> then jointly encode all information


class LocationAttention(nn.Module):
    def __init__(
            self,
            attn_input_img_size: int,
            text_features_size: int,
            beta=-1.0,
            norm_type: str = "batch_norm",
    ):
        """
            A simple linear layer that only takes an image embedding as input.

            Args:
                attn_input_img_size (_type_): input dim
                text_features_size (_type_): embed dim
                normalization (str): how to normalize the attention scores as probabilities, options: softmax or sigmoid
                norm_type (str): normalize inputs. values: batch_norm, layer_norm, or None
            """

        class SingleModule(nn.Module):
            def __init__(self, dim_in, dim_out):
                super.__init__()

                self.ln = torch.nn.linear(dim_in, dim_out)
                self.actv = torch.nn.functional.relu()

    # normalize text inputs to [0-1] range
    # TODO: research difference between BatchNorm and LayerNorm

    self.lc = nn.Sequential(
        SingleModule(),
        SingleModule()
    )

    def forward(self, x):
        weights = self.lc(x)
        return weights


class LatLongHead(nn.Module):
    def __init__(
            self,
            initial_dim=512,
            hidden_dim=[128, 32, 2],
            final_dim=2,
            norm=nn.InstanceNorm1d,
            activation=nn.ReLU,
            aux_data=[],
    ):
        """
        Initializes an MLP Classification Head
        Args:
            hidden_dim (list): list of hidden dimensions for the MLP
            norm (nn.Module): normalization layer
            activation (nn.Module): activation layer
        """
        super().__init__()
        self.aux_data = aux_data
        self.aux = len(self.aux_data) > 0
        if self.aux:
            hidden_dim_aux = hidden_dim
            hidden_dim_aux[-1] = 128
            final_dim_aux_dict = {
                "land_cover": 12,
                "climate": 30,
                "soil": 14,
                "road_index": 1,
                "drive_side": 1,
                "dist_sea": 1,
            }
            self.idx = {}
            final_dim_aux = 0
            for col in self.aux_data:
                self.idx[col] = [
                    final_dim_aux + i for i in range(final_dim_aux_dict[col])
                ]
                final_dim_aux += final_dim_aux_dict[col]
            dim = [initial_dim] + hidden_dim_aux + [final_dim_aux]
            args = self.init_layers(dim, norm, activation)
            self.mlp_aux = nn.Sequential(*args)
        dim = [initial_dim] + hidden_dim + [final_dim]
        args = self.init_layers(dim, norm, activation)
        self.mlp = nn.Sequential(*args)

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
        if self.aux:
            out = {"gps": self.mlp(x[:, 0, :])}
            x = self.mlp_aux(x[:, 0, :])
            for col in list(self.idx.keys()):
                out[col] = x[:, self.idx[col]]
            return out
        return self.mlp(x[:, 0, :])
