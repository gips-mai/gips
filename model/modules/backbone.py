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