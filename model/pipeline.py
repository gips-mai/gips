import backbone as bb
import attention_module as am
import geolocation_head as gh
from torch import nn


class GeoguessrPipeline(nn.Module):
    """ Assembly of the pipeline components. """

    def __init__(self):
        super().__init__()
        # visual parts
        self.img_encoder = bb.StreetCLIP()
        self.img_descriptor = bb.ImageDescriptor()
        # textual parts
        self.location_attention = am.LocationAttention()
        self.lat_long_head = gh.LatLongHead()

    def forward(self, x):

        # decompose x into its components for the individual modules
        img = x['img']
        clues = x['clues']

        x = self.backbone(x)
        x = self.location_attention(x)
        x = self.lat_long_head(x)
        return x