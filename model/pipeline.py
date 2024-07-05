import backbone as bb
import attention_module as am
import geolocation_head as gh
from torch import nn


class GeoguessrPipeline(nn.Module):
    """ Assembly of the pipeline components. """

    def __init__(self):
        super().__init__()
        self.img_encoder = bb.StreetCLIP()  # Image encoder
        self.location_attention = am.LinearAttention()
        self.lat_long_head = gh.LatLongHead()

    def forward(self, img, enc_clues):
        """ Forward pass of the pipeline.
         Args:
             img (Tensor): Image tensor of shape (B, C, H, W)
             clues (Tensor): Class tensor of shape (B, C, H, W) """

        enc_img = self.img_encoder(img)

        att_weights = self.location_attention(enc_img, enc_clues)

        # TODO: complete!!!