from torch.nn.modules.module import T
import torch.nn
import modules.heads.country_pred_head as cph
from modules.heads.geoloc_head import GeoLogHead
from modules import attention_module as am, backbone as bb
from torch import nn
from datasets import load_dataset
from huggingface_hub import PyTorchModelHubMixin
from modules.heads.easy_reg_head import RegressionHead
from pathlib import Path



class Gips(nn.Module, PyTorchModelHubMixin):
    """ Assembly of the gips pipeline components. """

    def __init__(self, img_embedding_size, descript_embedding_size,
                 clue_embedding_size, use_multimodal_inputs, is_training, use_reg_head=False):
        super().__init__()

        quad_tree_path, clues = self._prepare_data()
        self.use_multimodal_inputs = use_multimodal_inputs
        self.use_reg_head = use_reg_head

        self.training = is_training
        self.init_modules(img_embedding_size, descript_embedding_size, clue_embedding_size, clues, quad_tree_path)



    def init_modules(self, img_embedding_size, descript_embedding_size, clue_embedding_size, clues, quad_tree_path):
        """ Initialize the modules of the Gips model."""
        #self.img_encoder = bb.StreetCLIP()  # Image encoder
        self.lin_att = am.LinearAttention(attn_input_img_size=img_embedding_size,
                                          text_features_size=len(clues),
                                          hidden_layer_size_0=1024,
                                          hidden_layer_size_1=1024)
        self.att_weight_aggr = am.AttentionWeightedAggregation(temperature=1, clues=clues)
        if self.use_multimodal_inputs:
            self.guiding_head = cph.GuidingHead(aggr_clue_emb_size=clue_embedding_size, clues=clues)
            mid_initial_dim = img_embedding_size + clue_embedding_size + descript_embedding_size
        else:
            mid_initial_dim = img_embedding_size

        if not self.use_reg_head:
            self.lat_long_head = GeoLogHead(mid_initial_dim=mid_initial_dim,
                                            quad_tree_path=quad_tree_path,
                                            is_training=self.training)
        else:
            hidden_dim = [128, 32, 2]
            if self.use_multimodal_inputs:
                hidden_dim = [1024] + hidden_dim

            self.lat_long_head = RegressionHead(initial_dim=mid_initial_dim, hidden_dim=hidden_dim)

    def forward(self, enc_img, enc_descr, target_cell=None):
        """ Forward pass of the Gips model. Predicts the latitude and longitude of the image.
         Args:
             enc_img (Tensor): Image tensor.
              enc_descr (Tensor): Description tensor .
        Returns:
            GipsOutput: Model prediction. """

        attn_scores = None
        aggr_clues = None
        # If multimodal inputs are used compute the aggregated clues representation and use it together with the encoded
        # image and the encoded description to predict the latitude and longitude
        if self.use_multimodal_inputs:
            aggr_clues, attn_scores = self._compute_clue_attention(enc_img)
            x = torch.cat((enc_img, enc_descr, aggr_clues), dim=1)
        else:
            x = enc_img

        # calculate predic based on set head
        latt_long_pred = self.lat_long_head(x, target_cell)

        return GipsOutput(latt_long_pred, attn_scores, aggr_clues)

    def get_individual_losses(self: T, enc_img, enc_descr, target_cell, target_country, coordinate_target)  -> T:
        """Returns model prediction, total loss, lat_long_loss and guiding_loss"""

        prediction = self.forward(enc_img, enc_descr, target_cell)
        total_loss = 0.0

        if self.use_multimodal_inputs:
            lat_long_pred, aggr_clues, attn_scores = (prediction.lat_long_pred,
                                                      prediction.aggr_clues,
                                                      prediction.attn_scores)

            country_pred = self.guiding_head(aggr_clues)
            guiding_loss = self.guiding_head.get_comb_loss(country_pred, target_country, attn_scores)
            total_loss += guiding_loss
        else:
            guiding_loss = None
            lat_long_pred = prediction.lat_long_pred

        lat_long_loss = self.lat_long_head.get_loss(lat_long_pred, target_cell, coordinate_target)
        total_loss += lat_long_loss
        return prediction, total_loss, lat_long_loss, guiding_loss

    def _prepare_data(self):
        quad_tree_path = str(Path(__file__).parent.parent / "data" / "quad_tree" / "quadtree_10_1000.csv")
        clues = load_dataset("gips-mai/all_clues_enc", split='train')

        return quad_tree_path, clues

    def _compute_clue_attention(self, enc_img):
        """ Compute the attention weights for the clues.
        And return an aggregated version of the clues based on the attention weights.
        Args:
            enc_img (Tensor): Encoded image tensor """

        attn_scores = self.lin_att(enc_img)

        return self.att_weight_aggr.forward(attn_scores), attn_scores


class GipsOutput():
    """ The output prediction of the Gips model.
    Consists of the predicted latitude and longitude, the aggregated clues representation and its attention scores."""

    def __init__(self, lat_long_pred, attn_scores, aggr_clues):
        self.lat_long_pred = lat_long_pred
        self.aggr_clues = aggr_clues
        self.attn_scores = attn_scores

        if type(lat_long_pred) is dict:
            self.label = lat_long_pred['label']
            self.gps = lat_long_pred['gps']
            self.size = lat_long_pred['size']
            self.center = lat_long_pred['center']
            self.reg = lat_long_pred['reg']
        else:
            self.gps = lat_long_pred
            self.label = None
            self.size = None
            self.center = None
            self.reg = lat_long_pred

        self.pred_geoloc_head = {
            "label": self.label,
            "gps": self.gps,
            "size": self.size,
            "center": self.center,
            "reg": self.reg,
            "attn_scores": self.attn_scores
        }

    def items(self):
        return self.pred_geoloc_head.items()

class GipsBase(nn.Module):
    def __init__(self, img_embedding_size, descript_embedding_size,
                 clue_embedding_size, use_multimodal_inputs, is_training, use_reg_head=False):
        super().__init__()

        quad_tree_path, clues = self._prepare_data()
        self.use_multimodal_inputs = use_multimodal_inputs

        self.training = is_training
        self.use_reg_head = use_reg_head
        self.init_modules(img_embedding_size, descript_embedding_size, clue_embedding_size, clues, quad_tree_path)

    def init_modules(self, img_embedding_size, descript_embedding_size, clue_embedding_size, clues, quad_tree_path):
        """ Initialize the modules of the Gips model."""
        self.img_encoder = bb.StreetCLIP()  # Image encoder
        self.lin_att = am.LinearAttention(attn_input_img_size=img_embedding_size,
                                          text_features_size=len(clues),
                                          hidden_layer_size_0=1024,
                                          hidden_layer_size_1=1024)
        self.att_weight_aggr = am.AttentionWeightedAggregation(temperature=1, clues=clues)
        self.guiding_head = cph.GuidingHead(aggr_clue_emb_size=clue_embedding_size, clues=clues)

        if self.use_multimodal_inputs:
            mid_initial_dim = img_embedding_size + clue_embedding_size + descript_embedding_size
        else:
            mid_initial_dim = img_embedding_size

        if not self.use_reg_head:
            self.lat_long_head = GeoLogHead(mid_initial_dim=mid_initial_dim,
                                            quad_tree_path=quad_tree_path,
                                            is_training=self.training)
        else:

            hidden_dim = [128, 32, 2]
            if self.use_multimodal_inputs:
                hidden_dim = [1024] + hidden_dim

            self.lat_long_head = RegressionHead(initial_dim=mid_initial_dim, hidden_dim=hidden_dim)