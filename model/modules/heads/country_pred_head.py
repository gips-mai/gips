import torch
import torch.nn as nn


class GuidingHead(nn.Module):
    def __init__(self, aggr_clue_emb_size, clues, alpha=0.75,
                 hot_encoding_size=221) -> None:
        super().__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Parameter settings
        self.alpha = alpha  # The influence of the pseudo label loss w.r.t the country classification loss
        self.aggr_clue_emb_size = aggr_clue_emb_size  # Size of the aggr clues embedding
        self.hot_encoding_size = hot_encoding_size
        # Initializations
        self.init_layers()
        self.init_loss_functions(clues, hot_encoding_size)

    def init_layers(self):
        """ Initializes the MLP layers of the classifier head used to predict the country only from the aggregated
        static clues. Used as a guiding head for the main model to learn which clues are important
        for a given image. """
        self.classifier = nn.Sequential(
            nn.Linear(self.aggr_clue_emb_size, 512),
            #nn.Linear(1024, 512),
            nn.Linear(512, self.hot_encoding_size),
        )

    def init_loss_functions(self, clues, hot_encoding_size):
        """ Initializes the loss functions used by the guiding head. """
        # Create a matrix of the country encodings
        self.clue_mat = torch.zeros((len(clues['country_one_hot_enc']), hot_encoding_size))
        for i, country_encs in enumerate(clues['country_one_hot_enc']):
            for country_enc in country_encs:
                if len(country_enc) > 0:
                    self.clue_mat[i] += torch.tensor(country_enc)
        # The matrix is a one-hot encoding of the countries, therfore countries that were added multiple times
        # must be clipped to 1
        self.clue_mat = torch.clip(self.clue_mat, 0, 1).to(self.device)

        self.bce_loss = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(1))  # can be changed
        self.ce_loss = torch.nn.CrossEntropyLoss()

    def comp_pseudo_label_loss(self, attention_prediction, gt_country):
        """ Compute the pseudo label loss.
        The pseudo label loss is the difference between the attention scores and the ground truth country encoding.
        This loss is used to guide the main model to learn which clues are important for a given image.
        Args:
            attention_prediction: The attention scores prediction vector (length = number of clues)
            gt_country: The ground truth country one hot encoding vector
        Returns:
            The binary cross entropy loss between the attention scores and the ground truth country encoding """

        # gt = num clue X n_enc * n_enc X 1
        # gt = num clue X 1
        # loss = gt - attention_prediction

        batch_size = gt_country.size(0)

        # Convert gt_country into a matrix of column vectors of size (batch_size, num_clues, 1) for the scalar product
        gt_country = gt_country.reshape(batch_size, -1, 1)
        # Adjust the clue matrix to the batch size to allow for the scalar product on each sample
        # Batch_size x Clue_num x embedding size
        # Scalar product returns a vector which only has 1s where the country is present in the sample
        gt_attention = self.clue_mat.unsqueeze(dim=0).expand((batch_size, -1, -1)) @ gt_country
        gt_attention = gt_attention.squeeze(dim=2)
        return self.bce_loss(attention_prediction, gt_attention)

    def comp_country_loss(self, country_prediction, gt_country):
        """ Computes the country classification loss.
        The country classification loss is the cross entropy loss between the predicted country and the ground truth.
        This loss is used to additionally guide the main model to learn which clues are important for a given image.
        Args:
            country_prediction: The predicted country one hot encoding vector
            gt_country: The ground truth country one hot encoding vector
        Returns:
            The cross entropy loss between the predicted country and the ground truth """
        return self.ce_loss(country_prediction, gt_country)

    def forward(self, x):
        return nn.functional.softmax(self.classifier(x), dim=1)

    def get_comb_loss(self, country_pred, country_target, attention_scores):
        """ Computes the combined loss of the Guiding Head.
        Args:
            country_pred: The predicted country one hot encoding
            country_target: The ground truth country one hot encoding
            attention_scores: The attention scores prediction vector (length = number of clues)
        Returns:
            The combined loss of the Guiding Head """
        # Country loss + Pseudo label loss

        country_loss = self.comp_country_loss(country_pred, country_target)
        pseudo_loss = self.comp_pseudo_label_loss(attention_scores, country_target)

        # print("Country_loss: " + str(country_loss))
        # print("Pseudo loss: " + str(pseudo_loss))

        return country_loss * (1 - self.alpha) + pseudo_loss * self.alpha

    def training_step(self, aggr_clues, target_country, attn_scores):
        """ Performs a forward pass and computes the loss of the Guiding Head."""
        country_pred = self.forward(aggr_clues)
        loss = self.get_comb_loss(country_pred, target_country, attn_scores)
        return loss
