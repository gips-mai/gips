import torch

# Inspiration from https://github.com/gastruc/osv5m/blob/main/metrics/utils.py

class Metric:
    def __init__(self):
        self.haversine_sum = 0
        self.geogame_sum = 0
        self.count = 0

    def update(self, pred:torch.Tensor, gt:torch.Tensor):
        self.haversine_sum += Metric.haversine(pred, gt)
        self.geogame_sum += Metric.geogame_score(pred, gt)
        self.count += pred.shape[0] # batch size

    def compute(self):
        # compute the Metric for all predictions
        output = {
            "Haversine": self.haversine_sum / self.count,
            "Geoguessr": self.geogame_sum/ self.count,
        }

        return output

    def haversine(pred:torch.Tensor, gt:torch.Tensor):
        # from https://github.com/gastruc/osv5m/blob/main/metrics/utils.py
        # expects inputs to be np arrays in (lat, lon) format as radians
        # N x 2

        # calculate the difference in latitude and longitude between the predicted and ground truth points
        lat_diff = pred[:, 0] - gt[:, 0]
        lon_diff = pred[:, 1] - gt[:, 1]

        # calculate the haversine formula components
        lhs = torch.sin(lat_diff / 2) ** 2
        rhs = torch.cos(pred[:, 0]) * torch.cos(gt[:, 0]) * torch.sin(lon_diff / 2) ** 2
        a = lhs + rhs

        # calculate the final distance using the haversine formula
        c = 2 * torch.arctan2(torch.sqrt(a), torch.sqrt(1 - a))
        distance = 6371 * c

        return distance

    def geogame_score(pred:torch.Tensor, gt:torch.Tensor, factor:int=2000):
        # expects inputs to be np arrays in (lat, lon) format
        # N x 2

        # the factor is typically 2000 or 1492.7

        return 5000 * torch.exp(-Metric.haversine(pred, gt)/2000)