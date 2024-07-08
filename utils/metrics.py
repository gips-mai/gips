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
        R = 6371  # radius of the earth

        lat1, lon1 = torch.deg2rad(pred).t()
        lat2, lon2 = torch.deg2rad(gt).t()

        dlat = lat2 - lat1
        dlon = lon2 - lon1

        a = torch.sin(dlat/2)**2 + torch.cos(lat1) * torch.cos(lat2) * torch.sin(dlon/2)**2
        c = 2 * torch.atan2(torch.sqrt(a), torch.sqrt(1-a))

        distance = R * c

        return distance

    def geogame_score(pred:torch.Tensor, gt:torch.Tensor, factor:int=2000):
        # expects inputs to be np arrays in (lat, lon) format
        # N x 2

        # the factor is typically 2000 or 1492.7

        return 5000 * torch.exp(-Metric.haversine(pred, gt)/factor)