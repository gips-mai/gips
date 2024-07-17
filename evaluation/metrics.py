import torch

# Inspiration from https://github.com/gastruc/osv5m/blob/main/metrics/utils.py

class Metric:
    def __init__(self):
        self.haversine_sum = 0
        self.count = 0

    def update(self, pred:torch.Tensor, gt:torch.Tensor):
        self.haversine_sum += Metric.haversine(pred, gt).sum(dim=0)
        self.count += pred.shape[0] # batch size

    def compute(self):
        # compute the Metric for all predictions
        output = {
            "Haversine": float(self.haversine_sum.item())/ self.count,
            "count": self.count
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
