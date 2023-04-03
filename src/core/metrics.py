import torch.nn as nn

class MetricCollection(nn.ModuleDict):
    def __init__(self, metrics):
        super().__init__()
        self.add_metrics(metrics)

    def forward(self, *args):
        res = {k: m(*args).item() for k, m in self.items()}
        return res
    def add_metrics(self, metrics: dict):
        for name in sorted(metrics.keys()):
            metric = metrics[name]
            if not isinstance(metric, nn.Module):
                raise ValueError(
                    f"Value {metric} belonging to key {name} is not an instance of"
                    " `nn.Module` or `torchmetrics.Metric`"
                )
            if isinstance(metric, nn.Module):
                name = metric.__class__.__name__
                if name in self:
                    raise ValueError(f"Encountered two metrics both named {name}")
                self[name] = metric
            else:
                for k, v in metric.items():
                    self[k] = v
