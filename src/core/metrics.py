import torch.nn as nn

class MetricCollection(nn.ModuleDict):
    def __init__(self, metrics, prefix=None, postfix=None):
        super().__init__()
        self.prefix = prefix
        self.postfix = postfix
        self.add_metrics(metrics)

    def forward(self, *args):
        res = {k: m(*args).item() for k, m in self.items()}
        return {self._set_name(k): v for k, v in res.items()}

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

    def clone(self, prefix=None, postfix=None):
        mc = deepcopy(self)
        if prefix:
            mc.prefix = prefix
        if postfix:
            mc.postfix = postfix
        return mc

    def _set_name(self, base: str) -> str:
        name = base if self.prefix is None else self.prefix + base
        name = name if self.postfix is None else name + self.postfix
        return name

