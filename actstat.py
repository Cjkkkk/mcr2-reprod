import torch
from mcr2 import MaximalCodingRateReduction

class ActStat(torch.nn.Module):
    def __init__(self, act_func):
        super(ActStat, self).__init__()
        self.act_func = act_func
        self.mcr2_func = MaximalCodingRateReduction()
        
        self._enable = False
        self._labels = None
        self._mcr2 = []
    
    def enable(self):
        print("begin mcr2 record.")
        self._enable = True

    def disable(self):
        self._enable = False

    def labels(self, _labels):
        self._labels = _labels
    
    def calc_mcr2(self):
        res = 0
        for cr2 in self._mcr2:
            res += cr2
        res = res / len(self._mcr2)
        self._mcr2 = []
        return res
    
    def forward(self, x):
        x = self.act_func(x)
        if self._enable:
            self._mcr2.append(self.mcr2_func(x, self._labels, 10))
        return x