import torch.nn as nn

class CombinedModel(nn.Module):
    def __init__(self, model1, model2):
        super(CombinedModel, self).__init__()
        self.model1 = model1
        self.model2 = model2

    def forward(self, x , target):
        out1,loss = self.model1(x)
        out2 = self.model2(out1,target)
        return out1,loss,out2  # 可以根据任务进一步处理
