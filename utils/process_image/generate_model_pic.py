import torch
from module.myModel2 import MyNet

from tensorboardX import SummaryWriter

x = torch.rand(1, 3, 512, 512)
model = MyNet()

with SummaryWriter(comment='MyNet') as w:
    w.add_graph(model, x)
