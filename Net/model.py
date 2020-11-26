import torch.nn as nn
import torch
from tensorboardX import SummaryWriter
from torchsummary import summary



class Net(nn.Module):
    def __init__(self, features):
        super(Net, self).__init__()

        self.layer0 = nn.Sequential(nn.Linear(features, 16), nn.ReLU(),nn.BatchNorm1d(16))

        self.layer1 = nn.Sequential(nn.Linear(16, 32), nn.ReLU())
        self.dropout1 = nn.Dropout(p=0.25)

        self.layer2 = nn.Sequential(nn.Linear(32, 64), nn.ReLU())
        self.dropout2 = nn.Dropout(p=0.25)

        self.layer3 = nn.Sequential(nn.Linear(64, 128), nn.ReLU())
        self.dropout3 = nn.Dropout(p=0.25)

        self.layer4 = nn.Sequential(nn.Linear(128, 256), nn.ReLU())
        self.dropout4 = nn.Dropout(p=0.25)

        self.layer5 = nn.Sequential(nn.Linear(256, 512), nn.ReLU())

        self.layer6 = nn.Sequential(nn.Linear(512, 1024), nn.ReLU())

        self.layer7 = nn.Sequential(nn.Linear(1024, 1024), nn.ReLU())

        self.layer8 = nn.Sequential(nn.Linear(1024, 1024), nn.ReLU())

        self.layer9 = nn.Sequential(nn.Linear(1024, 1024), nn.ReLU())

        self.layer10 = nn.Sequential(nn.Linear(1024, 256), nn.ReLU())

        self.layer11 = nn.Sequential(nn.Linear(256, 64), nn.ReLU())

        self.layer12 = nn.Sequential(nn.Linear(64, 16), nn.ReLU())

        self.layer13 = nn.Sequential(nn.Linear(16, 1), nn.ReLU())

    def forward(self, x):
        y_pred = self.layer0(x)
        y_pred = self.layer1(y_pred)
        # y_pred = self.dropout1(y_pred)
        y_pred = self.layer2(y_pred)
        # y_pred = self.dropout2(y_pred)
        y_pred = self.layer3(y_pred)
        # y_pred = self.dropout3(y_pred)
        y_pred = self.layer4(y_pred)
        # y_pred = self.dropout4(y_pred)
        y_pred = self.layer5(y_pred)
        y_pred = self.layer6(y_pred)
        y_pred = self.layer7(y_pred)
        y_pred = self.layer8(y_pred)
        y_pred = self.layer9(y_pred)
        y_pred = self.layer10(y_pred)
        y_pred = self.layer11(y_pred)
        y_pred = self.layer12(y_pred)
        y_pred = self.layer13(y_pred)

        return y_pred


class Howard(nn.Module):
    def __init__(self, features):
        super(Howard, self).__init__()

        self.linear_relu1 = nn.Linear(features, 64)
        self.linear_relu2 = nn.Linear(64, 256)
        self.linear_relu3 = nn.Linear(256, 256)
        self.linear_relu4 = nn.Linear(256, 256)
        self.linear_relu5 = nn.Linear(256, 256)
        self.linear_relu6 = nn.Linear(256, 256)
        self.linear_relu7 = nn.Linear(256, 256)
        self.linear_relu8 = nn.Linear(256, 256)
        self.linear_relu9 = nn.Linear(256, 256)
        self.linear_relu10 = nn.Linear(256, 256)
        self.linear_relu11 = nn.Linear(256, 256)
        self.linear_relu12 = nn.Linear(256, 256)
        self.linear_relu13 = nn.Linear(256, 256)
        self.linear_relu14 = nn.Linear(256, 16)
        self.linear_relu15 = nn.Linear(16, features)
        self.linear_relu16 = nn.Linear(features, 1)

    def forward(self, x):

        y_pred = self.linear_relu1(x)
        y_pred = nn.functional.relu(y_pred)

        y_pred = self.linear_relu2(y_pred)
        y_pred = nn.functional.relu(y_pred)

        y_pred = self.linear_relu3(y_pred)
        y_pred = nn.functional.relu(y_pred)

        y_pred = self.linear_relu4(y_pred)
        y_pred = nn.functional.relu(y_pred)

        y_pred = self.linear_relu5(y_pred)
        y_pred = nn.functional.relu(y_pred)

        y_pred = self.linear_relu6(y_pred)
        y_pred = nn.functional.relu(y_pred)

        y_pred = self.linear_relu7(y_pred)
        y_pred = nn.functional.relu(y_pred)

        y_pred = self.linear_relu8(y_pred)
        y_pred = nn.functional.relu(y_pred)

        y_pred = self.linear_relu9(y_pred)
        y_pred = nn.functional.relu(y_pred)

        y_pred = self.linear_relu10(y_pred)
        y_pred = nn.functional.relu(y_pred)

        y_pred = self.linear_relu11(y_pred)
        y_pred = nn.functional.relu(y_pred)

        y_pred = self.linear_relu12(y_pred)
        y_pred = nn.functional.relu(y_pred)

        y_pred = self.linear_relu13(y_pred)
        y_pred = nn.functional.relu(y_pred)

        y_pred = self.linear_relu14(y_pred)
        y_pred = nn.functional.relu(y_pred)

        y_pred = self.linear_relu15(y_pred)
        y_pred = nn.functional.relu(y_pred)

        y_pred = self.linear_relu16(y_pred)

        return y_pred


class JackNet(nn.Module):
    def __init__(self, features):
        super(JackNet, self).__init__()

        self.layer0 = nn.Sequential(nn.Linear(features, 128), nn.ReLU())

        self.layer1 = nn.Sequential(nn.Linear(128, 256), nn.ReLU())
        self.dropout1 = nn.Dropout(p=0.25)

        self.layer2 = nn.Sequential(nn.Linear(256, 256), nn.ReLU())
        self.dropout2 = nn.Dropout(p=0.25)

        self.layer3 = nn.Sequential(nn.Linear(256, 256), nn.ReLU())
        self.dropout3 = nn.Dropout(p=0.25)

        self.layer4 = nn.Sequential(nn.Linear(256, 256), nn.ReLU())
        self.dropout4 = nn.Dropout(p=0.25)

        self.layer5 = nn.Sequential(nn.Linear(256, 256), nn.ReLU())

        self.layer6 = nn.Sequential(nn.Linear(256, 128), nn.ReLU())

        self.layer7 = nn.Sequential(nn.Linear(128, 1))

    def forward(self, x):
        y_pred = self.layer0(x)
        y_pred = self.layer1(y_pred)
        # y_pred = self.dropout1(y_pred)
        y_pred = self.layer2(y_pred)
        # y_pred = self.dropout2(y_pred)
        y_pred = self.layer3(y_pred)
        # y_pred = self.dropout3(y_pred)
        y_pred = self.layer4(y_pred)
        # y_pred = self.dropout4(y_pred)
        y_pred = self.layer5(y_pred)
        y_pred = self.layer6(y_pred)
        y_pred = self.layer7(y_pred)
        # y_pred = self.layer8(y_pred)
        # y_pred = self.layer9(y_pred)
        # y_pred = self.layer10(y_pred)
        # y_pred = self.layer11(y_pred)
        # y_pred = self.layer12(y_pred)

        return y_pred
class fusion_net(nn.Module):
    def __init__(self, features):
        super(fusion_net, self).__init__()
        self.layer1 = nn.Sequential(nn.Linear(features, 8), nn.ReLU())
        self.layer2 = nn.Sequential(nn.Linear(8, 8), nn.ReLU())
        self.layer3 = nn.Sequential(nn.Linear(8, 4), nn.ReLU())
        self.layer4 = nn.Sequential(nn.Linear(4, 2), nn.ReLU())
        self.layer5 = nn.Sequential(nn.Linear(2, 1))

    def forward(self, x):
        y_pred = self.layer1(x)
        y_pred = self.layer2(y_pred)
        y_pred = self.layer3(y_pred)
        y_pred = self.layer4(y_pred)
        y_pred = self.layer5(y_pred)

        return y_pred

#畫出模型架構
# x = torch.rand(1, 5)
# model = fusion_net(5).cuda()
# summary(model, (1,5))

# with SummaryWriter(comment='Net') as w:
#     w.add_graph(model, x)