import torch
from torch import nn
from utils.freeze import freeze


class ConvBlock(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(ConvBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch_out, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class UpConv(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(UpConv, self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.up(x)
        return x


class UNet(nn.Module):
    def __init__(self, backbone, classification_net, num_classes, layers, flag='resnet50'):
        super(UNet, self).__init__()
        self.backbone = backbone
        self.classification_net = classification_net
        if flag == 'resnet50':
            self.frozen_backbone = self.classification_net
            self.classification_head = None
        else:
            self.frozen_backbone = self.classification_net.features
            self.classification_head = nn.Sequential(self.classification_net.avgpool, nn.Flatten(1),
                                                     self.classification_net.classifier)
        self.layers = layers
        freeze(self.classification_net)
        self.up5 = UpConv(self.layers["-1"]["dim"], self.layers["-2"]["dim"])
        self.up_conv5 = ConvBlock(self.layers["-2"]["dim"] * 2, self.layers["-2"]["dim"])
        self.up4 = UpConv(self.layers["-2"]["dim"], self.layers["-3"]["dim"])
        self.up_conv4 = ConvBlock(self.layers["-3"]["dim"] * 2, self.layers["-3"]["dim"])
        self.up3 = UpConv(self.layers["-3"]["dim"], self.layers["-4"]["dim"])
        self.up_conv3 = ConvBlock(self.layers["-4"]["dim"] * 2, self.layers["-4"]["dim"])
        self.up2 = UpConv(self.layers["-4"]["dim"], self.layers["-4"]["dim"] // 2)
        self.up_conv2 = ConvBlock(self.layers["-4"]["dim"] // 2, self.layers["-4"]["dim"] // 4)
        self.up1 = UpConv(self.layers["-4"]["dim"] // 4, self.layers["-4"]["dim"] // 8)
        self.up_conv1 = ConvBlock(self.layers["-4"]["dim"] // 8, self.layers["-4"]["dim"] // 8)
        self.classifier = nn.Conv2d(self.layers["-4"]["dim"] // 8, num_classes, kernel_size=1)

    def forward(self, x):
        output = {}
        layer_cnt = -4
        for name, layer in self.backbone._modules.items():
            x = layer(x)
            if name == self.layers["{}".format(layer_cnt)]["name"]:
                output["{}".format(layer_cnt)] = x
                layer_cnt += 1
                if layer_cnt >= 0:
                    break
        x5 = output["-1"]
        x4 = output["-2"]
        x3 = output["-3"]
        x2 = output["-4"]
        d5 = self.up5(x5)
        d5 = torch.cat((x4, d5), dim=1)
        d5 = self.up_conv5(d5)
        d4 = self.up4(d5)
        d4 = torch.cat((x3, d4), dim=1)
        d4 = self.up_conv4(d4)
        d3 = self.up3(d4)
        d3 = torch.cat((x2, d3), dim=1)
        d3 = self.up_conv3(d3)
        d2 = self.up2(d3)
        d2 = self.up_conv2(d2)
        d1 = self.up1(d2)
        d1 = self.up_conv1(d1)
        d0 = self.classifier(d1)
        return d0
