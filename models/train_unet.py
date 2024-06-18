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
        d = torch.argmax(self.classification_net(x), dim=1, keepdim=True)
        output = {}
        layer_cnt = -4
        p = {'-4': 1, '-3': 1, '-2': 1}
        for name, layer in self.backbone._modules.items():
            x = layer(x)
            if name == self.layers["{}".format(layer_cnt)]["name"]:
                output["{}".format(layer_cnt)] = x
                if layer_cnt == -4:
                    x2 = x
                    flag = False
                    for name_f, layer_f in self.frozen_backbone._modules.items():
                        if flag:
                            if name_f == 'fc':
                                x2 = torch.flatten(x2, 1)
                            x2 = layer_f(x2)
                        if name_f == self.layers["-4"]["name"]:
                            flag = True
                    if self.classification_head is not None:
                        p['-4'] = torch.gather(self.classification_head(x2), 1, d)
                    else:
                        p['-4'] = torch.gather(x2, 1, d)
                elif layer_cnt == -3:
                    x3 = x
                    flag = False
                    for name_f, layer_f in self.frozen_backbone._modules.items():
                        if flag:
                            if name_f == 'fc':
                                x3 = torch.flatten(x3, 1)
                            x3 = layer_f(x3)
                        if name_f == self.layers["-3"]["name"]:
                            flag = True
                    if self.classification_head is not None:
                        p['-3'] = torch.gather(self.classification_head(x3), 1, d)
                    else:
                        p['-3'] = torch.gather(x3, 1, d)
                elif layer_cnt == -2:
                    x4 = x
                    flag = False
                    for name_f, layer_f in self.frozen_backbone._modules.items():
                        if flag:
                            if name_f == 'fc':
                                x4 = torch.flatten(x4, 1)
                            x4 = layer_f(x4)
                        if name_f == self.layers["-2"]["name"]:
                            flag = True
                    if self.classification_head is not None:
                        p['-2'] = torch.gather(self.classification_head(x4), 1, d)
                    else:
                        p['-2'] = torch.gather(x4, 1, d)
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
        return d0, p
