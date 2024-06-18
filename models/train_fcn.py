from torch import nn
import torch
from utils.freeze import freeze


class FCN8s(nn.Module):

    def __init__(self, backbone, classification_net, num_classes, layers, flag='resnet50') -> None:
        super().__init__()
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
        self.relu = nn.ReLU(inplace=True)
        freeze(self.classification_net)
        self.deconv1 = nn.ConvTranspose2d(self.layers["-1"]["dim"], self.layers["-2"]["dim"], kernel_size=3, stride=2,
                                          padding=1, dilation=1, output_padding=1)
        self.bn1 = nn.BatchNorm2d(self.layers["-2"]["dim"])
        self.deconv2 = nn.ConvTranspose2d(self.layers["-2"]["dim"], self.layers["-3"]["dim"], 3, 2, 1, 1, 1)
        self.bn2 = nn.BatchNorm2d(self.layers["-3"]["dim"])
        self.deconv3 = nn.ConvTranspose2d(self.layers["-3"]["dim"], self.layers["-3"]["dim"] // 2, 3, 2, 1, 1, 1)
        self.bn3 = nn.BatchNorm2d(self.layers["-3"]["dim"] // 2)
        self.deconv4 = nn.ConvTranspose2d(self.layers["-3"]["dim"] // 2, self.layers["-3"]["dim"] // 4, 3, 2, 1, 1, 1)
        self.bn4 = nn.BatchNorm2d(self.layers["-3"]["dim"] // 4)
        self.deconv5 = nn.ConvTranspose2d(self.layers["-3"]["dim"] // 4, self.layers["-3"]["dim"] // 8, 3, 2, 1, 1, 1)
        self.bn5 = nn.BatchNorm2d(self.layers["-3"]["dim"] // 8)
        self.classifier = nn.Conv2d(self.layers["-3"]["dim"] // 8, num_classes, kernel_size=1)

    def forward(self, x):
        d = torch.argmax(self.classification_net(x), dim=1, keepdim=True)
        output = {}
        layer_cnt = -3
        p = {'-3': 1, '-2': 1}
        for name, layer in self.backbone._modules.items():
            x = layer(x)
            if name == self.layers["{}".format(layer_cnt)]["name"]:
                output["{}".format(layer_cnt)] = x
                if layer_cnt == -3:
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
        score = self.relu(self.deconv1(x5))
        score = self.bn1(score + x4)
        score = self.relu(self.deconv2(score))
        score = self.bn2(score + x3)
        score = self.bn3(self.relu(self.deconv3(score)))
        score = self.bn4(self.relu(self.deconv4(score)))
        score = self.bn5(self.relu(self.deconv5(score)))
        score = self.classifier(score)
        return score, p
