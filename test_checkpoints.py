import torch
import os
import numpy as np
from torch.utils.data import DataLoader
from torchvision.models import ResNet50_Weights, resnet50, VGG16_BN_Weights, vgg16_bn, efficientnet_b0, \
    EfficientNet_B0_Weights
from config import Config
from dataset import SegDataset
from models.test_fcn import FCN8s
from models.test_unet import UNet
from models.test_deeplabv3_plus import DeepLab
from utils.evaluate_metric import IOUMetric


def test(model, test_dataloader):
    metric = IOUMetric(Config.label_num)
    model.eval()
    with torch.no_grad():
        for index, batch in enumerate(test_dataloader):
            x, ground_truth = batch
            x = x.to(device)
            ground_truth = ground_truth.to(device)
            y_pre = model(x)
            y_pre = y_pre.softmax(dim=1)
            label = torch.argmax(y_pre, dim=1)
            metric.add_batch(label.detach().cpu().numpy(), ground_truth.detach().cpu().numpy())
    acc, acc_cls, iu, mean_iu, fwavacc = metric.evaluate()
    return acc, acc_cls, iu, mean_iu, fwavacc


if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if Config.pretrain_name == 'resnet50':
        weights = ResNet50_Weights.DEFAULT
        backbone = resnet50(weights=weights).to(device)
        frozen_backbone = resnet50(weights=weights).to(device)
        classifier_dim = Config.resnet50_layers
    elif Config.pretrain_name == 'vgg16':
        weights = VGG16_BN_Weights.DEFAULT
        backbone = vgg16_bn(weights=weights).features.to(device)
        frozen_backbone = vgg16_bn(weights=weights).to(device)
        classifier_dim = Config.vgg16_bn_layers
    else:
        weights = EfficientNet_B0_Weights.DEFAULT
        backbone = efficientnet_b0(weights=weights).features.to(device)
        frozen_backbone = efficientnet_b0(weights=weights).to(device)
        classifier_dim = Config.efficientnet_b0_layers
    num_classes = Config.label_num
    if Config.net_name == "fcn":
        my_model = FCN8s(backbone, frozen_backbone, num_classes, classifier_dim, Config.pretrain_name).to(device)
    elif Config.net_name == "unet":
        my_model = UNet(backbone, frozen_backbone, num_classes, classifier_dim, Config.pretrain_name).to(device)
    else:
        my_model = DeepLab(backbone, frozen_backbone, num_classes, classifier_dim, Config.pretrain_name).to(device)
    my_model.load_state_dict(torch.load(os.path.join('checkpoints', Config.loading_checkpoint_path)))
    test_dataset = SegDataset(Config.data_path, 'test', num_classes=Config.label_num)
    test_dataloader = DataLoader(test_dataset, batch_size=Config.batch_size, shuffle=True)
    acc, acc_cls, iu, mean_iu, fwavacc = test(my_model, test_dataloader)
    res = np.concatenate((acc, acc_cls, mean_iu, fwavacc, iu), axis=0)
    np.savetxt('result/res_' + Config.loading_checkpoint_path + '.out', res)
