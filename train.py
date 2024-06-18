import torch
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from torchvision.models import ResNet50_Weights, resnet50, VGG16_BN_Weights, vgg16_bn, efficientnet_b0, \
    EfficientNet_B0_Weights
from config import Config
from dataset import SegDataset
from models.train_fcn import FCN8s
from models.train_unet import UNet
from models.train_deeplabv3_plus import DeepLab

device = 'cuda' if torch.cuda.is_available() else 'cpu'


def train(backbone, frozen_backbone, train_dataloader, val_dataloader, classifier_dim):
    lr = Config.lr
    weight_decay = Config.weight_decay
    epoch_num = Config.epoch_num
    num_classes = Config.label_num
    if Config.net_name == "fcn":
        model = FCN8s(backbone, frozen_backbone, num_classes, classifier_dim, Config.pretrain_name).to(device)
    elif Config.net_name == "unet":
        model = UNet(backbone, frozen_backbone, num_classes, classifier_dim, Config.pretrain_name).to(device)
    else:
        model = DeepLab(backbone, frozen_backbone, num_classes, classifier_dim, Config.pretrain_name).to(device)
    loss_func = torch.nn.CrossEntropyLoss(reduction='none').to(device)
    params = []
    flag = False
    if Config.net_name == "deeplabv3+":
        for name, layer in model.backbone._modules.items():
            if flag:
                params.append({'params': layer.parameters()})
            if name == classifier_dim["-3"]["name"]:
                flag = True
            elif name == classifier_dim["-1"]["name"]:
                flag = False
    else:
        for name, layer in model.backbone._modules.items():
            if flag:
                params.append({'params': layer.parameters()})
            if name == classifier_dim["-3"]["name"]:
                flag = True
            elif name == classifier_dim["-2"]["name"]:
                flag = False
    optimizer_minus_3 = torch.optim.Adam(params, lr=Config.minus3_lr, weight_decay=weight_decay)
    optimizer_minus_2 = None
    if Config.net_name == "fcn" or Config.net_name == "unet":
        params = []
        flag = False
        for name, layer in model.backbone._modules.items():
            if flag:
                params.append({'params': layer.parameters()})
            if name == classifier_dim["-2"]["name"]:
                flag = True
            elif name == classifier_dim["-1"]["name"]:
                flag = False
        optimizer_minus_2 = torch.optim.Adam(params, lr=Config.minus2_lr, weight_decay=weight_decay)
    optimizer_minus_4 = None
    if Config.net_name == "unet":
        params = []
        flag = False
        for name, layer in model.backbone._modules.items():
            if flag:
                params.append({'params': layer.parameters()})
            if name == classifier_dim["-4"]["name"]:
                flag = True
            elif name == classifier_dim["-3"]["name"]:
                flag = False
        optimizer_minus_4 = torch.optim.Adam(params, lr=Config.minus4_lr, weight_decay=weight_decay)
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr,
                                 weight_decay=weight_decay)
    writer = SummaryWriter()
    for epoch in range(1, epoch_num + 1):
        loss_sum = 0
        model.train()
        for index, batch in enumerate(train_dataloader):
            x, ground_truth = batch
            x = x.to(device)
            ground_truth = ground_truth.to(device)
            if optimizer_minus_4 is not None:
                y_pre, p = model(x)
                loss_base = loss_func(y_pre.to(torch.float), ground_truth.to(torch.long))
                loss_minus_4 = torch.mean(p['-4'].reshape(-1, 1, 1).expand_as(loss_base) * loss_base)
                optimizer_minus_4.zero_grad()
                loss_minus_4.backward()
                optimizer_minus_4.step()
            y_pre, p = model(x)
            loss_base = loss_func(y_pre.to(torch.float), ground_truth.to(torch.long))
            loss_minus_3 = torch.mean(p['-3'].reshape(-1, 1, 1).expand_as(loss_base) * loss_base)
            optimizer_minus_3.zero_grad()
            loss_minus_3.backward()
            optimizer_minus_3.step()
            if optimizer_minus_2 is not None:
                y_pre, p = model(x)
                loss_base = loss_func(y_pre.to(torch.float), ground_truth.to(torch.long))
                loss_minus_2 = torch.mean(p['-2'].reshape(-1, 1, 1).expand_as(loss_base) * loss_base)
                optimizer_minus_2.zero_grad()
                loss_minus_2.backward()
                optimizer_minus_2.step()
            y_pre, _ = model(x)
            loss_base = loss_func(y_pre.to(torch.float), ground_truth.to(torch.long))
            loss = torch.mean(loss_base)
            loss_sum += loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print('epoch_{}: train_loss is {}'.format(epoch, loss_sum))
        writer.add_scalar('Train loss', loss_sum, epoch)
        loss_sum = 0
        model.eval()
        with torch.no_grad():
            for index, batch in enumerate(val_dataloader):
                x, ground_truth = batch
                x = x.to(device)
                ground_truth = ground_truth.to(device)
                y_pre, _ = model(x)
                loss = torch.mean(loss_func(y_pre.to(torch.float32), ground_truth.to(torch.long)))
                loss_sum += loss
        print('epoch_{}: val_loss is {}'.format(epoch, loss_sum))
        writer.add_scalar('Val loss', loss_sum, epoch)
        if epoch % 5 == 0:
            torch.save(model.state_dict(),
                       'checkpoints/' + Config.model_name + '_' + Config.dataset_name + '_{}.pth'.format(epoch))
    writer.close()
    return model


if __name__ == '__main__':
    train_dataset = SegDataset(Config.data_path, 'train', num_classes=Config.label_num)
    train_dataloader = DataLoader(train_dataset, batch_size=Config.batch_size, shuffle=True)
    val_dataset = SegDataset(Config.data_path, 'val', num_classes=Config.label_num)
    val_dataloader = DataLoader(val_dataset, batch_size=Config.batch_size, shuffle=True)
    test_dataset = SegDataset(Config.data_path, 'test', num_classes=Config.label_num)
    test_dataloader = DataLoader(test_dataset, batch_size=Config.batch_size, shuffle=True)
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
    model = train(backbone, frozen_backbone, train_dataloader, val_dataloader, classifier_dim)
    torch.save(model.state_dict(),
               'checkpoints/' + Config.model_name + '_' + Config.dataset_name + '_final.pth')
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
