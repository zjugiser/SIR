class Config:
    dataset_name = 'WHDLD'  # 'WHDLD', 'Vaihingen'
    data_path = 'your dataset path'  # dataset path
    img_sub_path = 'Images'  # image file name
    label_sub_path = 'ImagesPNG'  # label file name
    label_num = 6
    lr = 1e-4  # decoder initial learning rate
    minus2_lr = 1e-5  # encoder initial learning rate (by step)
    minus3_lr = 1e-5  # encoder initial learning rate (by step)
    minus4_lr = 1e-5  # encoder initial learning rate (by step)
    weight_decay = 1e-4
    batch_size = 16
    epoch_num = 20
    net_name = "fcn"  # deeplabv3+, fcn, unet
    pretrain_name = "efficientnetb0"  # efficientnetb0, resnet50, vgg16,
    model_name = net_name + pretrain_name
    resnet50_layers = {"-4": {"name": "layer1", "dim": 256}, "-3": {"name": "layer2", "dim": 512},
                       "-2": {"name": "layer3", "dim": 1024}, "-1": {"name": "layer4", "dim": 2048}}
    vgg16_bn_layers = {"-5": {"name": "6", "dim": 64}, "-4": {"name": "13", "dim": 128},
                       "-3": {"name": "23", "dim": 256}, "-2": {"name": "33", "dim": 512},
                       "-1": {"name": "43", "dim": 512}}
    efficientnet_b0_layers = {"-5": {"name": "1", "dim": 16}, "-4": {"name": "2", "dim": 24},
                              "-3": {"name": "3", "dim": 40}, "-2": {"name": "5", "dim": 112},
                              "-1": {"name": "7", "dim": 320}}
    loading_checkpoint_path = 'fcnefficientnetb0_WHDLD_final.pth'
