def freeze(layer):
    for child in layer.children():
        for param in child.parameters():
            param.requires_grad = False