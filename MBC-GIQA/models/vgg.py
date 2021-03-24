'''VGG for CIFAR10. FC layers are removed.
(c) YANG, Wei 
'''
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import math
import torch

__all__ = [
    'VGG', 'vgg11', 'vgg11_bn', 'vgg13', 'vgg13_bn', 'vgg16', 'vgg16_bn',
    'vgg19_bn', 'vgg19', 'vgg19_bn_reg', 'VGG_reg', 'vgg19_bn_mulcla'
]


model_urls = {
    'vgg11': 'https://download.pytorch.org/models/vgg11-bbd30ac9.pth',
    'vgg13': 'https://download.pytorch.org/models/vgg13-c768596a.pth',
    'vgg16': 'https://download.pytorch.org/models/vgg16-397923af.pth',
    'vgg19': 'https://download.pytorch.org/models/vgg19-dcbb9e9d.pth',
}

class VGG_mulcla(nn.Module):
    def __init__(self, features, num_classes, input_size=192):
        super(VGG_mulcla, self).__init__()
        self.input_size = input_size
        self.features = features
        if input_size == 192:
            self.regression1 = nn.Linear(512*3*3, 1024)
        elif input_size == 256 or input_size == 128 or input_size == 64 or input_size == 32:
            self.regression1 = nn.Linear(512*4*4, 1024)
        elif input_size ==224:
            self.regression1 = nn.Linear(512*7*7, 1024)
        else:
            print("not a good choice !!!!!!!")
            self.regression1 = nn.Linear(512*2*2, 1024)
        self.num_class = num_classes
        self.regression2 = nn.Linear(1024, num_classes)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self._initialize_weights()

    def forward(self, x, split_patch=False):
        # print("ok1 ")
        x = self.features(x)
        # print("ok 2 ")
        # print(x.size())
        if split_patch == True and self.input_size!=256 and self.input_size!=224 and self.input_size!=192:
            input_size = self.input_size
            assert input_size == 128 or input_size == 64 or input_size == 32
            patch_number = int(256/input_size)
            all_tensor = torch.zeros(x.size()[0], self.num_class).type_as(x)
            
            for i in range(patch_number):
                for j in range(patch_number):
                    this_patch = x[:,:,4*i:4*i+4,4*j:4*j+4]
            
                
                    y = this_patch.contiguous().view(this_patch.size(0), -1)
                    y = self.regression1(y)
                    y = self.relu(y)
                    y = self.regression2(y)
                    y = self.sigmoid(y)
                    all_tensor = all_tensor + y
            x = all_tensor/(patch_number*patch_number)
        else:
            x = x.view(x.size(0), -1)
            x = self.regression1(x)
            # print("ok 3 ")
            x = self.relu(x)
            x = self.regression2(x)
            x = self.sigmoid(x)
        
        
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

class VGG_reg(nn.Module):
    def __init__(self, features):
        super(VGG_reg, self).__init__()
        self.features = features
        self.regression1 = nn.Linear(512*3*3, 1024)
        self.regression2 = nn.Linear(1024, 1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.regression1(x)
        x = self.relu(x)
        x = self.regression2(x)
        x = self.sigmoid(x)

        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()


class VGG(nn.Module):

    def __init__(self, features, num_classes=1000):
        super(VGG, self).__init__()
        self.features = features
        self.classifier = nn.Linear(512, num_classes)
        self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()


def make_layers(cfg, batch_norm=False):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


cfg = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
    'F': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M', 'M'],
    'G': [64, 64, 128, 128, 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
    'H': [64, 64, 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
    'I': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
    'J': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M', 512, 512, 'M'],
    'K': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


def vgg11(**kwargs):
    """VGG 11-layer model (configuration "A")

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = VGG(make_layers(cfg['A']), **kwargs)
    return model


def vgg11_bn(**kwargs):
    """VGG 11-layer model (configuration "A") with batch normalization"""
    model = VGG(make_layers(cfg['A'], batch_norm=True), **kwargs)
    return model


def vgg13(**kwargs):
    """VGG 13-layer model (configuration "B")

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = VGG(make_layers(cfg['B']), **kwargs)
    return model


def vgg13_bn(**kwargs):
    """VGG 13-layer model (configuration "B") with batch normalization"""
    model = VGG(make_layers(cfg['B'], batch_norm=True), **kwargs)
    return model


def vgg16(**kwargs):
    """VGG 16-layer model (configuration "D")

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = VGG(make_layers(cfg['D']), **kwargs)
    return model


def vgg16_bn(**kwargs):
    """VGG 16-layer model (configuration "D") with batch normalization"""
    model = VGG(make_layers(cfg['D'], batch_norm=True), **kwargs)
    return model


def vgg19(**kwargs):
    """VGG 19-layer model (configuration "E")

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = VGG(make_layers(cfg['E']), **kwargs)
    return model


def vgg19_bn(**kwargs):
    """VGG 19-layer model (configuration 'E') with batch normalization"""
    model = VGG(make_layers(cfg['E'], batch_norm=True), **kwargs)
    return model

def vgg19_bn_reg():
    """VGG 19-layer model (configuration 'E') with batch normalization"""
    model = VGG_reg(make_layers(cfg['F'], batch_norm=True))
    return model

def vgg19_bn_mulcla(num_classes=8, input_size=192):
    """VGG 19-layer model (configuration 'E') with batch normalization"""
    if input_size == 192:
        model = VGG_mulcla(make_layers(cfg['F'], batch_norm=True), num_classes, input_size)
    if input_size == 32:
        model = VGG_mulcla(make_layers(cfg['G'], batch_norm=True), num_classes, input_size)
    if input_size == 64:
        model = VGG_mulcla(make_layers(cfg['H'], batch_norm=True), num_classes, input_size)
    if input_size == 128:
        model = VGG_mulcla(make_layers(cfg['I'], batch_norm=True), num_classes, input_size)
    if input_size == 256:
        model = VGG_mulcla(make_layers(cfg['J'], batch_norm=True), num_classes, input_size)
    if input_size == 224:
        model = VGG_mulcla(make_layers(cfg['K'], batch_norm=True), num_classes, input_size)
    return model
