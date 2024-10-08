import torch.nn as nn
import torchvision
import torch.nn.functional as F
from efficientnet_pytorch import EfficientNet

from collections import OrderedDict
from torchvision.models import ResNet50_Weights

class FE(nn.Module):
    def __init__(self):
        super().__init__()

    def get_cnn(self):
        raise NotImplementedError()

    def get_n_features(self):
        raise NotImplementedError()

    def forward(self, inputs):
        '''
        :param inputs: [B, C, H, W]
        :returms: [B, C', H', W']
        '''
        return self.get_cnn()(inputs)  # [B, C', H', W']


class VGGFE(FE):
    version = {
        'vgg16': torchvision.models.vgg16,
        'vgg16_bn': torchvision.models.vgg16_bn,
        'vgg19_bn': torchvision.models.vgg19_bn,
    }

    def __init__(self, version, pretrained, droplast):
        super().__init__()
        vgg = self.version[version](pretrained).features[:-1]
        max_pool_idx = list(filter(lambda x: isinstance(x[1], nn.MaxPool2d), enumerate(vgg)))

        if droplast == 0:
            self.cnn = vgg
        else:
            self.cnn = vgg[:max_pool_idx[-droplast][0]]

        if isinstance(self.cnn[-2], nn.BatchNorm2d):
            self.n_features = self.cnn[-2].num_features
        else:
            self.n_features = self.cnn[-2].out_channels

    def get_cnn(self):
        return self.cnn

    def get_n_features(self):
        return self.n_features

    def forward(self, inputs):
        return self.cnn(inputs)


class DenseNetFE(FE):

    version = {
        'densenet161': torchvision.models.densenet161,
        'densenet121': torchvision.models.densenet121,
    }

    def __init__(self, version, pretrained, memory_efficient):
        super().__init__()
        if version == 'custom':
            densenet = torchvision.models.DenseNet(
                growth_rate=96,
                block_config=(4,4,4,),
                num_init_features=48,
                memory_efficient=memory_efficient,
            )
        else:
            densenet = self.version[version](pretrained, memory_efficient)
        self.cnn = densenet.features
        self.n_features = densenet.classifier.in_features

    def get_cnn(self):
        return self.cnn

    def get_n_features(self):
        return self.n_features

    def forward(self, inputs):
        return self.cnn.forward(inputs)

class EfficientNetFE(FE):
    def __init__(self, pretrained_name="efficientnet-b0"):
        super().__init__()
        self.effnet = EfficientNet.from_pretrained(pretrained_name, advprop=True)
        self.cnn = self.effnet.extract_features
        self.n_features = self.effnet._fc.in_features

    def get_cnn(self):
        return self.cnn

    def get_n_features(self):
        return self.n_features

    def forward(self, inputs):
        features = super().forward(inputs)
        out = F.adaptive_avg_pool2d(features, (1, None))
        return out

class SqueezeNetFE(FE):
    def __init__(self):
        super().__init__()
        squeezenet = torchvision.models.squeezenet1_1(weights=ResNet50_Weights.DEFAULT)
        self.cnn = squeezenet.features
        self.n_features = 512

    def get_cnn(self):
        return self.cnn
    def get_n_features(self):
        return self.n_features

'''
LeNet-5
https://engmrk.com/lenet-5-a-classic-cnn-architecture/
https://github.com/bollakarthikeya/LeNet-5-PyTorch/blob/master/lenet5_gpu.py
https://www.kaggle.com/usingtc/lenet-with-pytorch
'''
class LeNetFE(FE):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, stride=1, padding=2, bias=True)
        self.max_pool_1 = nn.MaxPool2d(kernel_size=2)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1, padding=0, bias=True)
        self.max_pool_2 = nn.MaxPool2d(kernel_size=2) 
        self.fc1 = nn.Linear(16*5*5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.activation = F.relu()

        self.n_features = 84

    def forward(self, inputs):
        x = self.activation(self.conv1(inputs))
        x = self.max_pool_1(x)
        x = self.activation(self.conv2(x))
        x = self.max_pool_2(x)
        x = x.view(-1, 16*5*5)
        x = self.activation(self.fc1(x))
        x = self.fc2(x)
        return x

    def get_n_features(self):
        return self.n_features

class CustomFE(FE):
    '''
    An Efficient End-to-End Neural Model for Handwritten Text Recognition (2018)
    '''
    def __init__(self, in_channel=3):
        super().__init__()

        self.cnn = nn.Sequential(OrderedDict([
            ('conv1', nn.Conv2d(in_channel, 16, 3, padding=1)),
            ('lrelu1', nn.LeakyReLU(inplace=True)),
            ('pool1', nn.MaxPool2d(2)),
            ('bn1', nn.BatchNorm2d(16)),

            ('conv2', nn.Conv2d(16, 32, 3, padding=1)),
            ('lrelu2', nn.LeakyReLU(inplace=True)),
            ('pool2', nn.MaxPool2d(2)),
            ('bn2', nn.BatchNorm2d(32)),

            ('conv3', nn.Conv2d(32, 64, 3, padding=1)),
            ('lrelu3', nn.LeakyReLU(inplace=True)),
            ('bn3', nn.BatchNorm2d(64)),

            ('conv4', nn.Conv2d(64, 64, 3, padding=1)),
            ('lrelu4', nn.LeakyReLU(inplace=True)),
            ('bn4', nn.BatchNorm2d(64)),

            ('conv5', nn.Conv2d(64, 128, 3, padding=1)),
            ('lrelu5', nn.LeakyReLU(inplace=True)),
            ('pool5', nn.MaxPool2d((1,2))),
            ('bn5', nn.BatchNorm2d(128)),

            ('conv6', nn.Conv2d(128, 128, 3, padding=1)),
            ('lrelu6', nn.LeakyReLU(inplace=True)),
            ('pool6', nn.MaxPool2d((1,2))),
            ('bn6', nn.BatchNorm2d(128)),

            ('conv7', nn.Conv2d(128, 128, (2,2))),
            ('lrelu7', nn.LeakyReLU(inplace=True)),
            ('bn7', nn.BatchNorm2d(128)),
        ]))
        self.n_features = 128

    def get_cnn(self):
        return self.cnn
    def get_n_features(self):
        return self.n_features

class ResnetFE(FE):

    version = {
        'resnet50': torchvision.models.resnet50,
        'resnet18': torchvision.models.resnet18,
        'resnet34': torchvision.models.resnet34,
    }

    def __init__(self, version='resnet50', droplast: int = 0):
        super().__init__()
        resnet = ResnetFE.version[version](weights=ResNet50_Weights.DEFAULT)
        self.cnn = nn.Sequential(*list(resnet.children())[:-2-droplast])
        for child in reversed(list(self.cnn[-1][-1].children())):
            if isinstance(child, nn.BatchNorm2d):
                self.n_features = child.num_features
                break

    def get_cnn(self):
        return self.cnn

    def get_n_features(self):
        return self.n_features

    def forward(self, x):
        x = self.cnn(x)
        return x

class ResnextFE(FE):

    version = {
        'resnext50': torchvision.models.resnext50_32x4d,
        'resnext101': torchvision.models.resnext101_32x8d,
    }

    def __init__(self, version='resnext50', droplast: int = 0):
        super().__init__()
        resnet = ResnextFE.version[version](weights=ResNet50_Weights.DEFAULT)
        self.cnn = nn.Sequential(*list(resnet.children())[:-2-droplast])
        for child in reversed(list(self.cnn[-1][-1].children())):
            if isinstance(child, nn.BatchNorm2d):
                self.n_features = child.num_features
                break

    def get_cnn(self):
        return self.cnn

    def get_n_features(self):
        return self.n_features

    def forward(self, x):
        x = self.cnn(x)
        return x

class DeformResnetFE(ResnetFE):

    version = {
        'resnet50': torchvision.models.resnet50,
        'resnet18': torchvision.models.resnet18,
        'resnet34': torchvision.models.resnet34,
    }

    def __init__(self, version='resnet50', groups=1, kernel_size=(3,3)):
        super().__init__(version)
        self.offsets = nn.Conv2d(self.n_features, 2*groups*kernel_size[0]*kernel_size[1], kernel_size=kernel_size)
        self.deform = torchvision.ops.DeformConv2d(self.n_features, self.n_features, kernel_size, groups=groups)
        self.bn = nn.BatchNorm2d(self.deform.out_channels)

    def get_cnn(self):
        return self.cnn

    def get_n_features(self):
        return self.n_features

    def forward(self, x):
        x = self.cnn(x)
        offsets = self.offsets(x)
        x = self.deform(x, offsets)
        x = F.relu(self.bn(x))
        x = self.pool(x)
        return x
