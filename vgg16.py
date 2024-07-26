import torch
import torch.nn as nn
from torch.hub import load_state_dict_from_url


#--------------------------------------#
#   VGG16的结构
#--------------------------------------#
class VGG(nn.Module):
    def __init__(self, features, num_classes=1000, init_weights=True):
        super(VGG, self).__init__()
        self.features = features
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))        #平均池化到7x7大小
        self.classifier = nn.Sequential(        #分类部分
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, num_classes),
        )
        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        x = self.features(x)        #特征提取
        x = self.avgpool(x)        #平均池化
        x = torch.flatten(x, 1)        #平铺后
        x = self.classifier(x)        #分类部分
        return x
    '''
    初始化权重的函数
    '''
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode = 'fan_out', nonlinearity = 'relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

'''
假设输入图像为(600, 600, 3),随着cfg的循环,特征层变化如下：
600,600,3 -> 600,600,64 -> 600,600,64 -> 300,300,64 -> 300,300,128 -> 300,300,128 -> 150,150,128 -> 150,150,256 -> 150,150,256 -> 150,150,256 
-> 75,75,256 -> 75,75,512 -> 75,75,512 -> 75,75,512 -> 37,37,512 ->  37,37,512 -> 37,37,512 -> 37,37,512
到cfg结束,我们获得了一个37,37,512的特征层
'''

cfg = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M']


#--------------------------------------#
#   特征提取部分
#cfg：列表，定义了卷积层的结构，列表中的每个元素可以是一个整数v，表示卷积层的输出通道数；或者字符'M'，表示最大池化层
#batch_norm：布尔值，默认为False。如果设置为True，则在卷积层后添加批归一化层
#最后返回一个nn.Sequential对象，包含了所有的卷积层
#--------------------------------------#
'''
这个函数的作用是根据给定的配置cfg创建一个卷积层序列，这个序列可以作为VGG模型的特征提取部分的基础网络结构
'''
def make_layers(cfg, batch_norm = False):
    layers = []
    in_channels = 3  #RGB图像
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size = 2, stride = 2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size = 3, padding = 1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace = True)]
            else:
                layers += [conv2d, nn.ReLU(inplace = True)]
            in_channels = v 
    return nn.Sequential(*layers)
'''
decom_vgg16函数对VGG16模型进行分解
pretrained是一个布尔值，用于指定是否加载预训练的VGG16模型的权重
'''
def decom_vgg16(pretrained = False):
    model = VGG(make_layers(cfg))
    if pretrained:
        state_dict = load_state_dict_from_url("https://download.pytorch.org/models/vgg16-397923af.pth", model_dir = "./model_data")  #如果 pretrained 为 True，则使用 load_state_dict_from_url 函数从指定的URL下载预训练的VGG16模型权重，并加载到 model 中
        model.load_state_dict(state_dict)

    features = list(model.features)[:30]    #获取特征提取部分,最终获得一个37,37,1024的特征层
    classifier = list(model.classifier)    #获取分类部分,需要除去Dropout部分
    del classifier[6]
    del classifier[5]
    del classifier[2]

    features = nn.Sequential(*features)
    classifier = nn.Sequential(*classifier)
    return features, classifier  #将特征提取部分和分类部分分别封装成 nn.Sequential 对象，并返回这两个对象
