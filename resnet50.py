import math

import torch.nn as nn
from torch.hub import load_state_dict_from_url

#用于构建残差网络中的瓶颈块。Bottleneck模块是ResNet中的基本构建块，通过堆叠多个Bottleneck模块可以构建深层的ResNet网络
class Bottleneck(nn.Module):
    expansion = 4
    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, stride=stride, bias=False)  # 1x1的卷积层，用于降维或升维
        self.bn1 = nn.BatchNorm2d(planes) #conv1的批归一化层

        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)#3x3的卷积层，用于进行特征提取
        self.bn2 = nn.BatchNorm2d(planes) #conv2的批归一化层

        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False) #1x1的卷积层，用于恢复特征维度
        self.bn3 = nn.BatchNorm2d(planes * 4) #conv3的批归一化层

        self.relu = nn.ReLU(inplace=True)#使用ReLU激活函数，并将计算结果直接覆盖原始输入，节省内存空间
        self.downsample = downsample #进行高和宽压缩的残差边（downsample）
        self.stride = stride  #用于控制卷积层的步长（stride）的参数。步长决定了卷积核再输入图像上滑动的步幅大小，从而影响输出特征图的大小。默认值为1，表示卷积核每次滑动一个像素。如果stride大于1，卷积层会将输入图像进行压缩，从而减少输出特征图的尺寸

    #模块的前向传播过程，首先将输入x保存为残差项residual，然后对输入进行一系列的卷积、批归一化和激活操作。将得到的特征图与残差项相加，并通过激活函数ReLU进行线性变换。最后返回输出特征图out
    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)
        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out
'''
定义了一个ResNet模型的构造函数。在构造函数中，定义了ResNet模型的各个组件，包括卷积层、批归一化层、激活函数、池化层、残差块等
'''
class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=1000):

        self.inplanes = 64        #   假设输入进来的图片是600,600,3
        super(ResNet, self).__init__()

        # 600,600,3 -> 300,300,64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)  #定义一个卷积层，输入通道数为3，输出通道数为64，卷积核大小为7*7，步长为2，padding为3
        self.bn1 = nn.BatchNorm2d(64)  #定义一个批归一化层，对卷积层的输出进行归一化处理
        self.relu = nn.ReLU(inplace=True)  #定义一个ReLU激活函数，对卷积层输出进行非线性激活

        # 300,300,64 -> 150,150,64
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=0, ceil_mode=True) #定义一个最大池化层，池化窗口大小为3*3，步长为2，padding为0

        # 150,150,64 -> 150,150,256
        self.layer1 = self._make_layer(block, 64, layers[0])  #定义ResNet的第一个残差块，输入通道数为64，输出通道数为64，重复layer[0]次
        # 150,150,256 -> 75,75,512
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)  #定义ResNet的第二个残差块，输入通道数为64，输出通道数为128，重复layer[1]次
        # 75,75,512 -> 38,38,1024 到这里可以获得一个38,38,1024的共享特征层
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2) #定义ResNet的第三个残差块，输入通道数为128，输出通道数为256，重复layer[2]次
        # self.layer4被用在classifier模型中
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2) #定义ResNet的第四个残差块，输入通道数为256，输出通道数为512，重复layer[3]次
        
        self.avgpool = nn.AvgPool2d(7)  #定义一个平均池化层，池化窗口大小为7*7，用于将特征图降维
        self.fc = nn.Linear(512 * block.expansion, num_classes) #定义一个全连接层，输入大小为512 * block.expansion，输出大小为num_classes，用于分类
        '''
        初始化神经网络模型的参数
        对模型的所有模块进行参数初始化，使得模型的参数满足某种分布或初始化策略
        '''
        for m in self.modules():
            if isinstance(m, nn.Conv2d):  #卷积层（nn.Conv2d）
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels #n为卷积核的大小乘以输出通道数
                m.weight.data.normal_(0, math.sqrt(2. / n))  #使用正态分布随机初始化权重，均值为0，标准差为math.sqrt(2./n)
            elif isinstance(m, nn.BatchNorm2d):  #批归一化层（nn.BatcNorm2d）
                m.weight.data.fill_(1)  #将权重填充为1
                m.bias.data.zero_()  #将偏差填充为0，确保初始时网络的输出具有较小的方差和较小的偏差
        '''
        用于构建残差块的函数_make_layer
        '''
    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        #-------------------------------------------------------------------#
        #   当模型需要进行高和宽的压缩的时候，就需要用到残差边的downsample
        #   如果stride不等于1或者输入通道数self.inplanes不等于输出通道数planes*block.expansion，则需要使用残差边的downsample
        #-------------------------------------------------------------------#
        if stride != 1 or self.inplanes != planes * block.expansion:
            '''
            downsample被定义为一个由一个卷积层和一个归一化层组成的顺序结构。、
            卷积层的输入通道数为self.inplanes，输出通道数为planes * block.expansion，卷积核大小为1，步幅为stride，偏差为False。批归一化层的输入通道数为planes * block.expansion
            '''
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )
        layers = []  #定义一个空的列表layers用于存储残差块的层
        layers.append(block(self.inplanes, planes, stride, downsample))  #将第一个残差块的层添加到layers中，该残差快的输入通道数为self。inplanes，输出通道数为planes，步幅为stride，残差便为downsample
        self.inplanes = planes * block.expansion  #更新 self.inplanes 为 planes * block.expansion
        '''
        通过循环，将剩余的blocks-1个残差块的层添加到layers中。这些残差快的输入通道数为self.inplanes，输出通道数为planes，步幅为1，残差边为None
        '''
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)  #使用nn.Sequetial将layers中的层组合成一个顺序结构，并返回该结构

    '''
    神经网络的前向传播函数，它将输入数据x通过网络的各个层进行计算，并返回最后的输出。
    1、输入数据通过第一个卷积层self.conv1进行卷积操做
    2、卷积结果通过批归一化层self.bn1进行归一化操做
    3、归一化结果通过ReLU激活函数self.relu进行激活操做
    4、激活结果通过最大池化层self.maxpool进行池化操作
    5、池化结果通过网络的第一个残差块self.layer1进行残差连接操作
    6、残差块的输出通过输出网络的第二个残差块self.layer2进行残差连接操作
    7、残差块的输出通过输出网络的第三个残差块self.layer3进行残差连接操作
    8、残差块的输出通过网络的第四个残差块self.layer4进行残差连接操作
    9、最后一个残差块的输出通过平均池化层self.avgpool进行池化操做
    10、池化结果通过展平操做x.view(x.size(0),-1)操做转换为一维向量
    11、一维向量通过全连接层self.fc进行线性变换操做
    12、最终输出的结果x是网络的预测结果
    '''
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x



'''
定义了一个ResNet50模型，如果pretrained参数为True，则加载预训练模型的权重。函数返回的是特征提取部分和分类部分的模型。特征提取部分包括几个卷积层和残差模块，用于从输入图像中提取特征。分类部分包括最后一个残差模块和平均池化层，用于将提取到的特征进行分类
'''
def resnet50(pretrained = False):
    model = ResNet(Bottleneck, [3, 4, 6, 3])
    if pretrained:
        state_dict = load_state_dict_from_url("https://download.pytorch.org/models/resnet50-19c8e357.pth", model_dir="./model_data")
        model.load_state_dict(state_dict)
    #----------------------------------------------------------------------------#
    #   获取特征提取部分，从conv1到model.layer3，最终获得一个38,38,1024的特征层
    #----------------------------------------------------------------------------#
    features = list([model.conv1, model.bn1, model.relu, model.maxpool, model.layer1, model.layer2, model.layer3])
    #----------------------------------------------------------------------------#
    #   获取分类部分，从model.layer4到model.avgpool
    #----------------------------------------------------------------------------#
    classifier = list([model.layer4, model.avgpool])
    
    features = nn.Sequential(*features)
    classifier = nn.Sequential(*classifier)
    return features, classifier
