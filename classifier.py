import warnings

import torch
from torch import nn
from torchvision.ops import RoIPool

warnings.filterwarnings("ignore")

class VGG16RoIHead(nn.Module):
    def __init__(self, n_class, roi_size, spatial_scale, classifier):  #初始化函数（n_class类别数量, roi_sizeRol池化后的大小, spatial_scale空间缩放因子, classifier用于特征提取的分类器 ）
        super(VGG16RoIHead, self).__init__()
        self.classifier = classifier

        # nn.Linear创建全连接层，对ROIPooling后的的结果进行回归预测
        self.cls_loc = nn.Linear(4096, n_class * 4)
        # 对ROIPooling后的的结果进行分类
        self.score = nn.Linear(4096, n_class)

        # 权值初始化
        normal_init(self.cls_loc, 0, 0.001)
        normal_init(self.score, 0, 0.01)

        # 创建RolPool层，用于对特征图进行Rol池化操作
        self.roi = RoIPool((roi_size, roi_size), spatial_scale)
        
    def forward(self, x, rois, roi_indices, img_size):#前向传播的过程（x输入张量, rois感兴趣区域, roi_indicesRol的索引, img_size图像大小）
        n, _, _, _ = x.shape
        if x.is_cuda:
            roi_indices = roi_indices.cuda()   #建议框的序号
            rois = rois.cuda()   #建议框
        # print('Base_layers:',x.size())
        # print('Base_layers:', roi_indices.size())
        # print('Base_layers:', rois.size())
        rois = torch.flatten(rois, 0, 1)   #将rois、roi_indices张量平铺为一维张量
        roi_indices = torch.flatten(roi_indices, 0, 1)

        # 根据图像大小将感兴趣区域的坐标映射到Rol特征图上
        rois_feature_map = torch.zeros_like(rois)
        rois_feature_map[:, [0, 2]] = rois[:, [0, 2]] / img_size[1] * x.size()[3]
        rois_feature_map[:, [1, 3]] = rois[:, [1, 3]] / img_size[0] * x.size()[2]

        # 拼接roi_indices、rois_feature_map（建议框序号核建议框内容进行堆叠）
        indices_and_rois = torch.cat([roi_indices[:, None], rois_feature_map], dim = 1)
        # 利用建议框对公用特征层进行截取。对x, indices_and_rois进行池化操做
        pool = self.roi(x, indices_and_rois)
        # 利用classifier网络进行特征提取
        pool = pool.view(pool.size(0), -1)
        # 当输入为一张图片的时候，这里获得的f7的shape为[300, 4096]
        fc7 = self.classifier(pool)

        #对fc7张量进行预测
        roi_cls_locs = self.cls_loc(fc7)
        roi_scores = self.score(fc7)

        #调整roi_cls_locs、roi_scores为(n, -1, roi_cls_locs.size(1))形状的张量
        roi_cls_locs = roi_cls_locs.view(n, -1, roi_cls_locs.size(1))
        roi_scores = roi_scores.view(n, -1, roi_scores.size(1))
        return roi_cls_locs, roi_scores

class Resnet50RoIHead(nn.Module):
    def __init__(self, n_class, roi_size, spatial_scale, classifier):
        super(Resnet50RoIHead, self).__init__()
        self.classifier = classifier
        self.cls_loc = nn.Linear(2048, n_class * 4)        #对ROIPooling后的的结果进行回归预测
        self.score = nn.Linear(2048, n_class)        #对ROIPooling后的的结果进行分类

        normal_init(self.cls_loc, 0, 0.001)        #权值初始化
        normal_init(self.score, 0, 0.01)

        self.roi = RoIPool((roi_size, roi_size), spatial_scale)

    def forward(self, x, rois, roi_indices, img_size):
        n, _, _, _ = x.shape
        if x.is_cuda:
            roi_indices = roi_indices.cuda()
            rois = rois.cuda()
        rois = torch.flatten(rois, 0, 1)
        roi_indices = torch.flatten(roi_indices, 0, 1)
        
        rois_feature_map = torch.zeros_like(rois)
        rois_feature_map[:, [0, 2]] = rois[:, [0, 2]] / img_size[1] * x.size()[3]
        rois_feature_map[:, [1, 3]] = rois[:, [1, 3]] / img_size[0] * x.size()[2]

        indices_and_rois = torch.cat([roi_indices[:, None], rois_feature_map], dim = 1)
        pool = self.roi(x, indices_and_rois)        #利用建议框对公用特征层进行截取
        fc7 = self.classifier(pool)        #利用classifier网络进行特征提取
        fc7 = fc7.view(fc7.size(0), -1)        #当输入为一张图片的时候，这里获得的f7的shape为[300, 2048]

        roi_cls_locs = self.cls_loc(fc7)
        roi_scores = self.score(fc7)
        roi_cls_locs = roi_cls_locs.view(n, -1, roi_cls_locs.size(1))
        roi_scores = roi_scores.view(n, -1, roi_scores.size(1))
        return roi_cls_locs, roi_scores



'''
初始化函数，用来初始化神经网络层的权重和偏置
truncated参数为True：
使用截断正态分布进行初始化；使用normal_()函数生成一个标准正态分布的随机数，再使用fmod_()函数将权重矩阵中的数值取模2，再使用mul_()函数乘以标准差stddev，最后使用add_()函数加上均值mean
truncated参数为False：
使用正态分布进行初始化。使用normal()函数生成一个均值为mean，标准差为stddev的正态分布随机数
最终偏置初始化为0
'''
def normal_init(m, mean, stddev, truncated = False):
    if truncated:
        m.weight.data.normal_().fmod_(2).mul_(stddev).add_(mean)  # not a perfect approximation
    else:
        m.weight.data.normal_(mean, stddev)
        m.bias.data.zero_()
