import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torchvision.ops import nms
from utils.anchors import _enumerate_shifted_anchor, generate_anchor_base
from utils.utils_bbox import loc2bbox


class ProposalCreator():
    def __init__(
        self, 
        mode,
        nms_iou = 0.7,
        n_train_pre_nms = 12000,  #保留的建议框的个数（在非极大抑制之前）
        n_train_post_nms = 600,   #最后返回建议框的数量（在非极大抑制之后）
        n_test_pre_nms = 3000,
        n_test_post_nms = 300,
        min_size = 16
    
    ):
        self.mode = mode        #设置预测还是训练
        self.nms_iou = nms_iou        #建议框非极大抑制的iou大小
        self.n_train_pre_nms = n_train_pre_nms        #训练用到的建议框数量（在非极大抑制之前）
        self.n_train_post_nms = n_train_post_nms     #训练用到的建议框数量（在非极大抑制之后）
        self.n_test_pre_nms = n_test_pre_nms        #预测用到的建议框数量（在非极大抑制之前）
        self.n_test_post_nms = n_test_post_nms     #预测用到的建议框数量（在非极大抑制之后）
        self.min_size = min_size                  #最小边框的大小

    def __call__(self, loc, score, anchor, img_size, scale=1.):
        if self.mode == "training":
            n_pre_nms = self.n_train_pre_nms
            n_post_nms = self.n_train_post_nms
        else:
            n_pre_nms = self.n_test_pre_nms
            n_post_nms = self.n_test_post_nms

        anchor = torch.from_numpy(anchor).type_as(loc)        #将先验框转换成tensor
        #将RPN网络预测结果转化成建议框
        roi = loc2bbox(anchor, loc)
        #利用slice进行分割，防止建议狂超出图像边缘
        roi[:, [0, 2]] = torch.clamp(roi[:, [0, 2]], min = 0, max = img_size[1])
        roi[:, [1, 3]] = torch.clamp(roi[:, [1, 3]], min = 0, max = img_size[0])

        # 建议框的宽高的最小值不可以小于16（自己设置）
        min_size = self.min_size * scale
        #防止建议框过小
        keep = torch.where(((roi[:, 2] - roi[:, 0]) >= min_size) & ((roi[:, 3] - roi[:, 1]) >= min_size))[0]
        roi = roi[keep, :]        #将对应的建议框保留下来
        score = score[keep]

        # 根据得分进行排序，取出建议框
        order = torch.argsort(score, descending=True)
        '''
        对变量进行切片操做，取出order中前n_pre_nms个元素的索引，并将这些索引作用于变量roi、score
        '''
        if n_pre_nms > 0:
            order = order[:n_pre_nms]
        roi = roi[order, :]
        score = score[order]

        '''
        对建议框进行非极大抑制,对每一个种类的建议框进行限制
        使用官方的非极大抑制会快非常多
        
        最后返回建议框
        '''
        keep = nms(roi, score, self.nms_iou)
        if len(keep) < n_post_nms:            #保证最后的到的建议框的数量要小于n_post_nms
            index_extra = np.random.choice(range(len(keep)), size=(n_post_nms - len(keep)), replace=True)
            keep = torch.cat([keep, keep[index_extra]])
        keep = keep[:n_post_nms]
        roi = roi[keep]
        return roi

'''
用于生成候选区域（Region Proposal Network）
'''
class RegionProposalNetwork(nn.Module):
    def __init__(
        self, 
        in_channels = 512,
        mid_channels = 512,
        ratios = [0.5, 1, 2],
        anchor_scales = [8, 16, 32],
        feat_stride = 16,
        mode = "training",
    ):
        super(RegionProposalNetwork, self).__init__()
        self.anchor_base = generate_anchor_base(anchor_scales = anchor_scales, ratios = ratios)   #生成基础先验框，shape为[9, 4]
        n_anchor = self.anchor_base.shape[0]

        self.conv1 = nn.Conv2d(in_channels, mid_channels, 3, 1, 1)        #先进行一个3x3的卷积，可理解为特征整合
        self.score = nn.Conv2d(mid_channels, n_anchor * 2, 1, 1, 0)        #分类预测先验框内部是否包含物体
        self.loc = nn.Conv2d(mid_channels, n_anchor * 4, 1, 1, 0)        #回归预测对先验框进行调整
        self.feat_stride = feat_stride                              #特征点间距步长
        self.proposal_layer = ProposalCreator(mode)        #用于对建议框解码并进行非极大抑制

        '''
        对FPN(特征金字塔网络)的网络部分的权重进行初始化
        将self.conv1、self.score、self.loc等层的权重参数初始化为服从均值为0、标准差为0.01的正态分布
        '''
        normal_init(self.conv1, 0, 0.01)        #对FPN的网络部分进行权值初始化
        normal_init(self.score, 0, 0.01)
        normal_init(self.loc, 0, 0.01)


    '''
    前向传播函数，用于完成目标检测中的区域提议网络（Region Proposal Network，RPN）的计算过程
    通过输入特征图x，利用RPN网络生成区域提议，并返回生成的区域提议，对应的样本索引，先验框的坐标调整量和分类得分，这些输出将用于后续的目标检测任务
    '''
    def forward(self, x, img_size, scale=1.):  #x：输入的特征图，shape为（n，c，h，w）n为样本数；c为通道数，h为高度，w为宽度； img_size输入图像的大小，用于生成的区域提议的坐标； scale=1.缩放因子，用于调整生成的区域提议的大小
        n, _, h, w = x.shape

        '''
        建议框
        '''
        #对共享特征层进行一个3x3的卷积，可理解为特征整合
        x = F.relu(self.conv1(x))

        # 分类预测    先验框内部是否包含物体
        rpn_scores = self.score(x)
        rpn_scores = rpn_scores.permute(0, 2, 3, 1).contiguous().view(n, -1, 2)

        # 回归预测     对先验框进行调整
        rpn_locs = self.loc(x)
        rpn_locs = rpn_locs.permute(0, 2, 3, 1).contiguous().view(n, -1, 4)

        #softmax损失函数
        #进行softmax概率计算，每个先验框只有两个判别结果
        #内部包含物体或者内部不包含物体，rpn_softmax_scores[:, :, 1]的内容为包含物体的概率
        rpn_softmax_scores = F.softmax(rpn_scores, dim=-1)
        rpn_fg_scores = rpn_softmax_scores[:, :, 1].contiguous()
        rpn_fg_scores = rpn_fg_scores.view(n, -1)

        '''
        生成先验框，此时获得的anchor是布满网格点的，当输入图片为600,600,3的时候，shape为(12996, 4)
        利用回归分类对先验框进行调整 proposal_layer
        '''
        anchor = _enumerate_shifted_anchor(np.array(self.anchor_base), self.feat_stride, h, w)
        rois = list()
        roi_indices = list()
        for i in range(n):
            roi = self.proposal_layer(rpn_locs[i], rpn_fg_scores[i], anchor, img_size, scale = scale)
            batch_index = i * torch.ones((len(roi),))
            rois.append(roi.unsqueeze(0))
            roi_indices.append(batch_index.unsqueeze(0))

        rois = torch.cat(rois, dim=0).type_as(x)
        roi_indices = torch.cat(roi_indices, dim=0).type_as(x)
        anchor = torch.from_numpy(anchor).unsqueeze(0).float().to(x.device)
        
        return rpn_locs, rpn_scores, rois, roi_indices, anchor

def normal_init(m, mean, stddev, truncated=False):
    if truncated:
        m.weight.data.normal_().fmod_(2).mul_(stddev).add_(mean)  # not a perfect approximation
    else:
        m.weight.data.normal_(mean, stddev)
        m.bias.data.zero_()
