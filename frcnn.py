import torch.nn as nn
from nets.classifier import Resnet50RoIHead, VGG16RoIHead
from nets.resnet50 import resnet50
from nets.rpn import RegionProposalNetwork
from nets.vgg16 import decom_vgg16

'''

'''
class FasterRCNN(nn.Module):
    def __init__(self,  num_classes,  
                    mode = "training",
                    # 特征步长。在目标检测任务中，特征图的大小通常会比输入图像的大小小很多。为了将特征图上的位置映射回输入图像上的位置，需要知道特征图相对于输入图像的缩放倍数。这个缩放倍数就是特征步长。
                    #特征步长表示输入图像上的一个像素在特征图上的尺寸。例如，如果特征步长为16，则输入图像上的一个像素在特征图上的尺寸为16x16
                    #在 Faster R-CNN 中，特征步长通常用于计算锚框的位置。锚框是一种用于生成候选框的固定大小和宽高比的框。特征步长可以决定锚框在特征图上的位置和尺度。
                    feat_stride = 16,
                    anchor_scales = [8, 16, 32],        #列表，用于指定生成锚框的尺寸，每个元素表示一个尺度的大小
                    ratios = [0.5, 1, 2],               #列表，用于指定生成锚框的宽高比，每个元素表示一个宽高比
                    backbone = 'vgg',                   #字符串，用于指定使用的特征提取网络的类型
                    pretrained = False):
        super(FasterRCNN, self).__init__()
        self.feat_stride = feat_stride
        #---------------------------------#
        #   一共存在两个主干
        #   vgg和resnet50
        #---------------------------------#
        if backbone == 'vgg':
            self.extractor, classifier = decom_vgg16(pretrained)
            #构建建议框网络
            self.rpn = RegionProposalNetwork(
                512, 512,
                ratios = ratios,
                anchor_scales = anchor_scales,
                feat_stride = self.feat_stride,
                mode = mode
            )
            #---------------------------------#
            #   构建分类器网络
            #---------------------------------#
            self.head = VGG16RoIHead(
                n_class = num_classes + 1,
                roi_size  = 7,
                spatial_scale = 1,
                classifier = classifier
            )
        elif backbone == 'resnet50':
            self.extractor, classifier = resnet50(pretrained)
            #---------------------------------#
            #   构建classifier网络
            #---------------------------------#
            self.rpn = RegionProposalNetwork(
                1024, 512,
                ratios = ratios,
                anchor_scales = anchor_scales,
                feat_stride = self.feat_stride,
                mode = mode
            )
            #---------------------------------#
            #   构建classifier网络
            #---------------------------------#
            self.head = Resnet50RoIHead(
                n_class = num_classes + 1,
                roi_size = 14,
                spatial_scale = 1,
                classifier = classifier
            )


    '''
    定义Faster R-CNN模型的前向传播函数forward，根据不同的mode参数，函数会执行不同的操做
    
    mode == "forward":首先计算输入图片的大小img_size，然后利用主干网络提取特征base_feature。通过区域生成网络Region proposal Network，RPN获得建议框rois和对应的索引roi_indices。利用头网络（RoI Head）对提取的特征进行分类和回归，
                      得到分类结果 roi_cls_locs 和得分 roi_scores，并返回这些结果以及建议框和索引
    
    mode == "extractor"：仅执行特征提取的过程。函数直接利用主干网络提取特征 base_feature，并返回它
    
    mode == "rpn":仅执行区域生成网络的过程。函数接受主干网络提取的特征 base_feature 和输入图片的大小 img_size，
                  然后通过 RPN 获得建议框的位置偏移 rpn_locs、得分 rpn_scores、建议框 rois、索引 roi_indices，以及锚框 anchor，并返回这些结果
    
    mode == "head":仅执行头网络的过程。函数接受主干网络提取的特征 base_feature、建议框 rois、索引 roi_indices，以及输入图片的大小 img_size，然后通过头网络获得分类结果 roi_cls_locs 和得分 roi_scores，并返回这些结果
    '''
    def forward(self, x, scale=1., mode="forward"):
        if mode == "forward":
            img_size = x.shape[2:]            #   计算输入图片的大小

            '''
            利用主干网络提取特征
            '''

            base_feature = self.extractor.forward(x)
            '''
            获得建议框
            '''
            _, _, rois, roi_indices, _ = self.rpn.forward(base_feature, img_size, scale)

            roi_cls_locs, roi_scores = self.head.forward(base_feature, rois, roi_indices, img_size)            #获得classifier的分类结果和回归结果
            return roi_cls_locs, roi_scores, rois, roi_indices
        elif mode == "extractor":
            #---------------------------------#
            #   利用主干网络提取特征
            #---------------------------------#
            base_feature = self.extractor.forward(x)
            return base_feature
        elif mode == "rpn":
            base_feature, img_size = x
            #---------------------------------#
            #   获得建议框
            #---------------------------------#
            rpn_locs, rpn_scores, rois, roi_indices, anchor = self.rpn.forward(base_feature, img_size, scale)
            return rpn_locs, rpn_scores, rois, roi_indices, anchor
        elif mode == "head":
            base_feature, rois, roi_indices, img_size = x
            #---------------------------------------#
            #   获得classifier的分类结果和回归结果
            #---------------------------------------#
            roi_cls_locs, roi_scores    = self.head.forward(base_feature, rois, roi_indices, img_size)
            return roi_cls_locs, roi_scores


    '''
    将模型中的BatchNormalization层设置为评估模式
    在深度学习模型训练过程中BatchNormalization层通常会根据每个小批量的输入数据计算并更新均值和方差，以便对输入数据进行归一化，然而，在模型的评估阶段，我们希望使用固定的均值和方差来进行归一化，以保持一致性。
    通过调用m.eval()，可以将BatchNormalization层设置为评估模式，在评估模式下，BatchNormalization层不会更细均值和方差，而是使用之前训练阶段计算的固定值进行归一化，这样可以确保在评估模型时输出的结果于训练模型时保持一致。
    freeze_bn 方法遍历模型的所有模块，如果某个模块是 Batch Normalization 层（nn.BatchNorm2d 类型），则将其调用 eval() 方法，将其设置为评估模式。
    在调用freeze_bn 方法后，模型中的所有Batch Normalization层都会被设置为评估模式
    '''
    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()
