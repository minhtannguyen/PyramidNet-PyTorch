import torch
import torch.nn as nn
import torch.nn.functional as F

import math
#from math import round
import torch.utils.model_zoo as model_zoo

from shake_drop_function import get_alpha_beta_bdrop, shake_function


def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

class NRelu(nn.Module):
    """
    -max(-x,0)
    Parameters
    ----------
    Input shape: (N, C, W, H)
    Output shape: (N, C * W * H)
    """
    def __init__(self, inplace):
        super(NRelu, self).__init__()
        self.inplace = inplace

    def forward(self, x):
        return -F.relu(-x, inplace=self.inplace)

class BasicBlock(nn.Module):
    outchannel_ratio = 1

    def __init__(self, inplanes, planes, droprate, shake_config, stride=1, downsample=None, same_input=False):
        super(BasicBlock, self).__init__()
        
        self.shake_config = shake_config
        self.bn1 = nn.BatchNorm2d(inplanes)
        if not self.same_input:
            self.bn1min = nn.BatchNorm2d(inplanes)
        self.conv1 = conv3x3(inplanes, planes, stride)        
        
        self.bn2 = nn.BatchNorm2d(planes)
        if not self.same_input:
            self.bn2min = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes)
        
        self.bn3 = nn.BatchNorm2d(planes)
        self.bn3min = nn.BatchNorm2d(planes)
        
        self.relu = nn.ReLU(inplace=False)
        self.nrelu = NRelu(inplace=False)
        self.downsample = downsample
        self.stride = stride
        self.droprate = droprate

    def forward(self, x):
        xmax = self.bn1(x[0])
        xmax = self.conv1(xmax)
        xmax = self.bn2(xmax)
        xmax = self.relu(xmax)
        xmax = self.conv2(xmax)
        xmax = self.bn3(xmax)
        
        if not self.same_input:
            xmin = self.bn1min(x[1])
        else:
            xmin = self.bn1(x[1])
        xmin = self.conv1(xmin)
        if not self.same_input:
            xmin = self.bn2min(xmin)
        else:
            xmin = self.bn2(xmin)
        xmin = self.nrelu(xmin)
        xmin = self.conv2(xmin)
        xmin = self.bn3min(xmin)
        
        if self.training:
            shake_config = self.shake_config
        else:
            shake_config = (False, False, False)

        alpha, beta, bdrop = get_alpha_beta_bdrop(x[0].size(0), self.droprate, shake_config, x[0].is_cuda)
        alpha_min, beta_min, bdropmin = get_alpha_beta_bdrop(x[1].size(0), self.droprate, shake_config, x[1].is_cuda)
        ymax = shake_function(xmax, alpha, beta, bdrop)
        ymin = shake_function(xmin, alpha_min, beta_min, bdropmin)
       
        if self.downsample is not None:
            shortcutmax = self.downsample(x[0])
            shortcutmin = self.downsample(x[1])
            featuremap_size = shortcutmax.size()[2:4]
        else:
            shortcutmax = x[0]
            shortcutmin = x[1]
            featuremap_size = ymax.size()[2:4]

        batch_size = ymax.size()[0]
        residual_channel = ymax.size()[1]
        shortcut_channel = shortcutmax.size()[1]

        if residual_channel != shortcut_channel:
            padding = torch.autograd.Variable(torch.cuda.FloatTensor(batch_size, residual_channel - shortcut_channel, featuremap_size[0], featuremap_size[1]).fill_(0)) 
            ymax += torch.cat((shortcutmax, padding), 1)
            ymin += torch.cat((shortcutmin, padding), 1)
        else:
            ymax += shortcutmax 
            ymin += shortcutmin 

        return ymax, ymin


class Bottleneck(nn.Module):
    outchannel_ratio = 4

    def __init__(self, inplanes, planes, droprate, shake_config, stride=1, downsample=None, same_input=False):
        super(Bottleneck, self).__init__()
        
        self.shake_config = shake_config
        self.same_input = same_input
        
        self.bn1 = nn.BatchNorm2d(inplanes)
        if not self.same_input:
            self.bn1min = nn.BatchNorm2d(inplanes)
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        
        self.bn2 = nn.BatchNorm2d(planes)
        if not self.same_input:
            self.bn2min = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, (planes*1), kernel_size=3, stride=stride,
                               padding=1, bias=False)
        
        self.bn3 = nn.BatchNorm2d((planes*1))
        self.bn3min = nn.BatchNorm2d((planes*1))
        self.conv3 = nn.Conv2d((planes*1), planes * Bottleneck.outchannel_ratio, kernel_size=1, bias=False)
        
        self.bn4 = nn.BatchNorm2d(planes * Bottleneck.outchannel_ratio)
        self.bn4min = nn.BatchNorm2d(planes * Bottleneck.outchannel_ratio)
        
        self.relu = nn.ReLU(inplace=False)
        self.nrelu = NRelu(inplace=False)
        self.downsample = downsample
        self.stride = stride
        self.droprate = droprate

    def forward(self, x):
        xmax = self.bn1(x[0])
        xmax = self.conv1(xmax)
        xmax = self.bn2(xmax)
        xmax = self.relu(xmax)
        xmax = self.conv2(xmax)
        xmax = self.bn3(xmax)
        xmax = self.relu(xmax)
        xmax = self.conv3(xmax)
        xmax = self.bn4(xmax)
        
        if not self.same_input:
            xmin = self.bn1min(x[1])
        else:
            xmin = self.bn1(x[1])
        xmin = self.conv1(xmin)
        if not self.same_input:
            xmin = self.bn2min(xmin)
        else:
            xmin = self.bn2(xmin)
        xmin = self.nrelu(xmin)
        xmin = self.conv2(xmin)
        xmin = self.bn3min(xmin)
        xmin = self.nrelu(xmin)
        xmin = self.conv3(xmin)
        xmin = self.bn4min(xmin)
        
        if self.training:
            shake_config = self.shake_config
        else:
            shake_config = (False, False, False)
            
        alpha, beta, bdrop = get_alpha_beta_bdrop(x[0].size(0), self.droprate, shake_config, x[0].is_cuda)
        alpha_min, beta_min, bdropmin = get_alpha_beta_bdrop(x[1].size(0), self.droprate, shake_config, x[1].is_cuda)
        ymax = shake_function(xmax, alpha, beta, bdrop)
        ymin = shake_function(xmin, alpha_min, beta_min, bdropmin)

        if self.downsample is not None:
            shortcutmax = self.downsample(x[0])
            shortcutmin = self.downsample(x[1])
            featuremap_size = shortcutmax.size()[2:4]
        else:
            shortcutmax = x[0]
            shortcutmin = x[1]
            featuremap_size = ymax.size()[2:4]

        batch_size = ymax.size()[0]
        residual_channel = ymax.size()[1]
        shortcut_channel = shortcutmax.size()[1]

        if residual_channel != shortcut_channel:
            padding = torch.autograd.Variable(torch.cuda.FloatTensor(batch_size, residual_channel - shortcut_channel, featuremap_size[0], featuremap_size[1]).fill_(0)) 
            ymax += torch.cat((shortcutmax, padding), 1)
            ymin += torch.cat((shortcutmin, padding), 1)
        else:
            ymax += shortcutmax 
            ymin += shortcutmin 

        return ymax, ymin

class PyramidNet(nn.Module):
        
    def __init__(self, config):
        super(PyramidNet, self).__init__()
        dataset = config['dataset']
        depth = config['depth']
        alpha = config['alpha']
        num_classes = config['num_classes']
        bottleneck = config['bottleneck']
        self.shake_config = (config['shake_forward'], config['shake_backward'],
                             config['shake_image'])
        self.dataset = dataset
        if self.dataset.startswith('cifar'):
            self.res_layer = 3
            self.inplanes = 16
            if bottleneck == True:
                n = int((depth - 2) / 9)
                block = Bottleneck
            else:
                n = int((depth - 2) / 6)
                block = BasicBlock

            self.addrate = alpha / (3*n*1.0)

            self.input_featuremap_dim = self.inplanes
            self.conv1 = nn.Conv2d(3, self.input_featuremap_dim, kernel_size=3, stride=1, padding=1, bias=False)
            self.bn1 = nn.BatchNorm2d(self.input_featuremap_dim)

            self.featuremap_dim = self.input_featuremap_dim 
            self.layer1 = self.pyramidal_make_layer(block, n, block_no=1, same_input=True)
            self.layer2 = self.pyramidal_make_layer(block, n, block_no=2, stride=2)
            self.layer3 = self.pyramidal_make_layer(block, n, block_no=3, stride=2)

            self.final_featuremap_dim = self.input_featuremap_dim
            self.bn_final= nn.BatchNorm2d(self.final_featuremap_dim)
            self.bn_finalmin= nn.BatchNorm2d(self.final_featuremap_dim)
            self.relu_final = nn.ReLU(inplace=True)
            self.nrelu_final = NRelu(inplace=False)
            self.avgpool = nn.AvgPool2d(8)
            self.fc = nn.Linear(self.final_featuremap_dim, num_classes)

        elif dataset == 'imagenet':
            self.res_layer = 4
            blocks ={18: BasicBlock, 34: BasicBlock, 50: Bottleneck, 101: Bottleneck, 152: Bottleneck, 200: Bottleneck}
            layers ={18: [2, 2, 2, 2], 34: [3, 4, 6, 3], 50: [3, 4, 6, 3], 101: [3, 4, 23, 3], 152: [3, 8, 36, 3], 200: [3, 24, 36, 3]}

            if layers.get(depth) is None:
                if bottleneck == True:
                    blocks[depth] = Bottleneck
                    temp_cfg = int((depth-2)/12)
                else:
                    blocks[depth] = BasicBlock
                    temp_cfg = int((depth-2)/8)

                layers[depth]= [temp_cfg, temp_cfg, temp_cfg, temp_cfg]
                print('=> the layer configuration for each stage is set to', layers[depth])

            self.inplanes = 64            
            self.addrate = alpha / (sum(layers[depth])*1.0)

            self.input_featuremap_dim = self.inplanes
            self.conv1 = nn.Conv2d(3, self.input_featuremap_dim, kernel_size=7, stride=2, padding=3, bias=False)
            self.bn1 = nn.BatchNorm2d(self.input_featuremap_dim)
            self.relu = nn.ReLU(inplace=True)
            self.nrelu = NRelu(inplace=False)
            self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

            self.featuremap_dim = self.input_featuremap_dim 
            self.layer1 = self.pyramidal_make_layer(blocks[depth], layers[depth][0], block_no=1, same_input=True)
            self.layer2 = self.pyramidal_make_layer(blocks[depth], layers[depth][1], block_no=2, stride=2)
            self.layer3 = self.pyramidal_make_layer(blocks[depth], layers[depth][2], block_no=3, stride=2)
            self.layer4 = self.pyramidal_make_layer(blocks[depth], layers[depth][3], block_no=4, stride=2)

            self.final_featuremap_dim = self.input_featuremap_dim
            self.bn_final= nn.BatchNorm2d(self.final_featuremap_dim)
            self.bn_finalmin= nn.BatchNorm2d(self.final_featuremap_dim)
            self.relu_final = nn.ReLU(inplace=True)
            self.nrelu_final = NRelu(inplace=False)
            self.avgpool = nn.AvgPool2d(7) 
            self.fc = nn.Linear(self.final_featuremap_dim, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def pyramidal_make_layer(self, block, block_depth, block_no, stride=1, same_input=False):
        downsample = None
        if stride != 1: # or self.inplanes != int(round(featuremap_dim_1st)) * block.outchannel_ratio:
            downsample = nn.AvgPool2d((2,2), stride = (2, 2), ceil_mode=True)
        
        layers = []
        self.featuremap_dim = self.featuremap_dim + self.addrate
        droprate_val = 1.0 - (1.0 - 0.5) * (1+(block_no-1)*block_depth)/(block_depth*self.res_layer)
        layers.append(block(self.input_featuremap_dim, int(round(self.featuremap_dim)), droprate=droprate_val, shake_config=self.shake_config, stride=stride, downsample=downsample, same_input=same_input))
        for i in range(1, block_depth):
            temp_featuremap_dim = self.featuremap_dim + self.addrate
            droprate_val = 1.0 - (1.0 - 0.5) * (i+1+(block_no-1)*block_depth)/(block_depth*self.res_layer)
            layers.append(block(int(round(self.featuremap_dim)) * block.outchannel_ratio, int(round(temp_featuremap_dim)), droprate=droprate_val, shake_config=self.shake_config, stride=1))
            self.featuremap_dim  = temp_featuremap_dim
        self.input_featuremap_dim = int(round(self.featuremap_dim)) * block.outchannel_ratio

        return nn.Sequential(*layers)

    def forward(self, x):
        if self.dataset == 'cifar10' or self.dataset == 'cifar100':
            x = self.conv1(x)
            x = self.bn1(x)
            
            xmax, xmin = self.layer1([x,x])
            xmax, xmin = self.layer2([xmax, xmin])
            xmax, xmin = self.layer3([xmax, xmin])

            xmax = self.bn_final(xmax)
            xmin = self.bn_finalmin(xmin)
            xmax = self.relu_final(xmax)
            xmin = self.nrelu_final(xmin)
            xmax = self.avgpool(xmax)
            xmin = self.avgpool(xmin)
            xmax = xmax.view(xmax.size(0), -1)
            xmin = xmin.view(xmin.size(0), -1)
            xmax = self.fc(xmax)
            xmin = self.fc(xmin)

        elif self.dataset == 'imagenet':
            x = self.conv1(x)
            x = self.bn1(x)
            xmax = self.relu(x)
            xmin = self.nrelu(x)
            xmax = self.maxpool(xmax)
            xmin = -self.maxpool(-xmin)

            xmax, xmin = self.layer1([x,x])
            xmax, xmin = self.layer2([xmax, xmin])
            xmax, xmin = self.layer3([xmax, xmin])
            xmax, xmin = self.layer4([xmax, xmin])

            xmax = self.bn_final(xmax)
            xmin = self.bn_finalmin(xmin)
            xmax = self.relu_final(xmax)
            xmin = self.nrelu_final(xmin)
            xmax = self.avgpool(xmax)
            xmin = self.avgpool(xmin)
            xmax = xmax.view(xmax.size(0), -1)
            xmin = xmin.view(xmin.size(0), -1)
            xmax = self.fc(xmax)
            xmin = self.fc(xmin)
    
        return xmax, xmin
