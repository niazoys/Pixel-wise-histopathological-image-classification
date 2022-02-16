
import torch
import torch.nn as nn
from torchvision import models

class QuadripleUp(nn.Module):
    ''' ConvTranspose2d + {Conv2d, BN, ReLU}x4 '''

    def __init__(self, in_chan, out_chan, mid_chan=None):
        super().__init__()

        self.up = nn.ConvTranspose2d(in_chan , out_chan, kernel_size=2, stride=2)
        if mid_chan == None:
            mid_chan = in_chan
        self.conv = QuadripleConv(mid_chan, out_chan)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        x = torch.cat([x1, x2], dim=1)
        return self.conv(x)

class DoubleUp(nn.Module):
    ''' ConvTranspose2d + {Conv2d, BN, ReLU}x2 '''

    def __init__(self, in_chan, out_chan, mid_chan=None):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_chan, out_chan, kernel_size=2, stride=2)
        if mid_chan == None:
            mid_chan = in_chan
        self.conv = DoubleConv(mid_chan, out_chan)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        x = torch.cat([x1, x2], dim=1)
        return self.conv(x)

class TripleUp(nn.Module):
    ''' ConvTranspose2d + {Conv2d, BN, ReLU}x3 '''

    def __init__(self, in_chan, out_chan, mid_chan=None):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_chan, out_chan, kernel_size=2, stride=2)
        if mid_chan == None:
            mid_chan = in_chan
        self.conv = TripleConv(mid_chan, out_chan)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        x = torch.cat([x1, x2], dim=1)
        return self.conv(x)

class DoubleConv(nn.Module):
    ''' {Conv2d, BN, ReLU}x2 '''

    def __init__(self, in_chan, out_chan):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_chan, out_chan, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_chan),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_chan, out_chan, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_chan),
            nn.ReLU(inplace=True))

    def forward(self, x):
        return self.double_conv(x)

class TripleConv(nn.Module):
    ''' {Conv2d, BN, ReLU}x3 '''

    def __init__(self, in_chan, out_chan):
        super().__init__()
        self.triple_conv = nn.Sequential(
            nn.Conv2d(in_chan, out_chan, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_chan),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_chan, out_chan, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_chan),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_chan, out_chan, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_chan),
            nn.ReLU(inplace=True))

    def forward(self, x):
        return self.triple_conv(x)



class QuadripleConv(nn.Module):
    ''' {Conv2d, BN, ReLU}x4 '''

    def __init__(self, in_chan, out_chan):
        super().__init__()
        self.quadriple_conv = nn.Sequential(
            nn.Conv2d(in_chan, out_chan, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_chan),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_chan, out_chan, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_chan),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_chan, out_chan, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_chan),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_chan, out_chan, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_chan),
            nn.ReLU(inplace=True))

    def forward(self, x):
        return self.quadriple_conv(x)

class OutConv(nn.Module):
    ''' Conv2d '''

    def __init__(self, in_chan, out_chan):
        super().__init__()
        self.conv = nn.Conv2d(in_chan, out_chan, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

class uNet16(nn.Module):
    
    def __init__(self, n_classes, pretrained=True):    
        super(uNet16, self).__init__()
        self.n_channels = 3
        self.n_classes = n_classes

        self.pool = nn.MaxPool2d(2, 2)
        self.relu = nn.ReLU(inplace=True)
        self.encoder = models.vgg16_bn(pretrained=pretrained).features
        self.enblock1 = nn.Sequential(self.encoder[0], self.encoder[1], self.relu, self.encoder[3], self.encoder[4], self.relu)
        self.enblock2 = nn.Sequential(self.pool, self.encoder[7], self.encoder[8], self.relu, self.encoder[10], self.encoder[11], self.relu)
        self.enblock3 = nn.Sequential(self.pool, self.encoder[14], self.encoder[15], self.relu, self.encoder[17], self.encoder[18], self.relu, self.encoder[20], self.encoder[21], self.relu)
        self.enblock4 = nn.Sequential(self.pool, self.encoder[24], self.encoder[25], self.relu, self.encoder[27], self.encoder[28], self.relu, self.encoder[30], self.encoder[31], self.relu)
        self.center = nn.Sequential(self.pool, self.encoder[34], self.encoder[35], self.relu, self.encoder[37], self.encoder[38], self.relu, self.encoder[40], self.encoder[41], self.relu)
        self.deblock1 = TripleUp(512,512,1024)
        self.deblock2 = TripleUp(512,256)
        self.deblock3 = DoubleUp(256,128)
        self.deblock4 = DoubleUp(128,64)
        self.outc = OutConv(64, n_classes)

    def forward(self, x):
        x1 = self.enblock1(x)
        x2 = self.enblock2(x1)
        x3 = self.enblock3(x2)
        x4 = self.enblock4(x3)
        x5 = self.center(x4)
        x = self.deblock1(x5, x4)
        x = self.deblock2(x, x3)
        x = self.deblock3(x, x2)
        x = self.deblock4(x, x1)
        logits = self.outc(x)
        return logits

class uNet19(nn.Module):
    
    def __init__(self, n_classes, pretrained=True):    
        super(uNet19, self).__init__()
        self.n_channels = 3
        self.n_classes = n_classes

        self.pool = nn.MaxPool2d(2, 2)
        self.relu = nn.ReLU(inplace=True)
        self.encoder = models.vgg19_bn(pretrained=pretrained).features
        self.enblock1 = nn.Sequential(self.encoder[0], self.encoder[1], self.relu, self.encoder[3], self.encoder[4], self.relu)
        self.enblock2 = nn.Sequential(self.pool, self.encoder[7], self.encoder[8], self.relu, self.encoder[10], self.encoder[11], self.relu)
        self.enblock3 = nn.Sequential(self.pool, self.encoder[14], self.encoder[15], self.relu, self.encoder[17], self.encoder[18], self.relu, self.encoder[20], self.encoder[21], self.relu, self.encoder[23], self.encoder[24], self.relu)
        self.enblock4 = nn.Sequential(self.pool, self.encoder[27], self.encoder[28], self.relu, self.encoder[30], self.encoder[31], self.relu, self.encoder[33], self.encoder[34], self.relu, self.encoder[36], self.encoder[37], self.relu)
        self.center = nn.Sequential(self.pool, self.encoder[40], self.encoder[41], self.relu, self.encoder[43], self.encoder[44], self.relu, self.encoder[46], self.encoder[47], self.relu, self.encoder[49], self.encoder[50], self.relu)
        self.deblock1 = QuadripleUp(512,512,1024)
        self.deblock2 = QuadripleUp(512,256)
        self.deblock3 = DoubleUp(256,128)
        self.deblock4 = DoubleUp(128,64)
        self.outc = OutConv(64, n_classes)

    def forward(self, x):
        x1 = self.enblock1(x)
        x2 = self.enblock2(x1)
        x3 = self.enblock3(x2)
        x4 = self.enblock4(x3)
        x5 = self.center(x4)
        x = self.deblock1(x5, x4)
        x = self.deblock2(x, x3)
        x = self.deblock3(x, x2)
        x = self.deblock4(x, x1)
        logits = self.outc(x)
        return logits
    