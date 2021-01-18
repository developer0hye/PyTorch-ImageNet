import torch
import torch.nn as nn
import torch.nn.functional as F


class Block(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 padding=0,
                 stride=1,
                 groups=1):
        super(Block, self).__init__()
        
        self.conv = nn.Conv2d(in_channels, 
                              out_channels, 
                              kernel_size, 
                              padding=kernel_size//2, 
                              stride=stride,
                              groups=groups,
                              bias=False)
        self.norm = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(self.norm(self.conv(x)))

class ConvPool(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 padding=0,
                 stride=1,
                 groups=1):
        super(ConvPool, self).__init__()
        
        self.conv = nn.Conv2d(in_channels, 
                              out_channels, 
                              kernel_size, 
                              padding=kernel_size//2, 
                              stride=stride,
                              groups=groups,
                              bias=False)
        self.norm = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        return self.norm(self.conv(x))

class LightWeightBlock(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 padding=0,
                 stride=1):
        super(LightWeightBlock, self).__init__()
        
        self.pw_conv = nn.Conv2d(in_channels, 
                              out_channels, 
                              1, 
                              padding=0, 
                              stride=stride,
                              groups=1,
                              bias=False)
        
        self.dw_conv = nn.Conv2d(out_channels, 
                              out_channels, 
                              kernel_size, 
                              padding=kernel_size//2, 
                              stride=stride,
                              groups=out_channels,
                              bias=False)

        self.norm = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(self.norm(self.dw_conv(self.pw_conv(x))))

#yolo v3 tiny backbone receptive field size: 254
#WhiteNet backbone receptive field size: 
class WhiteNet(nn.Module):
    def __init__(self, num_classes=1000):
        super(WhiteNet, self).__init__()

        self.stage_1 = nn.Sequential(Block(in_channels=3, out_channels=32, kernel_size=3, stride=2), 
                                     Block(in_channels=32, out_channels=32, kernel_size=3, stride=1)) # //2

        self.stage_2 = nn.Sequential(ConvPool(in_channels=32, out_channels=32, kernel_size=3, stride=2, groups=32),
                                     Block(in_channels=32, out_channels=64, kernel_size=3, stride=1),
                                     Block(in_channels=64, out_channels=64, kernel_size=3, stride=1)
                                     ) # //4
        
        self.stage_3 = nn.Sequential(ConvPool(in_channels=64, out_channels=64, kernel_size=3, stride=2, groups=64),
                                     Block(in_channels=64, out_channels=128, kernel_size=3, stride=1),
                                     Block(in_channels=128, out_channels=128, kernel_size=3, stride=1)
                                     ) # //8
        
        self.stage_4 = nn.Sequential(ConvPool(in_channels=128, out_channels=128, kernel_size=3, stride=2, groups=128),
                                     Block(in_channels=128, out_channels=256, kernel_size=3, stride=1),
                                     Block(in_channels=256, out_channels=256, kernel_size=3, stride=1),
                                     ) #//16
        
        self.stage_5 = nn.Sequential(ConvPool(in_channels=256, out_channels=256, kernel_size=3, stride=2, groups=256),
                                     Block(in_channels=256, out_channels=512, kernel_size=3, stride=1),
                                     Block(in_channels=512, out_channels=1024, kernel_size=1, stride=1),
                                     ) #//32

        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(1024, num_classes)

    def forward(self, x):
        x = self.stage_1(x)#//2
        x = self.stage_2(x)#//4
        x = self.stage_3(x)#//8
        x = self.stage_4(x)#//16
        x = self.stage_5(x)#//32

        x = self.gap(x)
        x = x.flatten(start_dim=1)
        x = self.fc(x)
        
        return x
# class WhiteNet(nn.Module):
#     def __init__(self, num_classes=1000):
#         super(WhiteNet, self).__init__()

#         self.stage_1 = nn.Sequential(Block(in_channels=3, out_channels=32, kernel_size=3, stride=2),
#                                      Block(in_channels=32, out_channels=32, kernel_size=3, stride=1)) # //2
        
#         self.stage_2 = nn.Sequential(Block(in_channels=32, out_channels=64, kernel_size=3, stride=2),
#                                      Block(in_channels=64, out_channels=64, kernel_size=3, stride=1)) # //4
        
#         self.stage_3 = nn.Sequential(Block(in_channels=64, out_channels=128, kernel_size=3, stride=2),
#                                      Block(in_channels=128, out_channels=128, kernel_size=3, stride=1)) # //8
        
#         self.stage_4 = nn.Sequential(Block(in_channels=128, out_channels=256, kernel_size=3, stride=2),
#                                      Block(in_channels=256, out_channels=256, kernel_size=3, stride=1)) # //16
        
#         self.stage_5 = nn.Sequential(Block(in_channels=256, out_channels=256, kernel_size=3, stride=2),
#                                      Block(in_channels=256, out_channels=256, kernel_size=3, stride=1),
#                                      Block(in_channels=256, out_channels=256, kernel_size=3, stride=1),
#                                      Block(in_channels=256, out_channels=512, kernel_size=3, stride=1)) # // 32

#         self.gap = nn.AdaptiveAvgPool2d((1, 1))
#         self.fc = nn.Linear(512, num_classes)

#     def forward(self, x):
#         x = self.stage_1(x)#//2
#         x = self.stage_2(x)#//4
#         x = self.stage_3(x)#//8
#         x = self.stage_4(x)#//16
#         x = self.stage_5(x)#//32

#         x = self.gap(x)
#         x = x.flatten(start_dim=1)
#         x = self.fc(x)
        
#         return x    

class Conv_BN_LeakyReLU(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 ksize,
                 padding=0,
                 stride=1,
                 dilation=1):
        super(Conv_BN_LeakyReLU, self).__init__()

        self.convs = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, ksize, padding=padding, stride=stride, dilation=dilation),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.1, inplace=True)
        )

    def forward(self, x):
        return self.convs(x)

# class WhiteNet(nn.Module):
#     def __init__(self, num_classes=1000):
#         super(WhiteNet, self).__init__()

#         self.stage_1 = nn.Sequential(Block(in_channels=3, out_channels=32, kernel_size=3, stride=2),
#                                      Block(in_channels=32, out_channels=32, kernel_size=3, stride=1)) # //2
        
#         self.stage_2 = nn.Sequential(Block(in_channels=32, out_channels=64, kernel_size=3, stride=2),
#                                      Block(in_channels=64, out_channels=64, kernel_size=3, stride=1)) # //4
        
#         self.stage_3 = nn.Sequential(Block(in_channels=64, out_channels=128, kernel_size=3, stride=2),
#                                      Block(in_channels=128, out_channels=128, kernel_size=3, stride=1)) # //8
        
#         self.stage_4 = nn.Sequential(Block(in_channels=128, out_channels=256, kernel_size=3, stride=2),
#                                      Block(in_channels=256, out_channels=256, kernel_size=3, stride=1)) # //16
        
#         self.stage_5 = nn.Sequential(Block(in_channels=256, out_channels=256, kernel_size=3, stride=2),
#                                      Block(in_channels=256, out_channels=256, kernel_size=3, stride=1),
#                                      Block(in_channels=256, out_channels=256, kernel_size=3, stride=1),
#                                      Block(in_channels=256, out_channels=512, kernel_size=3, stride=1)) # // 32

#         self.gap = nn.AdaptiveAvgPool2d((1, 1))
#         self.fc = nn.Linear(512, num_classes)

#     def forward(self, x):
#         x = self.stage_1(x)#//2
#         x = self.stage_2(x)#//4
#         x = self.stage_3(x)#//8
#         x = self.stage_4(x)#//16
#         x = self.stage_5(x)#//32

#         x = self.gap(x)
#         x = x.flatten(start_dim=1)
#         x = self.fc(x)
        
#         return x

class DarkNetTiny(nn.Module):
    def __init__(self):
        super(DarkNetTiny, self).__init__()

        self.conv_1 = Conv_BN_LeakyReLU(3, 16, 3, 1)
        self.maxpool_1 = nn.MaxPool2d((2, 2), 2)  # stride = 2

        self.conv_2 = Conv_BN_LeakyReLU(16, 32, 3, 1)
        self.maxpool_2 = nn.MaxPool2d((2, 2), 2)  # stride = 4

        self.conv_3 = Conv_BN_LeakyReLU(32, 64, 3, 1)
        self.maxpool_3 = nn.MaxPool2d((2, 2), 2)  # stride = 8

        self.conv_4 = Conv_BN_LeakyReLU(64, 128, 3, 1)
        self.maxpool_4 = nn.MaxPool2d((2, 2), 2)  # stride = 16

        self.conv_5 = Conv_BN_LeakyReLU(128, 256, 3, 1)
        self.maxpool_5 = nn.MaxPool2d((2, 2), 2)  # stride = 32

        self.conv_6 = Conv_BN_LeakyReLU(256, 512, 3, 1)
        self.maxpool_6 = nn.Sequential(
            nn.ZeroPad2d((0, 1, 0, 1)),
            nn.MaxPool2d((2, 2), 1)  # stride = 32
        )
        self.conv_7 = Conv_BN_LeakyReLU(512, 1024, 3, 1)
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(1024, 1000)
    def forward(self, x):
        x = self.conv_1(x)
        x = self.maxpool_1(x)
        x = self.conv_2(x)
        x = self.maxpool_2(x)
        x = self.conv_3(x)
        x = self.maxpool_3(x)
        x = self.conv_4(x)
        x = self.maxpool_4(x)
        C_4 = self.conv_5(x)  # stride = 16
        x = self.maxpool_5(C_4)
        x = self.conv_6(x)
        x = self.maxpool_6(x)
        C_5 = self.conv_7(x)  # stride = 32

        x = self.gap(C_5)
        x = x.flatten(start_dim=1)
        x = self.fc(x)
        
        return x


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        def conv_bn(inp, oup, stride):
            return nn.Sequential(
                nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
                nn.BatchNorm2d(oup),
                nn.ReLU(inplace=True)
            )

        def conv_dw(inp, oup, stride):
            return nn.Sequential(
                nn.Conv2d(inp, inp, 3, stride, 1, groups=inp, bias=False),
                nn.BatchNorm2d(inp),
                nn.ReLU(inplace=True),
    
                nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
                nn.ReLU(inplace=True),
            )

        self.model = nn.Sequential(
            conv_bn(  3,  32, 2), 
            conv_dw( 32,  64, 1),
            conv_dw( 64, 128, 2),
            conv_dw(128, 128, 1),
            conv_dw(128, 256, 2),
            conv_dw(256, 256, 1),
            conv_dw(256, 512, 2),
            conv_dw(512, 512, 1),
            conv_dw(512, 512, 1),
            conv_dw(512, 512, 1),
            conv_dw(512, 512, 1),
            conv_dw(512, 512, 1),
            conv_dw(512, 1024, 2),
            conv_dw(1024, 1024, 1),
            nn.AvgPool2d(7),
        )
        self.fc = nn.Linear(1024, 1000)

    def forward(self, x):
        x = self.model(x)
        x = x.view(-1, 1024)
        x = self.fc(x)
        return x

if __name__ == '__main__':
    scaler = torch.cuda.amp.GradScaler()
    #model = DarkNetTiny().cuda()
    model = WhiteNet().cuda()
    #model = Net().cuda()

    import torchvision.models as models
    #model = models.resnet18().cuda()
    model.eval()

    shape = (1, 3, 1024, 1024)
    with torch.no_grad():
        x = torch.randn(shape).cuda()
        import time
        model(x)
        avg_time = 0

        for i in range(0, 10):
            x = torch.randn(shape).cuda()

            torch.cuda.synchronize()
            t2 = time.time()
            #with torch.cuda.amp.autocast():
            model(x)

            torch.cuda.synchronize()
            t3 = time.time()

            avg_time += t3 - t2

        avg_time /= 10.0

    print('avg_time: ', avg_time)
    print(sum(p.numel() for p in model.parameters()))
