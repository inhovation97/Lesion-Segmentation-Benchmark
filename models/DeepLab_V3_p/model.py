import torch
import torch.nn as nn
import torch.nn.functional as F
# from modeling.sync_batchnorm.batchnorm import SynchronizedBatchNorm2d
from models.DeepLab_V3_p.aspp import build_aspp
from models.DeepLab_V3_p.decoder import build_decoder
from models.DeepLab_V3_p.backbone import build_backbone




class DeepLab(nn.Module):
    def __init__(self, backbone='xception', output_stride=16, num_classes=21,
                 sync_bn=False, freeze_bn=False):  # 둘다 False !!
        super(DeepLab, self).__init__()
        if backbone == 'drn':
            output_stride = 8

        #if sync_bn == True:
            #BatchNorm = SynchronizedBatchNorm2d
        
        #else:
            #BatchNorm = nn.BatchNorm2d
        
        BatchNorm = nn.BatchNorm2d
        
        self.backbone = build_backbone(backbone, output_stride, BatchNorm)
        self.aspp = build_aspp(backbone, output_stride, BatchNorm)
        self.decoder = build_decoder(num_classes, backbone, BatchNorm)

        self.freeze_bn = freeze_bn

    def forward(self, input):
        x, low_level_feat = self.backbone(input)
        x = self.aspp(x)
        x = self.decoder(x, low_level_feat)
        x = F.interpolate(x, size=input.size()[2:], mode='bilinear', align_corners=True)

        return x

    
if __name__ == "__main__":
    model = DeepLab(backbone='xception', output_stride=16)
    model.eval()
    input = torch.rand(1, 3, 513, 513)
    output = model(input)
    print(output.size())