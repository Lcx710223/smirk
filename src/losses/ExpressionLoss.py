import torch
import torch.nn as nn
from src.losses.resnet import resnet50

class ExpressionLoss(nn.Module):
    def __init__(self, device="cpu"):
        super(ExpressionLoss, self).__init__()
        # 支持 CPU/GPU 自动选择
        if isinstance(device, str):
            self.device = torch.device(device)
        else:
            self.device = device

        # 初始化 backbone
        self.backbone = resnet50(num_classes=100, include_top=False, emoca_specific=True).eval()
        self.backbone = self.backbone.to(self.device)

        # 加载权重
        emotion_checkpoint = torch.load(
            'assets/ResNet50/checkpoints/deca-epoch=01-val_loss_total/dataloader_idx_0=1.27607644.ckpt',
            map_location=self.device,
            weights_only=False
        )['state_dict']

        state_dict = {}
        for k, v in emotion_checkpoint.items():
            if k.startswith("backbone."):
                new_k = k.replace("backbone.", "")
                # 跳过 fc 层
                if "fc.weight" in new_k or "fc.bias" in new_k:
                    continue
                state_dict[new_k] = v

        self.backbone.load_state_dict(state_dict, strict=False)
        self.loss_fn = nn.MSELoss()

    def forward(self, pred, target):
        pred = pred.to(self.device)
        target = target.to(self.device)

        feat_pred = self.backbone(pred)
        feat_target = self.backbone(target)
        return self.loss_fn(feat_pred, feat_target)
