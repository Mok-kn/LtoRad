from mmengine.hooks import Hook
from mmengine.registry import HOOKS
import torch

@HOOKS.register_module()
class DomainPrototypeHook(Hook):
    def __init__(self, use_neck=False, momentum=0.999, save_path=None):
        self.use_neck = use_neck
        self.momentum = momentum
        self.save_path = save_path
        self.prototype = None  # EMA vector

    def after_train_iter(self, runner, batch_idx, data_batch, outputs):
        model = runner.model
        if hasattr(model, 'module'):
            model = model.module

        device = next(model.parameters()).device  # 获取设备

        # 对每张图像分别处理
        for img in data_batch['inputs']:
            img = img.to(device).float()

            # 可选：resize 到统一尺寸（如 800x800），否则部分 backbone 不接受任意大小
            img = torch.nn.functional.interpolate(img.unsqueeze(0), size=(800, 800), mode='bilinear', align_corners=False)

            # 提取特征
            if self.use_neck:
                feat = model.neck(model.backbone(img))[-1]  # [1, C, H, W]
            else:
                feat = model.backbone(img)[-1]

            feat_vector = feat.mean(dim=[2, 3]).squeeze(0)  # (C,)

            # EMA 更新
            with torch.no_grad():
                if self.prototype is None:
                    self.prototype = feat_vector.clone()
                else:
                    self.prototype = self.momentum * self.prototype + (1 - self.momentum) * feat_vector



    def after_train(self, runner):
        if self.save_path is not None and self.prototype is not None:
            torch.save(self.prototype.cpu(), self.save_path)
            print(f'✅ 域原型已保存至 {self.save_path}')
