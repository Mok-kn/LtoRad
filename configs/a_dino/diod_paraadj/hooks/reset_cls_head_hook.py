from mmengine.hooks import Hook
from torch import nn

class ResetClassifierHook(Hook):
    def __init__(self, num_classes=3):
        self.num_classes = num_classes

    def before_train(self, runner):
        model = runner.model
        print(f'Resetting classifier to num_classes = {self.num_classes}')

        # 获取 bbox_head
        bbox_head = model.bbox_head

        # 多层 decoder，每层都有一个 cls_branch
        for i, cls_branch in enumerate(bbox_head.cls_branches):
            in_features = cls_branch.in_features
            bbox_head.cls_branches[i] = nn.Linear(in_features, self.num_classes)

        # 如果 loss_cls 是 FocalLoss，还需设置其类别数
        if hasattr(bbox_head, 'loss_cls') and hasattr(bbox_head.loss_cls, 'num_classes'):
            bbox_head.loss_cls.num_classes = self.num_classes

        print('✅ Classifier reset completed.')
