# my_hooks.py

import torch
import torch.nn.functional as F
from mmengine.hooks import Hook
from mmdet.registry import HOOKS
from mmengine.runner import Runner
import os

@HOOKS.register_module()
class DomainProtoHook(Hook):
    """
    一个在训练过程中提取并更新领域特征原型的 Hook。

    Args:
        momentum (float): EMA 更新的动量。值越大，原型更新越慢。默认为 0.999.
    """
    def __init__(self, momentum: float = 0.99):
        super().__init__()
        self.momentum = momentum
        # self.prototypes 将在 before_run 中初始化
        self.prototypes = {}

    def _get_domain_name(self, runner: Runner) -> str:
        """从当前数据加载器中获取领域名称。"""
        # MMDetection 的 dataloader 会包装数据集
        # 我们需要访问到最内层的 dataset 对象来获取 metainfo
        dataset = runner.train_loop.dataloader.dataset
        # 处理 ConcatDataset 等情况
        while hasattr(dataset, 'dataset'):
            dataset = dataset.dataset
        
        if 'domain_name' not in dataset.metainfo:
            raise ValueError("当前数据集中未定义 'domain_name'，请在数据集配置文件中添加。")
        return dataset.metainfo['domain_name']

    def before_run(self, runner: Runner) -> None:
        """在训练开始前，初始化原型字典。"""
        self.prototypes = {}
        runner.logger.info("DomainPrototypeHook 已初始化，准备提取领域原型...")
        runner.logger.info(f"原型 EMA 更新动量: {self.momentum}")

    def after_train_iter(self,
                         runner: Runner,
                         batch_idx: int,
                         data_batch: dict,
                         outputs: dict) -> None:
        """
        在每个训练迭代后执行。
        1. 提取 backbone 特征。
        2. 计算当前 batch 的平均特征。
        3. 通过 EMA 更新领域原型。
        """
        # 兼容单卡和多卡(DDP)模式
        model = runner.model.module if hasattr(runner.model, 'module') else runner.model
        domain_name = self._get_domain_name(runner)

        # 1. 提取 backbone 特征
        # DINO 的 backbone (如 SwinTransformer) 会输出多层特征图
        # 我们通常使用最后一层的输出
        with torch.no_grad():
            # batch_inputs['inputs'] 是一个 (B, C, H, W) 的张量
            backbone_features = model.backbone(data_batch['inputs'])
            last_stage_features = backbone_features[-1] # shape: (B, C, H, W)

            # 2. 对特征图进行全局平均池化，得到每个样本的特征向量
            # shape: (B, C, 1, 1) -> (B, C)
            batch_vectors = F.adaptive_avg_pool2d(last_stage_features, (1, 1)).squeeze(-1).squeeze(-1)
            
            # 计算当前 batch 的特征原型
            current_batch_proto = batch_vectors.mean(dim=0)

        # 3. 通过 EMA 更新领域原型
        if domain_name not in self.prototypes:
            # 如果是第一次遇到这个领域，直接初始化原型
            self.prototypes[domain_name] = current_batch_proto
            runner.logger.info(f"已初始化领域 '{domain_name}' 的原型。")
        else:
            # 使用 EMA 进行更新
            self.prototypes[domain_name] = \
                self.momentum * self.prototypes[domain_name] + \
                (1 - self.momentum) * current_batch_proto
        
    def after_run(self, runner: Runner) -> None:
        """在整个训练流程结束后，保存所有领域的原型。"""
        if not self.prototypes:
            runner.logger.warning("未收集到任何领域原型，不进行保存。")
            return
            
        save_path = os.path.join(runner.work_dir, 'domain_prototypes.pth')
        runner.logger.info(f"训练结束，将所有领域原型保存至: {save_path}")
        
        # 将原型从 GPU 移至 CPU 再保存
        cpu_prototypes = {name: proto.cpu() for name, proto in self.prototypes.items()}
        torch.save(cpu_prototypes, save_path)
        runner.logger.info("领域原型保存完毕。")
        runner.logger.info(f"最终收集到的领域: {list(self.prototypes.keys())}")