# reset_query_embedding_hook.py

from mmengine.hooks import Hook
import torch.nn.init as init
from mmengine.registry import HOOKS

@HOOKS.register_module()
class ResetQueryEmbeddingHook(Hook):
    def before_train(self, runner):
        model = runner.model
        if hasattr(model, 'query_embedding'):
            print('🔁 正在重置 query_embedding 权重...')
            init.xavier_uniform_(model.query_embedding.weight)
            print('✅ query_embedding 已重置')
        else:
            print('⚠️ 模型中未找到 query_embedding')

print('✅ 自定义 Hook ResetQueryEmbeddingHook 已注册')

