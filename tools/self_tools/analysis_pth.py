import torch
from collections import defaultdict

def analyze_pth(pth_path, topn=10):
    # 加载 checkpoint
    ckpt = torch.load(pth_path, map_location='cpu')

    # 提取 state_dict（兼容不同保存格式）
    if 'state_dict' in ckpt:
        state_dict = ckpt['state_dict']
    elif isinstance(ckpt, dict):
        state_dict = ckpt
    else:
        raise ValueError("Unsupported .pth format")

    print(f"📦 Loaded {pth_path}, total params: {len(state_dict)}")

    # 初始化统计
    module_stats = defaultdict(lambda: {'count': 0, 'params': 0})
    total_params = 0

    # 遍历每个参数
    for name, param in state_dict.items():
        param_count = param.numel()
        total_params += param_count

        # 取一级模块名，如 backbone.xxx -> backbone
        module_key = name.split('.')[0]
        module_stats[module_key]['count'] += 1
        module_stats[module_key]['params'] += param_count

    # 输出分析结果
    print("\n📊 Parameter breakdown by top-level module:")
    print(f"{'Module':<20} {'#Params':>12} {'Ratio':>10}")
    print("=" * 45)
    for module, info in sorted(module_stats.items(), key=lambda x: -x[1]['params']):
        ratio = info['params'] / total_params * 100
        print(f"{module:<20} {info['params']:>12,} {ratio:>9.2f}%")

    print(f"\n🔢 Total parameters: {total_params:,}\n")

    # 可选：打印前 N 个具体参数
    print(f"🔍 Preview of first {topn} params:")
    for i, (name, param) in enumerate(state_dict.items()):
        print(f"{i+1:02d}. {name:<60} shape: {tuple(param.shape)}")
        if i + 1 >= topn:
            break

# ✅ 用法示例
if __name__ == "__main__":
    analyze_pth('/data1/sjc/work3/mmdetection/work_dirs/dino/diod_freeze_encoder/HR_LE_DI_DO_MS_SS/epoch_20.pth')  # 替换为你的路径
