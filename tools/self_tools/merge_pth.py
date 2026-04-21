import torch

def merge_pth(
    pth_a,           # 提供 backbone 的 pth
    pth_b,           # 提供其余模块的 pth
    output_path,     # 保存合并后的新 pth
    module_from_a=('backbone',),  # 指定从 A 加载哪些模块
):
    # 加载权重
    state_dict_a = torch.load(pth_a, map_location='cpu')
    state_dict_b = torch.load(pth_b, map_location='cpu')

    # 兼容 'state_dict' 包裹情况
    state_dict_a = state_dict_a.get('state_dict', state_dict_a)
    state_dict_b = state_dict_b.get('state_dict', state_dict_b)

    new_state_dict = {}

    for name, param in state_dict_a.items():
        if name.split('.')[0] in module_from_a:
            new_state_dict[name] = param

    for name, param in state_dict_b.items():
        if name not in new_state_dict:
            new_state_dict[name] = param

    print(f"🔁 模块合并完成：从 A 加载 {[m for m in module_from_a]}，其余来自 B")
    print(f"📝 总参数数：{len(new_state_dict)}，保存到 {output_path}")

    torch.save({'state_dict': new_state_dict}, output_path)
    print("✅ 保存完成！")


# ✅ 使用示例（请替换为你的路径）
if __name__ == "__main__":
    merge_pth(
        pth_a='/data1/sjc/work3/mmdetection/work_dirs/dino/diod/HR_LE_DI_DO/epoch_20.pth',
        pth_b='/data1/sjc/work3/mmdetection/work_dirs/dino/diod_freeze_encoder/HR_LE_DI_DO_MS_SS/epoch_20.pth',
        output_path='/data1/sjc/work3/mmdetection/work_dirs/dino/merged_pth/opt_encoder_query_sar/1.pth',
        module_from_a=('encoder', 'query_embedding')
    )
