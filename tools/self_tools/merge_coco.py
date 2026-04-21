import os
import json
import shutil
from tqdm import tqdm
import argparse

def merge_coco(json1_path, img1_dir, json2_path, img2_dir, output_dir):
    with open(json1_path, 'r', encoding='utf-8') as f:
        coco1 = json.load(f)
    with open(json2_path, 'r', encoding='utf-8') as f:
        coco2 = json.load(f)

    assert coco1['categories'] == coco2['categories'], "两个数据集的类别不一致"

    merged = {
        'images': [],
        'annotations': [],
        'categories': coco1['categories']
    }

    os.makedirs(os.path.join(output_dir, 'images'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'annotations'), exist_ok=True)

    ann_id = 1
    img_id = 1
    used_filenames = set()

    def process_dataset(coco, img_dir, prefix):
        nonlocal ann_id, img_id
        img_id_map = {}
        images = []
        annotations = []

        for img in coco['images']:
            old_id = img['id']
            fname = img['file_name']
            new_fname = f"{prefix}_{fname}"
            while new_fname in used_filenames:
                new_fname = f"{prefix}_{img_id}_{fname}"
            used_filenames.add(new_fname)

            src_path = os.path.join(img_dir, fname)
            dst_path = os.path.join(output_dir, 'images', new_fname)
            if not os.path.exists(src_path):
                print(f"[警告] 找不到图像文件: {src_path}")
                continue
            shutil.copyfile(src_path, dst_path)

            new_img = {
                'id': img_id,
                'file_name': new_fname,
                'width': img.get('width', 0),
                'height': img.get('height', 0)
            }
            img_id_map[old_id] = img_id
            img_id += 1
            images.append(new_img)

        for ann in coco['annotations']:
            old_img_id = ann['image_id']
            if old_img_id not in img_id_map:
                continue
            new_ann = ann.copy()
            new_ann['id'] = ann_id
            new_ann['image_id'] = img_id_map[old_img_id]
            ann_id += 1
            annotations.append(new_ann)

        return images, annotations

    print("[INFO] 正在处理第一个数据集...")
    imgs1, anns1 = process_dataset(coco1, img1_dir, 'ds1')
    print("[INFO] 正在处理第二个数据集...")
    imgs2, anns2 = process_dataset(coco2, img2_dir, 'ds2')

    merged['images'] = imgs1 + imgs2
    merged['annotations'] = anns1 + anns2

    out_json_path = os.path.join(output_dir, 'annotations', 'merged_annotations.json')
    with open(out_json_path, 'w', encoding='utf-8') as f:
        json.dump(merged, f, ensure_ascii=False, indent=2)

    print("\n✅ 合并完成：")
    print(f" - 图像总数：{len(merged['images'])}")
    print(f" - 标注总数：{len(merged['annotations'])}")
    print(f" - 合并标注文件路径：{out_json_path}")
    print(f" - 图像保存路径：{os.path.join(output_dir, 'images')}")

if __name__ == '__main__':
    json1_path = "/data1/sjc/work2/mmdetection/data/JointTest/annotations/instances_val2017.json"
    img1_dir = "/data1/sjc/work2/mmdetection/data/JointTest/val2017"
    json2_path = "/data1/sjc/work3/mmdetection/data/MSAR/annotations/instances_val2017.json"
    img2_dir = "/data1/sjc/work3/mmdetection/data/MSAR/val2017"
    output_dir = "/data1/sjc/work3/mmdetection/data/Joint_test"

    merge_coco(json1_path, img1_dir, json2_path, img2_dir, output_dir)
