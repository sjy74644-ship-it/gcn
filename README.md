# Visual-to-Radar SRRL Distillation (YOLOv8 + YOLO标签读取流程)

已按你的要求，把数据读取流程替换为你给出的 YOLO 流程：

- 从 `images/` 递归扫描图片
- 按相对路径到 `labels/` 找对应 `.txt`
- 校验标签列数：`1 + 4 + NKPT*KPT_DIM`
- 自动随机划分 train/val（`val_ratio` + `random_seed`）
- 输入预处理采用 YOLOv8 风格：`Resize(input_size)` + `ToTensor()`

> 默认关键点设置：`NKPT=13`, `KPT_DIM=3`

## 目录示例

```text
data_root/
  images/
    action1/0001.png
  labels/
    action1/0001.txt
```

## 训练

```bash
python train.py \
  --input_img_dir /path/to/data_root/images \
  --input_lbl_dir /path/to/data_root/labels \
  --visual_img_dir /path/to/visual_images_optional \
  --num_joints 13 \
  --kpt_dim 3 \
  --val_ratio 0.2 \
  --random_seed 42 \
  --input_size 320 \
  --yolo_model yolov8n.yaml \
  --feat_channels 256
```

说明：
- 如果不传 `--visual_img_dir`，则默认 `visual=radar`。
- 可通过 `--auto_create_empty_label` 自动创建缺失空标签。
- 已加入环境参数开关：`TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD`、`KMP_DUPLICATE_LIB_OK`、`OMP_NUM_THREADS`。

## 推理（输出雷达2D坐标）

```bash
python predict.py \
  --input_img_dir /path/to/data_root/images \
  --input_lbl_dir /path/to/data_root/labels \
  --checkpoint /path/to/student_best.pt \
  --output_csv /path/to/pred_radar_coords.csv \
  --num_joints 13 \
  --kpt_dim 3 \
  --input_size 320 \
  --yolo_model yolov8n.yaml \
  --feat_channels 256
```

输出格式：

```text
id,pred_x1,pred_y1,...,pred_x13,pred_y13
```
