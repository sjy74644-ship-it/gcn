# Visual Teacher -> Radar Student 2D Keypoint Distillation (Single-Frame)

本项目当前版本是**单帧蒸馏**，推荐的显式流程是：

1) 先做 **teacher-only pretraining**（视觉数据）
2) 导出 `teacher_backbone.pt` 和 `teacher_head.pt`
3) 再做 visual->radar distillation

---

## A. Teacher 预训练（你只有 yolov8n-pose.pt 的场景）

你现在只有 `yolov8n-pose.pt`，它会被当作**初始化来源**，不是最终 teacher。

### 关键策略

- 从 `yolov8n-pose.pt` 尽可能加载匹配到 `teacher.encoder` 的权重
- 对 shape/name 不匹配的参数跳过并打印日志
- 不直接复用原 YOLO pose head（本项目 head 自定义，关键点数可能不同）
- 自定义 `HeatmapHead` 单独随机初始化并监督训练

### 运行命令

```bash
python train_teacher.py \
  --visual_img_dir /path/to/visual/images \
  --input_lbl_dir /path/to/labels \
  --init_yolo_ckpt /path/to/yolov8n-pose.pt \
  --num_joints 13 \
  --kpt_dim 3 \
  --visual_input_h 270 \
  --visual_input_w 480 \
  --heatmap_size 64 \
  --yolo_model yolov8n.yaml \
  --feat_channels 256 \
  --num_epochs 20 \
  --save_dir ./teacher_ckpts
```

### 预训练输出

- `teacher_ckpts/teacher_backbone.pt`
- `teacher_ckpts/teacher_head.pt`
- `teacher_ckpts/teacher_best_full.pt`

---

## B. Distillation 训练（visual teacher + radar student）

```bash
python train.py \
  --input_img_dir /path/to/radar/images \
  --input_lbl_dir /path/to/labels \
  --visual_img_dir /path/to/visual/images \
  --teacher_backbone_ckpt ./teacher_ckpts/teacher_backbone.pt \
  --teacher_head_ckpt ./teacher_ckpts/teacher_head.pt \
  --num_joints 13 \
  --kpt_dim 3 \
  --visual_input_h 270 \
  --visual_input_w 480 \
  --radar_input_h 640 \
  --radar_input_w 640 \
  --heatmap_size 64 \
  --num_epochs 20 \
  --save_dir ./checkpoints
```

默认 teacher 冻结；若要联合训练 teacher：加 `--train_teacher`。

`train.py` 会按 `val_ratio` 自动划分 train/val，并在每个 epoch 打印训练和验证损失。

---

## C. 推理

### 1) 无标签纯推理

```bash
python predict.py \
  --input_img_dir /path/to/radar/images \
  --checkpoint /path/to/checkpoints/student_best.pt \
  --output_csv /path/to/preds.csv \
  --radar_input_h 640 \
  --radar_input_w 640
```

### 2) 可选带标签评估

```bash
python predict.py \
  --input_img_dir /path/to/radar/images \
  --input_lbl_dir /path/to/labels \
  --checkpoint /path/to/checkpoints/student_best.pt \
  --output_csv /path/to/preds.csv \
  --radar_input_h 640 \
  --radar_input_w 640
```

---

## 标签格式（YOLO Pose）

```text
class cx cy w h kx1 ky1 v1 ... kxK kyK vK
```

- `v <= 0` 的关键点视为无效点，不参与 heatmap 和 loss 监督。
- 空标签（空 txt）视为无人样本，不会被错误监督到左上角。

---

## 关键工程约束

- 单帧版（时序损失不在当前版本中）
- heatmap 输出会显式对齐到 `heatmap_size`
- 默认要求 `visual_img_dir`，避免静默退化为同模态蒸馏


> 视觉输入尺寸与雷达输入尺寸可不同（例如视觉 270x480，雷达 640x640）。

> 注意：YOLOv8 主干要求输入尺寸与 stride 对齐（默认 32）。
> 例如 `270x480` 会自动对齐为 `288x480` 以避免 concat 尺寸报错。
