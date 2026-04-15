# Visual Teacher -> Radar Student 2D Keypoint Distillation (Single-Frame)

本项目当前版本是**单帧蒸馏**（不是时序版）。

## 当前实现状态

- ✅ 跨模态蒸馏：visual teacher + radar student
- ✅ teacher checkpoint 显式加载（backbone/head可分开）
- ✅ 支持 teacher 冻结或联合训练（`--train_teacher`）
- ✅ 空标签/无人样本正确处理（不会监督到左上角）
- ✅ visibility/valid（`v`）参与 heatmap 构建与 loss mask
- ✅ 训练与推理脚本可运行
- ✅ 推理不强依赖标签目录（可仅输入图像）

## 标签格式（YOLO Pose）

每行格式：

```text
class cx cy w h kx1 ky1 v1 kx2 ky2 v2 ... kxK kyK vK
```

其中：
- `K = num_joints`（默认 13）
- `kpt_dim = 3`（默认 x,y,v）
- `v <= 0` 的关键点视为无效点，不画高斯热图，也不参与监督

空标签（空 txt）会被视为**无人样本**：
- `kp_valid` 全 0
- `heatmap_gt` 全 0
- 监督分支通过 mask 跳过，不会学成 `(0,0)` 有人

## 训练（推荐）

```bash
python train.py \
  --input_img_dir /path/to/radar/images \
  --input_lbl_dir /path/to/labels \
  --visual_img_dir /path/to/visual/images \
  --teacher_backbone_ckpt /path/to/teacher_backbone.pt \
  --teacher_head_ckpt /path/to/teacher_head.pt \
  --num_joints 13 \
  --kpt_dim 3 \
  --input_size 320 \
  --heatmap_size 64 \
  --yolo_model yolov8n.yaml \
  --feat_channels 256 \
  --batch_size 16 \
  --num_epochs 20 \
  --save_dir ./checkpoints
```

### teacher checkpoint 规则

- `--teacher_backbone_ckpt`：加载到 `teacher.encoder`
- `--teacher_head_ckpt`：加载到 `teacher.head`
- 默认必须提供至少一个 teacher ckpt；否则报错。
- 如你确实要随机 teacher（不推荐），显式加 `--allow_random_teacher`。

### teacher 是否训练

- 默认：`train_teacher=False`（冻结 teacher）
- 显式加 `--train_teacher`：teacher 参数加入 optimizer 并更新

## 推理

### 1) 仅推理（无标签）

```bash
python predict.py \
  --input_img_dir /path/to/radar/images \
  --checkpoint /path/to/checkpoints/student_best.pt \
  --output_csv /path/to/preds.csv \
  --num_joints 13 \
  --input_size 320 \
  --heatmap_size 64
```

### 2) 推理 + 简单评估（可选提供标签）

```bash
python predict.py \
  --input_img_dir /path/to/radar/images \
  --input_lbl_dir /path/to/labels \
  --checkpoint /path/to/checkpoints/student_best.pt \
  --output_csv /path/to/preds.csv
```

会额外打印 valid joints 上的 mean L2。

## 输出说明

`preds.csv` 格式：

```text
id,pred_x1,pred_y1,...,pred_xK,pred_yK
```

## 关键工程约束

- **单帧版**：当前已移除时序损失占位逻辑，避免“看起来有功能、实际未启用”。
- heatmap 尺寸对齐：模型输出会显式插值到 `heatmap_size`；训练时再做 shape 检查。
- 默认要求 `visual_img_dir`，避免静默退化成同模态蒸馏；若确需同模态，需要显式 `--allow_same_modal_distill`。
