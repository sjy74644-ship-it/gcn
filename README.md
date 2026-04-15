# Visual-to-Radar Pose Distillation Framework

这是一个**视觉向雷达的姿态估计知识蒸馏**训练框架（PyTorch），约束如下：

- Teacher 输入：视觉 PNG (`.png`)
- Student 输入：雷达 PNG (`.png`)
- 输出：姿态向量（默认 6DoF，可配置）

## 功能

- 成对 PNG 数据加载（visual/radar）
- Teacher（视觉）与 Student（雷达）双分支
- 三类损失：
  - 姿态监督损失（对 GT）
  - 输出蒸馏损失（teacher pose vs student pose）
  - 特征蒸馏损失（中间特征对齐）
- 支持 teacher 冻结（常见蒸馏设置）

## 目录结构（建议）

```text
project_root/
  data/
    visual/
      000001.png
      000002.png
    radar/
      000001.png
      000002.png
    labels.csv
```

`labels.csv` 示例：

```csv
id,tx,ty,tz,roll,pitch,yaw
000001,0.1,0.2,0.0,0.0,0.1,-0.1
000002,0.2,0.1,0.0,0.0,0.0,-0.2
```

## 快速开始

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python train.py --data_root ./data --labels_csv ./data/labels.csv
```

## 关键参数

- `--pose_dim`：姿态维度，默认 `6`
- `--alpha_pose`：学生对 GT 的监督权重
- `--alpha_kd_out`：输出蒸馏权重
- `--alpha_kd_feat`：特征蒸馏权重
- `--freeze_teacher`：是否冻结 teacher（默认 true）

## 说明

- 当前框架偏向“可落地骨架”，你可以很方便替换 backbone（例如 ResNet、Swin、ConvNeXt）。
- 如果你的雷达 PNG 是单通道热力图，框架会自动扩展到 3 通道。
