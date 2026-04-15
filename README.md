# Visual-to-Radar SRRL Pose Distillation Framework

这是一个**视觉向雷达的姿态估计知识蒸馏（SRRL风格）**训练框架（PyTorch）：

- Teacher 输入：视觉 PNG (`.png`)
- Student 输入：雷达 PNG (`.png`)
- 输出：K 关节 heatmap (`[K,H,W]`)

## 当前实现对应的损失

总损失：

\[
L = w_{sup}L_{sup} + w_{repr}L_{repr} + w_{head}L_{head} + w_{rel}L_{rel} + w_{temp}L_{temp}
\]

默认系数：

- `w_sup=1.0`
- `w_repr=0.5`
- `w_head=1.0`
- `w_rel=0.2`
- `w_temp=0.1`

具体项：

1. `L_sup`：学生 heatmap 对 GT heatmap（或伪标签）的监督。  
2. `L_repr`：SRRL 表征对齐，默认使用统计版（通道均值+方差）对齐。  
3. `L_head`：冻结视觉头 `D_v`，约束 `D_v(G(F_r))` 逼近视觉输出 `H_v`。  
4. `L_rel`：关节级关系矩阵蒸馏（heatmap token 相似矩阵）。  
5. `L_temp`：时间平滑（已实现接口，图像批训练时默认为 0）。

## 数据格式

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

`labels.csv` 使用关键点坐标生成 GT heatmap：

```csv
id,x1,y1,x2,y2,...,x17,y17
000001,120,90,130,100,...
000002,118,88,128,97,...
```

说明：
- 关键点坐标按 256×256 输入尺度解析。
- 训练时会自动生成 `K×heatmap_size×heatmap_size` 的高斯热图。

## 快速开始

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

python train.py \
  --data_root ./data \
  --labels_csv ./data/labels.csv \
  --num_joints 17 \
  --heatmap_size 64
```

### 常用选项

- `--use_pseudo_sup`：无高质量雷达标注时，用视觉输出作为伪标签监督。
- `--simple_repr`：切换到简单版 `||G(F_r)-sg(F_v)||^2`（默认是统计版）。
- `--train_teacher`：不冻结 teacher（默认冻结）。


## 指定输入/输出路径进行推理（你要求的形式）

已提供 `predict.py`，满足：

- 输入路径：`visual_dir`（视觉图像目录）、`radar_dir`（雷达图像目录）、`gt_csv`（2D坐标真值）
- 输出路径：`output_csv`（雷达预测的2D坐标）

```bash
python predict.py \
  --visual_dir /path/to/visual \
  --radar_dir /path/to/radar \
  --gt_csv /path/to/labels.csv \
  --checkpoint /path/to/student_best.pt \
  --output_csv /path/to/pred_radar_coords.csv
```

输出 CSV 格式：
- `id,pred_x1,pred_y1,...,pred_xK,pred_yK`

## 代码结构

- `distill_framework/dataset.py`：成对 PNG + 关键点到 GT heatmap 生成
- `distill_framework/models.py`：Teacher/Student、SRRL projector、soft-argmax
- `distill_framework/losses.py`：`L_sup/L_repr/L_head/L_rel/L_temp`
- `distill_framework/trainer.py`：蒸馏训练流程
- `train.py`：训练命令行入口
- `predict.py`：给定输入/输出路径，导出雷达预测2D坐标
