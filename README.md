# Visual-to-Radar SRRL Pose Distillation Framework (YOLOv8 Backbone)

按你的要求，骨干网络已切换为 **YOLOv8**，并且输入格式与 YOLOv8 保持一致：

- 输入图像：RGB PNG
- 预处理：Resize 到 `input_size`（默认 640），`ToTensor()` 转到 `[0,1]`
- Teacher 输入：视觉图像
- Student 输入：雷达图像
- 输出：雷达预测 2D 坐标（由 heatmap 经 soft-argmax 解码）

## 损失

\[
L = w_{sup}L_{sup} + w_{repr}L_{repr} + w_{head}L_{head} + w_{rel}L_{rel} + w_{temp}L_{temp}
\]

默认：`w_sup=1.0, w_repr=0.5, w_head=1.0, w_rel=0.2, w_temp=0.1`

## 数据格式

```text
project_root/
  data/
    visual/
      000001.png
    radar/
      000001.png
    labels.csv
```

`labels.csv`：

```csv
id,x1,y1,x2,y2,...,x17,y17
000001,120,90,130,100,...
```

坐标默认按 `input_size` 尺度解释（默认 640）。

## 训练

```bash
python train.py \
  --data_root ./data \
  --labels_csv ./data/labels.csv \
  --num_joints 17 \
  --input_size 640 \
  --yolo_model yolov8n.yaml \
  --feat_channels 256
```

## 指定输入/输出路径推理（你要的方式）

输入：视觉图像目录、雷达图像目录、2D 真值 CSV。  
输出：雷达预测 2D 坐标 CSV。

```bash
python predict.py \
  --visual_dir /path/to/visual \
  --radar_dir /path/to/radar \
  --gt_csv /path/to/labels.csv \
  --checkpoint /path/to/student_best.pt \
  --output_csv /path/to/pred_radar_coords.csv \
  --input_size 640 \
  --yolo_model yolov8n.yaml \
  --feat_channels 256
```

输出 CSV：

```text
id,pred_x1,pred_y1,...,pred_xK,pred_yK
```

## 文件

- `distill_framework/dataset.py`：YOLOv8 输入格式的数据读取与 GT heatmap 生成
- `distill_framework/models.py`：YOLOv8 backbone + heatmap head + projector
- `distill_framework/losses.py`：SRRL 多项损失
- `distill_framework/trainer.py`：训练流程
- `train.py`：训练入口
- `predict.py`：路径驱动的雷达2D坐标导出
