# 静态字形与动态在线数据联合分析训练项目

本项目实现**离线字形图像（GNT）与在线笔迹轨迹**的联合建模与分阶段训练，用于静态字形数据与动态在线数据的结合分析。包含数据预处理、三阶段模型训练（重建 → 跨模态对齐 → 质量排序）、评估与可视化，并预留 Stage4 少标注微调配置。

---

## 项目目标

- **数据**：解析 GNT 离线字形图像与在线轨迹 CSV（时间戳、x/y、压力 f），生成统一样本格式（图像张量 + 轨迹张量）。
- **Stage 1**：轨迹遮蔽重建（MAE），学习在线轨迹表示。
- **Stage 2**：图像-轨迹跨模态对齐（对比学习 InfoNCE）。
- **Stage 3**：质量敏感排序（劣化样本 vs 原始样本，Ranking Loss）。
- **Stage 4**：少标注微调（约 10% 质量标签，可选）。

每阶段均记录日志、保存模型权重与评估指标，便于复现与对比。

---

## 环境要求

- Python 3.8+
- PyTorch >= 1.9.0
- 其他依赖见 `requirements.txt`

```bash
pip install -r requirements.txt
```

---

## 项目结构

```
xunlian/
├── README.md                 # 本说明
├── requirements.txt          # 依赖
├── run_pipeline.py           # 一键运行：构建数据 → Stage1 → Stage2 → Stage3
│
├── data/
│   ├── raw/
│   │   ├── gnt/              # 原始 .gnt 文件
│   │   ├── online/            # 与 gnt 一一对应的在线轨迹（*_online.csv）
│   │   └── pairs.csv         # 样本配对表：sample_id, gnt_path, online_path, writer_id, char
│   └── processed/
│       └── samples/          # 预处理后的 .pt 样本（每条一个文件）
│
├── configs/
│   ├── stage1.yaml           # Stage1 重建配置
│   ├── stage2.yaml           # Stage2 对齐配置
│   ├── stage3.yaml           # Stage3 排序配置
│   └── stage4.yaml           # Stage4 少标注微调配置（需质量标签）
│
├── src/
│   ├── datasets/
│   │   ├── gnt_reader.py     # 解析 GNT 文件（头信息 + 灰度图）
│   │   ├── online_reader.py  # 解析在线 CSV，生成 (x, y, f, speed, dt)
│   │   ├── build_dataset.py  # 根据 pairs.csv 构建 .pt 样本
│   │   └── dataset.py        # PyTorch Dataset（加载 .pt，图像统一缩放）
│   ├── models/
│   │   ├── img_encoder.py    # 图像编码器（CNN）
│   │   ├── traj_encoder.py   # 轨迹编码器（LSTM）
│   │   ├── fusion_heads.py   # 投影头 / 融合头
│   │   └── decoders.py       # 轨迹重建解码器、质量得分头
│   ├── losses/
│   │   ├── mae.py            # 重建 MAE 损失
│   │   ├── contrastive.py    # InfoNCE 对比损失
│   │   └── ranking.py        # 排序 Hinge 损失
│   ├── trainers/
│   │   ├── train_stage1.py   # Stage1 训练
│   │   ├── train_stage2.py   # Stage2 训练
│   │   └── train_stage3.py   # Stage3 训练
│   ├── eval/
│   │   ├── eval_stage1.py    # Stage1 评估（Reconstruction MAE）
│   │   ├── eval_stage2.py    # Stage2 评估（Recall@1/5、对齐损失）
│   │   ├── eval_stage3.py    # Stage3 评估（排序准确率、Margin）
│   │   └── visualize.py      # 轨迹与排序可视化
│   └── utils/
│       ├── seed.py           # 随机种子
│       ├── logging.py       # 日志与 metrics 写入
│       └── metrics.py       # 各阶段评估指标计算
│
├── scripts/
│   └── prepare_10_samples.py # 从 dcshuju 随机选 10 对 gnt+csv 到 data/raw
│
└── outputs/
    └── runs/
        └── run_YYYYMMDD_HHMM/
            ├── stage1.yaml, stage2.yaml, stage3.yaml  # 本次运行配置备份
            ├── stage1/          # Stage1 输出
            │   ├── checkpoints/
            │   ├── logs.jsonl
            │   └── metrics.json
            ├── stage2/
            ├── stage3/
            ├── checkpoints/
            └── figures/
```

---

## 数据说明

### 原始数据

- **GNT**：CASIA-HWDB 等使用的离线字形格式。每个文件包含 10 字节头（样本长度、GB2312 字符码、宽、高）与后续灰度像素。`gnt_reader.py` 解析为 NumPy 数组（归一化到 [0,1]）并可从头部解码汉字。
- **在线轨迹 CSV**：表头为 `timestamp,x,y,f`，每行一个采样点。`online_reader.py` 会据此计算 `dt`（时间差）与 `speed`（瞬时速度），输出 5 维轨迹 `(x, y, f, speed, dt)`。

### 配对表 pairs.csv

| 列名       | 说明                          |
|------------|-------------------------------|
| sample_id  | 样本 ID（如 sample_00）        |
| gnt_path   | GNT 文件绝对/相对路径          |
| online_path| 对应在线 CSV 路径              |
| writer_id  | 书写者 ID（可选）              |
| char       | 从 GNT 解析的汉字（可选）      |

### 预处理后的 .pt 样本

每条样本保存为一个 `.pt` 文件，包含：

- `image`：`(1, H, W)` float32，归一化灰度图（Dataset 加载时会统一缩放到 64×64）。
- `traj`：`(N, 5)` float32，列为 `[x, y, f, speed, dt]`。
- `char`：字符串，汉字。
- `sample_id`、`writer_id`：元信息。

---

## 训练阶段说明

| 阶段   | 目标           | 输入           | 输出/损失           | 主要评估指标                |
|--------|----------------|----------------|---------------------|-----------------------------|
| Stage1 | 轨迹遮蔽重建   | 在线轨迹（部分遮蔽） | 重建 (x,y,f)，MAE 损失 | Reconstruction MAE          |
| Stage2 | 跨模态对齐     | 离线图像 + 在线轨迹 | 图像/轨迹嵌入，InfoNCE | Recall@1/5、对齐损失        |
| Stage3 | 质量敏感排序   | 原始轨迹 + 劣化轨迹 | 质量得分，Hinge 排序损失 | Pairwise Rank Accuracy、Margin Mean |
| Stage4 | 少标注微调     | 带质量标签的子集   | 质量评分，MSE/CE   | MAE/RMSE 或 Acc/F1（需自备标签） |

- **Stage1**：使用 LSTM 解码器对遮蔽后的轨迹做序列到序列重建，预测 (x, y, f)。
- **Stage2**：CNN 图像编码器 + LSTM 轨迹编码器，经投影头得到同维嵌入，用 InfoNCE 做图像-轨迹对齐。
- **Stage3**：对轨迹加噪得到“劣化”样本，轨迹编码器 + 得分头，优质得分应高于劣质，使用 Hinge 排序损失。
- **Stage4**：需额外提供约 10% 样本的质量标签（如 `quality_labels.csv`），当前仅提供 `configs/stage4.yaml` 占位，训练脚本可按需扩展。

---

## 快速开始

### 1. 准备原始数据（若尚未准备）

从你的 `dcshuju` 目录随机挑选 10 对 GNT + 在线 CSV 到本项目 `data/raw`：

```bash
python scripts/prepare_10_samples.py
```

脚本会复制 10 对文件到 `data/raw/gnt/` 与 `data/raw/online/`，并生成 `data/raw/pairs.csv`。如需修改数据源路径或样本数，可编辑脚本中的 `DCSHUJU` 与 `random.sample(..., 10)`。

### 2. 一键运行完整流程

在项目根目录执行：

```bash
python run_pipeline.py
```

将依次执行：

1. 根据 `pairs.csv` 构建 `data/processed/samples/` 下的 `.pt` 样本；
2. Stage1 训练（轨迹重建），保存最佳权重与 `metrics.json`；
3. Stage2 训练（跨模态对齐）；
4. Stage3 训练（质量排序）；

每次运行会在 `outputs/runs/` 下新建 `run_YYYYMMDD_HHMM`，其中包含各阶段配置备份、`stage1/`、`stage2/`、`stage3/` 的 checkpoints、`logs.jsonl` 与 `metrics.json`。

### 3. 分阶段单独运行

若只想跑某一阶段或调整超参，可单独调用对应 trainer（需先已构建 `data/processed/samples`）：

```bash
# 仅构建 processed 数据（需在代码中调用 build_processed_dataset，或先跑一遍 run_pipeline）
python -c "
from src.datasets.build_dataset import build_processed_dataset
import os
ROOT = os.path.dirname(os.path.abspath('.'))
build_processed_dataset(os.path.join(ROOT, 'data/raw/pairs.csv'), os.path.join(ROOT, 'data/processed/samples'))
"

# Stage1
python src/trainers/train_stage1.py --config configs/stage1.yaml --run_dir outputs/runs/my_stage1

# Stage2
python src/trainers/train_stage2.py --config configs/stage2.yaml --run_dir outputs/runs/my_stage2

# Stage3
python src/trainers/train_stage3.py --config configs/stage3.yaml --run_dir outputs/runs/my_stage3
```

评估脚本示例（需已训练得到对应 checkpoint）：

```bash
python src/eval/eval_stage1.py --processed_dir data/processed/samples --checkpoint outputs/runs/run_xxx/stage1/checkpoints/best.pt
python src/eval/eval_stage2.py --checkpoint outputs/runs/run_xxx/stage2/checkpoints/best.pt
python src/eval/eval_stage3.py --checkpoint outputs/runs/run_xxx/stage3/checkpoints/best.pt
```

---

## 配置文件说明

- **configs/stage1.yaml**：轨迹维度 `traj_dim`、遮蔽比例 `mask_ratio`、LSTM 层数/隐层、`pred_dim`（预测 x,y,f）、batch_size、epochs、lr 等。
- **configs/stage2.yaml**：图像/轨迹编码器结构、`embed_dim`、InfoNCE 的 `temperature`、训练轮数等。
- **configs/stage3.yaml**：轨迹编码器、劣化噪声 `degrade_noise_scale`、排序 margin、训练轮数等。
- **configs/stage4.yaml**：少标注比例、质量标签路径（需自行实现读取）、损失类型（MSE/CE）等占位。

修改上述 YAML 后，重新运行 `run_pipeline.py` 或对应 trainer 即可生效。

---

## 输出与日志

- **logs.jsonl**：每行一个 JSON，含 epoch、loss、以及该阶段评估指标（如 reconstruction_mae、recall@1、pairwise_rank_acc 等）。
- **metrics.json**：该阶段结束后的汇总指标（如 best Reconstruction MAE、Recall@1、Pairwise Rank Accuracy）。
- **checkpoints/best.pt**：该阶段最佳模型权重，可用于 `eval_stage*.py` 或后续 Stage4 微调。

---

## 注意事项

1. **数据量**：当前示例为 10 对样本，适合验证流程；扩大数据时请更新 `scripts/prepare_10_samples.py` 或直接维护 `pairs.csv`，并保证 `gnt_path` 与 `online_path` 可访问。
2. **图像尺寸**：Dataset 默认将图像缩放到 64×64，可在 `src/datasets/dataset.py` 中修改 `DEFAULT_IMG_SIZE`。
3. **Stage4**：需要质量标签文件（如 sample_id 与分数/等级），目前仅提供配置模板，实现质量回归/分类与 10% 采样逻辑需自行在 `src/trainers/` 中扩展。

---

## 许可与引用

本项目为静态字形与动态在线数据结合分析的大创/实验用途。若使用 CASIA-HWDB 等公开数据集，请遵守其使用条款；若使用 Make Me a Hanzi 等笔顺数据，请遵循对应许可。
