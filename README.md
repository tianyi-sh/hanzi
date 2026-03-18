# 结构约束自监督汉字书写质量模型

基于 **README_for_Cursor_结构约束自监督项目规范** 实现的完整工程：离线 GNT 字形 + 在线轨迹 (t,x,y,f)，无人工质量标签，通过结构约束自监督三阶段训练与质量敏感排序，输出 Recon MAE、Align KL、Ranking 等指标及结构热力图、轨迹异常可视化。

---

## 一、总体目标

- **数据**：离线 GNT 字形图像；在线 0~3 秒、20Hz、(t, x, y, f)。
- **流程**：自动构建字形结构图 G_S → 在线轨迹预处理 → 轨迹与结构空间对齐 → 软覆盖先验 π(i,k) → 语义对齐分布 a(i,k) → 三阶段训练（L_mae + L_align；+ L_cons；+ L_rank）→ 质量敏感排序 → 可解释可视化。

---

## 二、项目目录结构

```
xunlian2/
├── data/
│   ├── raw/
│   │   ├── gnt/
│   │   ├── online/
│   │   └── pairs.csv
│   └── processed/
│       ├── samples/
│       └── struct_graphs/
├── configs/
│   ├── struct.yaml
│   ├── stage1.yaml
│   ├── stage2.yaml
│   └── stage3.yaml
├── src/
│   ├── datasets/
│   │   ├── gnt_reader.py
│   │   ├── online_reader.py
│   │   ├── struct_builder.py
│   │   ├── align_utils.py
│   │   ├── build_dataset.py
│   │   └── dataset.py
│   ├── models/
│   │   ├── traj_encoder.py
│   │   ├── struct_encoder.py
│   │   ├── align_module.py
│   │   ├── decoder.py
│   │   └── quality_head.py
│   ├── losses/
│   │   ├── mae.py
│   │   ├── align_kl.py
│   │   ├── consistency.py
│   │   └── ranking.py
│   ├── trainers/
│   │   ├── train_stage1.py
│   │   ├── train_stage2.py
│   │   └── train_stage3.py
│   ├── eval/
│   │   ├── eval_reconstruction.py
│   │   ├── eval_alignment.py
│   │   ├── eval_ranking.py
│   │   ├── visualize_alignment.py
│   │   └── visualize_quality.py
│   └── utils/
│       ├── skeleton.py
│       ├── geometry.py
│       └── graph_ops.py
├── outputs/
│   └── runs/
├── scripts/
│   └── prepare_data.py
├── run_pipeline.py
└── requirements.txt
```

---

## 三、数据处理规范

1. **GNT 解析**（`gnt_reader.py`）  
   - 输出：`img` tensor `[1, 224, 224]`  
   - 灰度化 + 二值化（Otsu/阈值）+ 归一化

2. **结构图构建**（`struct_builder.py` + `utils/skeleton.py`, `graph_ops.py`）  
   - 二值图 → 骨架化（skeletonization）  
   - 提取端点与分叉点，构建结构边  
   - 输出：`G_S = { nodes: [N,2], edges: list_of_edges }`，并保存至 `struct_graphs/`

3. **在线轨迹处理**（`online_reader.py`）  
   - 输入：`(t, x, y, f)`  
   - 输出：`traj = [x, y, f, speed, dt]`，shape `[T, 5]`

---

## 四、对齐与结构建模

1. **空间对齐**（`align_utils.py`）  
   - 轨迹 bbox 缩放，映射到结构图坐标空间（与 224×224 图像一致）

2. **软覆盖先验**  
   - `d(i,k)` = 轨迹点到结构边 k 的平均距离  
   - `π(i,k) = softmax(-d/σ)`

3. **语义对齐分布**  
   - `a(i,k) = softmax(sim(z_S,i, z_G,k))`，在 `align_module` 中实现

---

## 五、三阶段训练流程

- **Stage 1**：`L = L_mae + λ1 L_align`  
- **Stage 2**：`L = L_mae + λ1 L_align + λ2 L_cons`  
- **Stage 3**：`L = L_mae + λ1 L_align + λ2 L_cons + λ3 L_rank`

损失定义：

- `L_align = KL(π || a)`  
- `L_cons = ||z_S - Σ_k a(i,k) z_G,k||²`  
- `L_rank = max(0, m - (s⁺ - s⁻))`

---

## 六、输出与评估

- **Recon MAE**  
- **Align KL**、**Align entropy**  
- **Ranking accuracy**、**Margin mean**  
- 结构热力图（`visualize_alignment.py`）  
- 轨迹异常/劣化对比（`visualize_quality.py`）  
- 每次运行：固定随机种子、保存 `config`、`best` checkpoint、`metrics.json`、`logs.jsonl`

---

## 七、快速开始

```bash
cd D:\大创资料\xunlian2
pip install -r requirements.txt
python scripts\prepare_data.py
python run_pipeline.py
```

- `prepare_data.py`：从 `D:\大创资料\dcshuju` 随机取 10 对 gnt + online csv 到 `data/raw`，生成 `pairs.csv`。  
- `run_pipeline.py`：构建 processed 与 struct_graphs → Stage1 → Stage2 → Stage3，结果在 `outputs/runs/run_YYYYMMDD_HHMM/`。

单独评估示例：

```bash
python src/eval/eval_reconstruction.py --checkpoint outputs/runs/run_xxx/stage1/checkpoints/best.pt
python src/eval/eval_alignment.py --checkpoint outputs/runs/run_xxx/stage1/checkpoints/best.pt
python src/eval/eval_ranking.py --checkpoint outputs/runs/run_xxx/stage3/checkpoints/best.pt
```

---

## 八、工程约束

- 固定随机种子（`utils/seed.py`）  
- 每次训练将当前 config 复制到 run 目录  
- 各阶段保存 best checkpoint  
- 输出 `metrics.json`、`logs.jsonl`
