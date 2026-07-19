# 项目贡献与创新说明

本文档说明本项目解决的问题、已经完成的工作、核心创新机制及其代码证据。所有表述均以当前仓库实现为准。

## 1. 问题与目标

汉字书写质量同时包含静态字形结构和动态书写过程。离线 GNT 提供最终字形，在线轨迹提供时间、位置和力度，但两种数据没有天然的逐点对应关系；同时，高质量人工评分标签通常难以大规模获得。

本项目的目标是在不依赖大规模人工质量标签的前提下，把离线字形结构作为监督信号，引导在线轨迹表示学习，并进一步学习原始书写与退化书写之间的相对质量顺序。

## 2. 我完成的工作

### 2.1 数据与结构表示

- 实现 GNT 首记录解析、灰度归一化和二值化，输出统一的 `1 × 224 × 224` 字形张量。
- 将在线 CSV 转换为 `[x, y, f, speed, dt]` 五维时序特征。
- 对字形进行骨架化，提取端点、分叉点和结构边，形成 `G_S = {nodes, edges}`。
- 将轨迹缩放到字形坐标空间，并生成相对路径的配对 manifest，保证数据流程可迁移、可复现。

对应实现：[`gnt_reader.py`](../src/datasets/gnt_reader.py)、[`online_reader.py`](../src/datasets/online_reader.py)、[`struct_builder.py`](../src/datasets/struct_builder.py)、[`prepare_data.py`](../scripts/prepare_data.py)。

### 2.2 结构约束自监督学习

- 根据轨迹点到每条结构边的平均距离构造几何软覆盖先验 `π(k)`。
- 分别编码轨迹序列与结构边，在共同嵌入空间计算语义对齐分布 `a(t,k)`。
- 使用 `KL(π || mean_t a)` 约束学习到的对齐分布。
- 使用 `||z_traj - a · z_struct||²` 约束轨迹表示能够由其对齐的结构边解释。

对应实现：[`align_utils.py`](../src/datasets/align_utils.py)、[`align_module.py`](../src/models/align_module.py)、[`align_kl.py`](../src/losses/align_kl.py)、[`consistency.py`](../src/losses/consistency.py)。

### 2.3 无人工评分的质量排序

- 对原始轨迹加入可控随机扰动，构造自监督的“原始—退化”样本对。
- 使用共享轨迹编码器和质量得分头得到 `s_good` 与 `s_bad`。
- 通过 `max(0, margin - (s_good - s_bad))` 学习相对质量顺序。
- 输出 pairwise ranking accuracy 与 margin mean，用于检查排序行为。

该机制学习的是相对排序，不等同于人工专家绝对评分。对应实现：[`train_stage3.py`](../src/trainers/train_stage3.py)、[`quality_head.py`](../src/models/quality_head.py)、[`ranking.py`](../src/losses/ranking.py)。

### 2.4 实验与工程闭环

- Stage 1：轨迹重建与结构对齐。
- Stage 2：在前述目标上加入结构一致性。
- Stage 3：进一步加入质量排序。
- 每阶段保存配置、日志、指标和最佳检查点，并提供重建、对齐、排序和可视化入口。
- 提供确定性数据抽样、输入校验、CPU 三阶段冒烟测试和 GitHub Actions 持续集成。

对应实现：[`run_pipeline.py`](../run_pipeline.py)、[`src/trainers/`](../src/trainers/)、[`src/eval/`](../src/eval/)、[`tests/test_pipeline_smoke.py`](../tests/test_pipeline_smoke.py)。

## 3. 核心创新机制

| 创新点 | 传统困难 | 本项目处理方式 | 可验证输出 |
| --- | --- | --- | --- |
| 字形结构图约束 | 像素相似不能直接表达端点、分叉和结构关系 | 从骨架中抽取节点与边，以结构单元参与训练 | 结构图 `.npz`、骨架与节点图 |
| 几何—语义双重对齐 | 离线字形与在线轨迹没有逐点配准 | 几何距离产生先验，嵌入相似度产生可学习对齐分布 | Align KL、Align entropy、对齐图 |
| 自监督质量排序 | 人工书写质量分数稀缺且主观 | 自动生成退化轨迹，以成对排序代替绝对分数监督 | Ranking accuracy、Margin mean |
| 分阶段多目标验证 | 多种损失混合后难以定位贡献 | 按重建、结构一致性、质量排序逐阶段增加目标并分别记录 | 各阶段 checkpoint、metrics、logs |

## 4. 当前证据与边界

仓库测试已经使用合成 GNT 与在线轨迹完成三阶段 CPU 训练、检查点保存和评估；已有图像展示了字形结构提取、轨迹退化和对齐过程。

当前仓库没有随附完整公开数据集、预训练权重或可直接复现的论文基准表，也没有验证 GPU/CUDA 路径。因此，本项目可以证明方法链路已经实现并可运行，但不在缺少统一公开实验设置时声明领先性能。
