# 数据目录

训练数据不直接存入 Git 仓库。

- `raw/`：由 `scripts/prepare_data.py` 从外部数据源生成。
- `processed/`：由 `run_pipeline.py` 生成的张量样本和结构图。
- `reference/`：本地保存的原始上传样本，仅用于追溯。

完整数据位置、环境变量和准备命令见项目根目录的 [DATASETS.md](../DATASETS.md)。
