# 整理记录

本目录从远程仓库 `tianyi-sh/hanzi` 克隆而来，并在不改动原始资料的前提下重新归类。

## 归类原则

- `src/`、`configs/`、`scripts/`：以原 `专利相关代码附readme文件` 为主工程。该目录与远程 README 描述及远程零散文件哈希完全对应。
- `data/`、`outputs/`：保留主工程的样例和历史结果；生成物由 `.gitignore` 排除。
- `legacy/xunlian/`：保存早期多阶段模型版本，避免与当前结构约束模型混用。
- `docs/`：按项目申报、专利、报告、答辩、软著、展板分类。
- `assets/figures/`：集中存放论文和实验图表。
- 完整原始数据不重复复制，位置和使用方法见 `DATASETS.md`。

## 远程根目录文件的恢复位置

| 原远程位置 | 新位置 |
| --- | --- |
| `stage*.yaml`、`struct.yaml` | `configs/` |
| `prepare_data.py` 等工具 | `scripts/` |
| `mae.py`、`align_kl.py` 等损失 | `src/losses/` |
| `sample_*_online.csv`、`pairs.csv` | `data/raw/` |
| `1.gnt`、`1_label.txt`、`1_online.csv` | `data/reference/source/` |
| `logs.jsonl`、`metrics.json` | `outputs/reference/stage3/` |

## 未纳入的内容

- `论文相关/` 是 LIBS 光谱项目，与汉字仓库无关。
- `Origin 2024/`、钢铁/光谱数据、仿真课程作业及个人材料未复制。
- 重复 ZIP/RAR、临时文件和下载中的文件未复制。
