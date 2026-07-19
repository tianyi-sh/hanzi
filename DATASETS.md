# 数据说明

本项目依赖“GNT 文件 + 在线轨迹 CSV”的一一配对数据。干净克隆不包含原始数据、processed 张量或模型权重。

## 输入目录契约

`scripts/prepare_data.py` 当前扫描编号 `1` 到 `50`，并匹配：

```text
<数据目录>/
├── 1.gnt
├── 1_online.csv
├── 2.gnt
├── 2_online.csv
└── ...
```

在线 CSV 支持以下表头：

| 字段 | 含义 |
| --- | --- |
| `timestamp` 或 `t` | 时间戳 |
| `x` | 横坐标 |
| `y` | 纵坐标 |
| `f` | 力度 |

轨迹读取后转换为 `[x, y, f, speed, dt]`。当前 `gnt_reader.py` 只解析每个 GNT 文件的首条记录；完整 GNT 语料的逐记录索引和展开不在本仓库实现范围内。只有 GNT、没有对应在线 CSV 的 HWDB 分包不能直接用于当前配对流程。

## 准备命令

在仓库根目录运行：

```powershell
python scripts\prepare_data.py --source-dir "<数据目录>" --sample-count 10 --seed 42
```

也可使用环境变量：

```powershell
$env:HANZI_DATA_DIR = "<数据目录>"
python scripts\prepare_data.py
```

若希望把源数据放在仓库目录内，可使用本地 `data/source/`；该目录已被 Git 忽略。命令行参数优先于环境变量，环境变量优先于 `data/source/`。

## 生成的 manifest

命令会创建 `data/raw/pairs.csv`：

| 字段 | 含义 |
| --- | --- |
| `sample_id` | 仓库内样本 ID |
| `gnt_path` | 相对 `pairs.csv` 的 GNT 路径 |
| `online_path` | 相对 `pairs.csv` 的在线轨迹路径 |
| `writer_id` | 当前准备脚本写入 `0` |
| `char` | 从 GNT 首记录解码的字符 |

数据不足、manifest 字段缺失、样本 ID 重复或输入文件缺失时，脚本会直接报错，避免静默跳过后继续训练。

## 版本控制边界

以下内容只保留在本地，不提交到 Git：

- `data/source/`、`data/raw/`、`data/processed/`
- `legacy/xunlian/data/`
- `outputs/runs/`、`outputs/figures/`
- `*.pt`、`*.pth`、`*.ckpt`
- 大型 GNT、Office/PDF 材料及压缩包

轻量参考指标和日志位于 `outputs/reference/`；可重复生成的示例图已移至 `assets/figures/generated_examples/`。
