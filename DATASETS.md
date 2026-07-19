# 数据位置

项目使用“离线 GNT 字形 + 在线轨迹 CSV”成对数据。为避免重复占用数 GB 空间，新仓库只收纳可复现实验所需的小样本；完整数据仍保留在原位置。

## 当前机器上的数据源

| 用途 | 原位置 | 规模 | 新仓库处理方式 |
| --- | --- | ---: | --- |
| 主实验数据（300 组 GNT/轨迹） | `D:\大创资料\部分实验数据` | 约 6.0 GB | 不复制；通过 `HANZI_DATA_DIR` 使用 |
| HWDB GNT 原始分包 | `D:\大创资料\HWDB1.1trn_gnt_part01` | 约 1.2 GB | 不复制；作为外部原始数据保留 |
| 主工程 10 组样例 | `data/raw/` | 约 210 MB | 已收纳；GNT 使用 Git LFS |
| 已处理样本和结构图 | `data/processed/` | 可重新生成 | 本地保留，Git 忽略 |
| 历史版本数据 | `legacy/xunlian/data/` | 约 210 MB | 本地保留，Git 忽略 |

PowerShell 中指定完整数据源：

```powershell
$env:HANZI_DATA_DIR = 'D:\大创资料\部分实验数据'
python scripts\prepare_data.py
```

也可以显式传参：

```powershell
python scripts\prepare_data.py --source-dir 'D:\大创资料\部分实验数据'
```

不要把 `部分实验数据.zip`、`专利相关代码附readme文件.zip` 或完整 HWDB 数据直接提交到 Git。
