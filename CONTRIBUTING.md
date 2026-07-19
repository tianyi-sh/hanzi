# 参与开发

## 环境准备

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## 开发流程

1. 从 `main` 创建功能分支。
2. 不要提交原始数据、模型权重、运行目录或 Office/PDF 材料。
3. 修改数据路径时使用参数或环境变量，不要写入个人电脑绝对路径。
4. 提交前运行：

```powershell
python -m compileall -q run_pipeline.py scripts src tests
python -m unittest discover -s tests -v
python -m pip check
```

5. 在 Pull Request 中说明变更目的、数据影响和验证结果。
