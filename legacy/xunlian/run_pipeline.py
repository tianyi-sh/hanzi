# -*- coding: utf-8 -*-
"""
一键运行：构建 processed 数据 -> Stage1 -> Stage2 -> Stage3
在项目根目录执行: python run_pipeline.py
"""
import os
import sys
from datetime import datetime

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

def main():
    run_name = datetime.now().strftime("run_%Y%m%d_%H%M")
    runs_dir = os.path.join(ROOT, "outputs", "runs")
    run_dir = os.path.join(runs_dir, run_name)
    os.makedirs(run_dir, exist_ok=True)
    os.makedirs(os.path.join(run_dir, "checkpoints"), exist_ok=True)
    os.makedirs(os.path.join(run_dir, "figures"), exist_ok=True)
    import shutil
    for c in ["stage1.yaml", "stage2.yaml", "stage3.yaml"]:
        src = os.path.join(ROOT, "configs", c)
        if os.path.isfile(src):
            shutil.copy2(src, os.path.join(run_dir, c))

    # 1. 构建 processed 数据
    from src.datasets.build_dataset import build_processed_dataset
    pairs_csv = os.path.join(ROOT, "data", "raw", "pairs.csv")
    processed_dir = os.path.join(ROOT, "data", "processed", "samples")
    n = build_processed_dataset(pairs_csv, processed_dir)
    print("Processed samples:", n)

    # 2. Stage 1
    from src.trainers.train_stage1 import train_stage1
    s1_dir = os.path.join(run_dir, "stage1")
    os.makedirs(os.path.join(s1_dir, "checkpoints"), exist_ok=True)
    cfg1 = os.path.join(ROOT, "configs", "stage1.yaml")
    mae = train_stage1(cfg1, s1_dir)
    print("Stage1 best Reconstruction MAE:", mae)

    # 3. Stage 2
    from src.trainers.train_stage2 import train_stage2
    s2_dir = os.path.join(run_dir, "stage2")
    os.makedirs(os.path.join(s2_dir, "checkpoints"), exist_ok=True)
    cfg2 = os.path.join(ROOT, "configs", "stage2.yaml")
    r1 = train_stage2(cfg2, s2_dir)
    print("Stage2 best Recall@1:", r1)

    # 4. Stage 3
    from src.trainers.train_stage3 import train_stage3
    s3_dir = os.path.join(run_dir, "stage3")
    os.makedirs(os.path.join(s3_dir, "checkpoints"), exist_ok=True)
    cfg3 = os.path.join(ROOT, "configs", "stage3.yaml")
    acc = train_stage3(cfg3, s3_dir)
    print("Stage3 best Pairwise Rank Accuracy:", acc)

    print("Run saved to:", run_dir)


if __name__ == "__main__":
    main()
