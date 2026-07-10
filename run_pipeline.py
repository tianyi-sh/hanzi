# -*- coding: utf-8 -*-
"""一键准备数据、构建结构图，并依次完成三阶段训练。"""

import os
import shutil
import sys
from datetime import datetime

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)


def main():
    run_name = datetime.now().strftime("run_%Y%m%d_%H%M")
    run_dir = os.path.join(ROOT, "outputs", "runs", run_name)
    os.makedirs(run_dir, exist_ok=True)
    os.makedirs(os.path.join(run_dir, "checkpoints"), exist_ok=True)
    os.makedirs(os.path.join(run_dir, "figures"), exist_ok=True)
    for config_name in ["struct.yaml", "stage1.yaml", "stage2.yaml", "stage3.yaml"]:
        source = os.path.join(ROOT, "configs", config_name)
        if os.path.isfile(source):
            shutil.copy2(source, os.path.join(run_dir, config_name))

    pairs_csv = os.path.join(ROOT, "data", "raw", "pairs.csv")
    if not os.path.isfile(pairs_csv):
        print("未找到 pairs.csv，正在调用 scripts/prepare_data.py。")
        from scripts.prepare_data import main as prepare_data

        prepare_data()

    from src.datasets.build_dataset import build_processed_dataset

    processed_dir = os.path.join(ROOT, "data", "processed", "samples")
    struct_dir = os.path.join(ROOT, "data", "processed", "struct_graphs")
    count = build_processed_dataset(pairs_csv, processed_dir, struct_dir)
    print("Processed samples:", count)

    from src.trainers.train_stage1 import train_stage1
    from src.trainers.train_stage2 import train_stage2
    from src.trainers.train_stage3 import train_stage3

    stage1_dir = os.path.join(run_dir, "stage1")
    os.makedirs(os.path.join(stage1_dir, "checkpoints"), exist_ok=True)
    mae = train_stage1(os.path.join(ROOT, "configs", "stage1.yaml"), stage1_dir)
    print("Stage1 best Recon MAE:", mae)

    stage2_dir = os.path.join(run_dir, "stage2")
    os.makedirs(os.path.join(stage2_dir, "checkpoints"), exist_ok=True)
    train_stage2(os.path.join(ROOT, "configs", "stage2.yaml"), stage2_dir)
    print("Stage2 done.")

    stage3_dir = os.path.join(run_dir, "stage3")
    os.makedirs(os.path.join(stage3_dir, "checkpoints"), exist_ok=True)
    accuracy = train_stage3(os.path.join(ROOT, "configs", "stage3.yaml"), stage3_dir)
    print("Stage3 best Ranking accuracy:", accuracy)
    print("Run saved to:", run_dir)


if __name__ == "__main__":
    main()
