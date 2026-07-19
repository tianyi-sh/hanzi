# -*- coding: utf-8 -*-
import os
import json
from datetime import datetime


def log_metrics(log_path, metrics_dict, step=None):
    os.makedirs(os.path.dirname(log_path) or ".", exist_ok=True)
    with open(log_path, "a", encoding="utf-8") as f:
        rec = {"ts": datetime.now().isoformat(), **metrics_dict}
        if step is not None:
            rec["step"] = step
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def save_metrics_json(path, metrics_dict):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(metrics_dict, f, ensure_ascii=False, indent=2)
