import csv
import math
import struct
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
import yaml

from src.datasets.build_dataset import build_processed_dataset
from src.eval.eval_alignment import eval_alignment
from src.eval.eval_ranking import eval_ranking
from src.eval.eval_reconstruction import eval_reconstruction
from src.trainers.train_stage1 import train_stage1
from src.trainers.train_stage2 import train_stage2
from src.trainers.train_stage3 import train_stage3


class PipelineSmokeTests(unittest.TestCase):
    @staticmethod
    def _write_gnt(path):
        width = height = 32
        pixels = np.zeros((height, width), dtype=np.uint8)
        pixels[4:28, 15:18] = 255
        pixels[15:18, 4:28] = 255
        payload = pixels.tobytes()
        path.write_bytes(
            struct.pack("<I", 10 + len(payload))
            + b"\xb0\xa1"
            + struct.pack("<HH", width, height)
            + payload
        )

    @staticmethod
    def _write_online(path):
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=["timestamp", "x", "y", "f"])
            writer.writeheader()
            for index, (x, y) in enumerate([(1, 5), (3, 5), (5, 5), (5, 3), (5, 1)]):
                writer.writerow({"timestamp": index * 0.1, "x": x, "y": y, "f": 1.0})

    def test_three_stage_pipeline_on_cpu(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            raw = root / "raw"
            raw.mkdir()
            self._write_gnt(raw / "sample.gnt")
            self._write_online(raw / "sample_online.csv")
            pairs = raw / "pairs.csv"
            with pairs.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=["sample_id", "gnt_path", "online_path", "writer_id", "char"],
                )
                writer.writeheader()
                writer.writerow({
                    "sample_id": "sample_00",
                    "gnt_path": "sample.gnt",
                    "online_path": "sample_online.csv",
                    "writer_id": 0,
                    "char": "test",
                })

            samples = root / "processed" / "samples"
            graphs = root / "processed" / "graphs"
            self.assertEqual(build_processed_dataset(pairs, samples, graphs), 1)
            sample = torch.load(samples / "sample_00.pt", map_location="cpu", weights_only=False)
            self.assertEqual(tuple(sample["image"].shape), (1, 224, 224))

            common_model = {
                "traj_encoder": {"hidden_size": 16, "num_layers": 2},
                "struct_encoder": {"hidden_size": 16},
                "align_module": {"embed_dim": 16},
                "decoder": {"pred_dim": 3},
            }
            stage1_config = {
                "data": {"processed_dir": str(samples), "traj_dim": 5},
                "model": common_model,
                "train": {
                    "batch_size": 1,
                    "epochs": 1,
                    "lr": 1.0e-3,
                    "lambda_mae": 1.0,
                    "lambda_align": 0.5,
                },
            }
            stage1_config_path = root / "stage1.yaml"
            stage1_config_path.write_text(yaml.safe_dump(stage1_config), encoding="utf-8")
            stage1_run = root / "stage1"
            (stage1_run / "checkpoints").mkdir(parents=True)
            best_mae = train_stage1(stage1_config_path, stage1_run)
            stage1_checkpoint = stage1_run / "checkpoints" / "best.pt"

            self.assertTrue(math.isfinite(best_mae))
            self.assertTrue(stage1_checkpoint.is_file())
            recon_mae = eval_reconstruction(samples, stage1_checkpoint, hidden_size=16, num_layers=2)
            alignment = eval_alignment(samples, stage1_checkpoint, embed_dim=16, hidden_size=16)
            self.assertTrue(math.isfinite(recon_mae))
            self.assertTrue(math.isfinite(alignment["align_kl"]))
            self.assertTrue(math.isfinite(alignment["align_entropy"]))

            stage2_config = {
                "data": {"processed_dir": str(samples), "traj_dim": 5},
                "model": common_model,
                "train": {
                    "batch_size": 1,
                    "epochs": 1,
                    "lr": 1.0e-3,
                    "lambda_mae": 1.0,
                    "lambda_align": 0.5,
                    "lambda_cons": 0.3,
                },
            }
            stage2_config_path = root / "stage2.yaml"
            stage2_config_path.write_text(yaml.safe_dump(stage2_config), encoding="utf-8")
            stage2_run = root / "stage2"
            (stage2_run / "checkpoints").mkdir(parents=True)
            stage2_mae = train_stage2(stage2_config_path, stage2_run)
            self.assertTrue(math.isfinite(stage2_mae))
            self.assertTrue((stage2_run / "checkpoints" / "best.pt").is_file())

            stage3_config = {
                "data": {
                    "processed_dir": str(samples),
                    "traj_dim": 5,
                    "degrade_noise_scale": 0.3,
                },
                "model": {**common_model, "quality_head": {"hidden": 64}},
                "train": {
                    "batch_size": 1,
                    "epochs": 1,
                    "lr": 1.0e-3,
                    "lambda_mae": 1.0,
                    "lambda_align": 0.5,
                    "lambda_cons": 0.3,
                    "lambda_rank": 0.5,
                    "rank_margin": 0.5,
                },
            }
            stage3_config_path = root / "stage3.yaml"
            stage3_config_path.write_text(yaml.safe_dump(stage3_config), encoding="utf-8")
            stage3_run = root / "stage3"
            (stage3_run / "checkpoints").mkdir(parents=True)
            ranking_accuracy = train_stage3(stage3_config_path, stage3_run)
            stage3_checkpoint = stage3_run / "checkpoints" / "best.pt"
            self.assertTrue(math.isfinite(ranking_accuracy))
            self.assertTrue(stage3_checkpoint.is_file())
            ranking = eval_ranking(
                samples,
                stage3_checkpoint,
                embed_dim=16,
                hidden_size=16,
                noise_scale=0.3,
            )
            self.assertTrue(math.isfinite(ranking["ranking_accuracy"]))
            self.assertTrue(math.isfinite(ranking["margin_mean"]))


if __name__ == "__main__":
    unittest.main()
