import csv
import struct
import tempfile
import unittest
from pathlib import Path

import numpy as np

from src.datasets.gnt_reader import read_gnt
from src.datasets.online_reader import read_online_csv


class ReaderTests(unittest.TestCase):
    def test_read_gnt_returns_normalized_tensor_and_metadata(self):
        width = 16
        height = 16
        pixels = np.zeros((height, width), dtype=np.uint8)
        pixels[4:12, 7:9] = 255
        pixels[7:9, 4:12] = 255
        tag = "啊".encode("gb2312")
        payload = pixels.tobytes()
        record_size = 10 + len(payload)

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "sample.gnt"
            path.write_bytes(
                struct.pack("<I", record_size)
                + tag
                + struct.pack("<HH", width, height)
                + payload
            )
            image, metadata = read_gnt(path)

        self.assertEqual(tuple(image.shape), (1, 224, 224))
        self.assertEqual(metadata["width"], width)
        self.assertEqual(metadata["height"], height)
        self.assertEqual(metadata["char"], "啊")
        self.assertGreater(float(image.max()), 0.0)
        self.assertGreaterEqual(float(image.min()), 0.0)
        self.assertLessEqual(float(image.max()), 1.0)

    def test_read_online_csv_builds_motion_features(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "sample_online.csv"
            with path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=["timestamp", "x", "y", "f"])
                writer.writeheader()
                writer.writerows(
                    [
                        {"timestamp": 0.0, "x": 0.0, "y": 0.0, "f": 0.5},
                        {"timestamp": 0.1, "x": 1.0, "y": 0.0, "f": 0.6},
                        {"timestamp": 0.2, "x": 1.0, "y": 1.0, "f": 0.7},
                    ]
                )
            trajectory = read_online_csv(path)

        self.assertEqual(trajectory.shape, (3, 5))
        self.assertAlmostEqual(float(trajectory[1, 4]), 0.1, places=5)
        self.assertGreater(float(trajectory[1, 3]), 0.0)


if __name__ == "__main__":
    unittest.main()
