import csv
import tempfile
import unittest
from pathlib import Path

from scripts.prepare_data import main as prepare_data


class PrepareDataTests(unittest.TestCase):
    @staticmethod
    def _make_source(root, count):
        source = root / "source"
        source.mkdir()
        for index in range(1, count + 1):
            (source / f"{index}.gnt").write_bytes(
                bytes([index, index, index, index, 0xB0, 0xA1])
            )
            (source / f"{index}_online.csv").write_text(
                "timestamp,x,y,f\n0.0,1.0,1.0,0.5\n",
                encoding="utf-8",
            )
        return source

    def test_sampling_is_deterministic_and_manifest_paths_are_relative(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = self._make_source(root, 3)
            output_a = root / "output_a"
            output_b = root / "output_b"
            rows_a = prepare_data(source, output_a, sample_count=2, seed=7)
            rows_b = prepare_data(source, output_b, sample_count=2, seed=7)

            self.assertEqual(len(rows_a), 2)
            self.assertEqual(
                [(output_a / row["gnt_path"]).read_bytes() for row in rows_a],
                [(output_b / row["gnt_path"]).read_bytes() for row in rows_b],
            )
            with (output_a / "pairs.csv").open(encoding="utf-8") as handle:
                manifest = list(csv.DictReader(handle))
            self.assertTrue(all(not Path(row["gnt_path"]).is_absolute() for row in manifest))
            self.assertTrue(all(not Path(row["online_path"]).is_absolute() for row in manifest))

    def test_insufficient_pairs_fail_fast(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = self._make_source(root, 1)
            with self.assertRaisesRegex(ValueError, "少于请求"):
                prepare_data(source, root / "output", sample_count=2)


if __name__ == "__main__":
    unittest.main()
