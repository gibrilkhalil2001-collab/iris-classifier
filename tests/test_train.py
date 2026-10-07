import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from train import run


def test_accuracy_is_at_least_90_percent(tmp_path):
    accuracy = run(test_size=0.2, random_state=42, output_dir=tmp_path)
    assert accuracy >= 0.9
    assert (tmp_path / "confusion_matrix.png").exists()
