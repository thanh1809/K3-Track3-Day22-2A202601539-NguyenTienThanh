from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class TrainingConfig:
    method: str
    beta: float = 0.1
    lambda_orpo: float = 0.1
    max_length: int = 512
    batch_size: int = 2

class PreferenceTrainer:
    """Interface for DPO/ORPO training implementations."""
    def __init__(self, config: TrainingConfig, output_dir: str | Path = "outputs") -> None:
        self.config = config
        self.output_dir = Path(output_dir)

    def train(self) -> None:
        """Run a CPU-safe mock training step and persist explicit metadata.

        This lab starter avoids pulling large model dependencies by default.
        The artifact records which alignment objective would be used by a
        TRL-backed implementation.
        """
        if self.config.method not in {"dpo", "orpo", "mock"}:
            raise ValueError("method must be one of: dpo, orpo, mock")

        self.output_dir.mkdir(parents=True, exist_ok=True)
        artifact = {
            "method": self.config.method,
            "beta": self.config.beta,
            "lambda_orpo": self.config.lambda_orpo,
            "max_length": self.config.max_length,
            "batch_size": self.config.batch_size,
            "status": "mock_trainer_completed",
        }
        (self.output_dir / "training_artifact.json").write_text(
            json.dumps(artifact, indent=2, sort_keys=True),
            encoding="utf-8",
        )
