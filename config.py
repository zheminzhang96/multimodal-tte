from pathlib import Path


class Config:
    """Default output locations and model dimensions for the public 2D workflow."""

    BASE_OUTPUT_DIR = Path("outputs/checkpoints")
    LOG_DIR = Path("outputs/tensorboard_logs")

    DIMS = {
        "MII": {"img": 1024},
        "CLMBR": {"ehr": 768},
        "ONEHOT": {"ehr": 128},
    }

    @staticmethod
    def get_checkpoint_dir(task: str, img_model: str, ehr_model: str) -> Path:
        path = Config.BASE_OUTPUT_DIR / task / f"ckpt_{img_model}_{ehr_model}"
        path.mkdir(parents=True, exist_ok=True)
        return path
