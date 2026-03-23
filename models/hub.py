"""HuggingFace Hub integration for model saving, loading, and pushing.

Provides a mixin class that adds save_pretrained, from_pretrained, and
push_to_hub methods to any nn.Module.
"""

import json
import os
from pathlib import Path

import torch
import torch.nn as nn


CONFIG_NAME = "config.json"
WEIGHTS_NAME = "model.pth"


class HubMixin:
    """Mixin for HuggingFace Hub integration.

    Subclasses must define `_hub_config_keys` — a list of __init__ kwarg
    names to persist in config.json. These are used to reconstruct the
    model architecture when loading.

    Example:
        class MyModel(nn.Module, HubMixin):
            _hub_config_keys = ["n_classes", "growth_rate"]
    """

    _hub_config_keys: list[str] = []

    def _get_config(self) -> dict:
        """Collect config values from the model's current state."""
        config = {"model_type": type(self).__name__}
        for key in self._hub_config_keys:
            if hasattr(self, key):
                config[key] = getattr(self, key)
            elif hasattr(self, f"_{key}"):
                config[key] = getattr(self, f"_{key}")
        return config

    def save_pretrained(self, save_directory: str | Path) -> None:
        """Save model weights and config to a directory.

        Args:
            save_directory: Path to save config.json and model.pth.
        """
        save_directory = Path(save_directory)
        save_directory.mkdir(parents=True, exist_ok=True)

        # Save config
        config = self._get_config()
        with open(save_directory / CONFIG_NAME, "w") as f:
            json.dump(config, f, indent=2)

        # Save weights
        torch.save(self.state_dict(), save_directory / WEIGHTS_NAME)

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str | Path,
        map_location: str = "cpu",
        **kwargs,
    ) -> "HubMixin":
        """Load a model from a local directory or HuggingFace Hub repo.

        Args:
            pretrained_model_name_or_path: Local path or HF Hub repo id
                (e.g. "username/skyscapesnet-dense").
            map_location: Device to map weights to.
            **kwargs: Override config values.

        Returns:
            Model with loaded weights.
        """
        path = Path(pretrained_model_name_or_path)

        if path.is_dir():
            # Local directory
            config_path = path / CONFIG_NAME
            weights_path = path / WEIGHTS_NAME
        else:
            # HuggingFace Hub
            from huggingface_hub import hf_hub_download

            config_path = hf_hub_download(
                repo_id=str(pretrained_model_name_or_path),
                filename=CONFIG_NAME,
            )
            weights_path = hf_hub_download(
                repo_id=str(pretrained_model_name_or_path),
                filename=WEIGHTS_NAME,
            )

        # Load config
        with open(config_path, "r") as f:
            config = json.load(f)

        # Remove model_type from config (not a constructor arg)
        config.pop("model_type", None)

        # Override with any user-provided kwargs
        config.update(kwargs)

        # Instantiate model
        model = cls(**config)

        # Load weights
        state_dict = torch.load(weights_path, map_location=map_location, weights_only=True)
        model.load_state_dict(state_dict)

        return model

    def push_to_hub(
        self,
        repo_id: str,
        commit_message: str = "Upload model",
        private: bool = False,
        token: str | None = None,
    ) -> str:
        """Push model to HuggingFace Hub.

        Args:
            repo_id: HF Hub repo id (e.g. "username/skyscapesnet-dense").
            commit_message: Commit message for the upload.
            private: Whether the repo should be private.
            token: HF API token (uses cached token if None).

        Returns:
            URL of the uploaded repo.
        """
        import tempfile

        from huggingface_hub import HfApi

        api = HfApi(token=token)

        # Create repo if it doesn't exist
        api.create_repo(repo_id=repo_id, exist_ok=True, private=private)

        # Save to temp dir and upload
        with tempfile.TemporaryDirectory() as tmp_dir:
            self.save_pretrained(tmp_dir)
            api.upload_folder(
                repo_id=repo_id,
                folder_path=tmp_dir,
                commit_message=commit_message,
            )

        return f"https://huggingface.co/{repo_id}"
