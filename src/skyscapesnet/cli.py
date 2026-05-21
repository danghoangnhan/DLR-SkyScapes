"""LightningCLI entry point for `skyscapesnet` command."""
from lightning.pytorch.cli import LightningCLI

from skyscapesnet.models.lightning_module import SkyScapesLitModule
from skyscapesnet.data.datamodule import SkyScapesDataModule


def main():
    LightningCLI(
        SkyScapesLitModule,
        SkyScapesDataModule,
        save_config_kwargs={"overwrite": True},
    )


if __name__ == "__main__":
    main()
