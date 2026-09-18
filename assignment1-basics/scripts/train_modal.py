from experiments.training.train_batch import train_batch

from cs336_basics.modal_utils import (
    app,
    build_image,
    VOLUME_MOUNTS,
    secrets,
)


@app.function(
    image=build_image(),
    gpu="H100",
    volumes=VOLUME_MOUNTS,
    secrets=secrets(include_wandb_secret=True),
    timeout=60 * 60 * 6,
)
def train_remote(
    lr: float,
    batch_size: int = 128,
):
    train_batch(lr, batch_size)


@app.local_entrypoint()
def main():
    for batch_size in [64, 256]:
        train_remote.remote(3e-3, batch_size)
