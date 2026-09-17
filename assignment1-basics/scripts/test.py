import os
import modal

from cs336_basics.modal_utils import app, build_image, secrets


@app.function(
    image=build_image(),
    secrets=secrets(include_wandb_secret=True),
)
def test_wandb():
    key = os.environ.get("WANDB_API_KEY")
    print("WANDB_API_KEY exists:", key is not None)
    print("WANDB_API_KEY length:", len(key) if key else 0)
