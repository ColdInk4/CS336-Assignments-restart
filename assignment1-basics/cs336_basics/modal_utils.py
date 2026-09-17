import modal
from pathlib import PurePosixPath

PROJECT_NAME = "cs336-basics"

app = modal.App(PROJECT_NAME)

volume = modal.Volume.from_name(
    f"{PROJECT_NAME}-data",
    create_if_missing=True,
)


def build_image(*, include_tests: bool = False) -> modal.Image:
    image = (
        modal.Image.debian_slim()
        .apt_install("wget", "gzip")
        .uv_sync()
        .workdir("/root/cs336")
    )

    image = image.add_local_python_source("cs336_basics")

    image = image.add_local_dir(
        "experiments",
        remote_path="/root/cs336/experiments",
    )

    image = image.add_local_file("AGENTS.md", "/root/AGENTS.md")
    image = image.add_local_file("CLAUDE.md", "/root/CLAUDE.md")

    if include_tests:
        image = image.add_local_dir(
            "tests",
            remote_path="/root/tests",
        )

    return image


VOLUME_MOUNTS: dict[
    str | PurePosixPath,
    modal.Volume | modal.CloudBucketMount,
] = {
    "/root/cs336/storage": volume,
}


def secrets(
    include_wandb_secret: bool = False,
    include_huggingface_secret: bool = False,
) -> list[modal.Secret]:
    result: list[modal.Secret] = []

    if include_wandb_secret:
        result.append(modal.Secret.from_name("wandb"))

    if include_huggingface_secret:
        result.append(modal.Secret.from_name("huggingface"))

    return result
