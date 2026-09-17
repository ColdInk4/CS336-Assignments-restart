# cs336_basics/modal_utils.py

from pathlib import Path, PurePosixPath

import modal

PROJECT_NAME = "cs336-basics"

(DATA_PATH := Path("data")).mkdir(exist_ok=True)

app = modal.App(PROJECT_NAME)

user_volume = modal.Volume.from_name(
    f"{PROJECT_NAME}-data",
    create_if_missing=True,
)


def build_image(*, include_tests: bool = False) -> modal.Image:
    image = modal.Image.debian_slim().apt_install("wget", "gzip").uv_sync()

    image = image.add_local_python_source("cs336_basics")
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
    f"/root/{DATA_PATH}": user_volume,
}


def secrets(
    include_huggingface_secret: bool = False,
) -> list[modal.Secret]:
    result: list[modal.Secret] = []

    # 需要的时候再添加你自己的 secret
    if include_huggingface_secret:
        result.append(modal.Secret.from_name("huggingface"))

    return result
