import modal

from experiments.inference.generate import (
    CkptConfig,
    GenerateConfig,
    ModelConfig,
    TokenizerConfig,
    main as generate,
)

from cs336_basics.modal_utils import (
    app,
    build_image,
    VOLUME_MOUNTS,
)


@app.function(
    image=build_image(),
    gpu="H100",
    volumes=VOLUME_MOUNTS,
    timeout=60 * 10,
)
def generate_remote(prompt: str):
    generate(
        ckpt_cfg=CkptConfig(
            path="storage/checkpoints/TinyStories/ckpt_0010000_final.pt",
        ),
        model_cfg=ModelConfig(
            device="cuda",
            dtype="float32",
        ),
        tokenizer_cfg=TokenizerConfig(
            vocab_path="storage/tokenizer/TinyStoriesV2-GPT4-vocab.txt",
            merges_path="storage/tokenizer/TinyStoriesV2-GPT4-merges.txt",
            special=["<|endoftext|>"],
        ),
        generate_cfg=GenerateConfig(
            temperature=1.0,
            max_new_tokens=500,
            top_p=0.2,
        ),
        prompt=prompt,
    )


@app.local_entrypoint()
def main():
    generate_remote.remote("Once upon a time, there was a little girl named Lily.")
