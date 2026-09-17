import modal
from experiments.training.train_lm import (
    AdamWConfig,
    CheckpointConfig,
    EvalConfig,
    LogConfig,
    ModelConfig,
    OptimizerConfig,
    ScheduleConfig,
    TrainingConfig,
    WandbConfig,
    train,
)

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
def train_remote():

    train(
        model_cfg=ModelConfig(
            vocab_size=10000,
            context_length=256,
            d_model=512,
            d_ff=1344,
            rope_theta=10000,
            num_layers=4,
            num_heads=16,
            device="cuda",
            dtype="float32",
        ),
        optimizer_cfg=OptimizerConfig(
            optim=AdamWConfig(
                lr=1e-3,
                beta1=0.9,
                beta2=0.999,
                eps=1e-8,
                weight_decay=0.01,
            )
        ),
        train_cfg=TrainingConfig(
            train_data_path="storage/tokenids/ts-train-tokenids.npy",
            val_data_path="storage/tokenids/ts-valid-tokenids.npy",
            max_steps=10000,
            batch_size=128,
            max_l2_norm=1.0,
            seed=42,
        ),
        ckpt_cfg=CheckpointConfig(
            interval=1000,
            out_dir="storage/checkpoints/TinyStories",
        ),
        schedule_cfg=ScheduleConfig(
            max_learning_rate=1e-3,
            min_learning_rate=1e-4,
            warmup_iters=100,
            cosine_cycle_iters=10000,
        ),
        log_cfg=LogConfig(
            log_interval=10,
        ),
        eval_cfg=EvalConfig(
            eval_interval=500,
            eval_batches=20,
        ),
        wandb_cfg=WandbConfig(
            project="CS336-new-assignment1",
            run_name="lr-1e-3",
        ),
    )


@app.local_entrypoint()
def main():
    train_remote.remote()
