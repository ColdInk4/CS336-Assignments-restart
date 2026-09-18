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


def train_batch(
    lr: float,
    batch_size: int = 128,
):

    TOTAL_TOKEN = 327680000
    context_length = 256
    max_steps = TOTAL_TOKEN // (batch_size * context_length)

    train(
        model_cfg=ModelConfig(
            vocab_size=10000,
            context_length=context_length,
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
                lr=lr,
                beta1=0.9,
                beta2=0.999,
                eps=1e-8,
                weight_decay=0.01,
            )
        ),
        train_cfg=TrainingConfig(
            train_data_path="storage/tokenids/ts-train-tokenids.npy",
            val_data_path="storage/tokenids/ts-valid-tokenids.npy",
            max_steps=max_steps,
            batch_size=batch_size,
            max_l2_norm=1.0,
            seed=42,
        ),
        ckpt_cfg=CheckpointConfig(
            interval=max_steps // 10,
            out_dir="storage/checkpoints/TinyStories",
        ),
        schedule_cfg=ScheduleConfig(
            max_learning_rate=lr,
            min_learning_rate=lr * 0.1,
            warmup_iters=100,
            cosine_cycle_iters=max_steps,
        ),
        log_cfg=LogConfig(
            log_interval=max_steps // 1000,
        ),
        eval_cfg=EvalConfig(
            eval_interval=max_steps // 20,
            eval_batches=20,
        ),
        wandb_cfg=WandbConfig(
            project="CS336-new-assignment1",
            run_name=f"lr-{lr:g}-steps-{max_steps}-batchsize-{batch_size}",
        ),
    )


def main():
    for batch_size in [64, 256]:
        train_batch(3e-3, batch_size)


if __name__ == "__main__":
    main()
