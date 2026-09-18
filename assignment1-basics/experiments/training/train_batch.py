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
    TOTAL_TOKEN = 327_680_000
    context_length = 256
    tokens_per_step = batch_size * context_length

    max_steps = TOTAL_TOKEN // tokens_per_step

    WARMUP_TOKENS = TOTAL_TOKEN // 100  # 1% of training tokens
    LOG_TOKENS = TOTAL_TOKEN // 1000  # 0.1%
    EVAL_TOKENS = TOTAL_TOKEN // 20  # 5%
    CHECKPOINT_TOKENS = TOTAL_TOKEN // 10  # 10%

    warmup_iters = max(1, WARMUP_TOKENS // tokens_per_step)
    log_interval = max(1, LOG_TOKENS // tokens_per_step)
    eval_interval = max(1, EVAL_TOKENS // tokens_per_step)
    checkpoint_interval = max(1, CHECKPOINT_TOKENS // tokens_per_step)
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
            interval=checkpoint_interval,
            out_dir=(
                f"storage/checkpoints/TinyStories"
                f"/batch-{batch_size}"
                f"-lr-{lr:g}"
                f"-tokens-{TOTAL_TOKEN // 1_000_000}M"
            ),
        ),
        schedule_cfg=ScheduleConfig(
            max_learning_rate=lr,
            min_learning_rate=lr * 0.1,
            warmup_iters=warmup_iters,
            cosine_cycle_iters=max_steps,
        ),
        log_cfg=LogConfig(
            log_interval=log_interval,
        ),
        eval_cfg=EvalConfig(
            eval_interval=eval_interval,
            eval_batches=20,
        ),
        wandb_cfg=WandbConfig(
            project="CS336-new-assignment1",
            run_name=(
                f"batch-{batch_size}"
                f"-lr-{lr:g}"
                f"-tokens-{TOTAL_TOKEN // 1_000_000}M"
            ),
        ),
    )
