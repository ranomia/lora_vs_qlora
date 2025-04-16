"""トレーニング関連の共通コード"""

import wandb
from transformers import TrainingArguments, Trainer
from src.config.config import (
    OUTPUT_DIR,
    BATCH_SIZE,
    EPOCHS,
    LEARNING_RATE
)
from src.training.callbacks import WandbCallback

def setup_trainer(model, train_dataset, eval_dataset, data_collator, mode="lora"):
    """トレーナーのセットアップを行う"""
    # トレーニング引数の設定
    training_args = TrainingArguments(
        output_dir=f"{OUTPUT_DIR}/{mode}",
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        evaluation_strategy="steps",
        eval_steps=500,
        do_eval=True,
        save_strategy="steps",
        save_steps=500,
        num_train_epochs=EPOCHS,
        learning_rate=LEARNING_RATE,
        fp16=True,
        logging_dir=f"{OUTPUT_DIR}/{mode}/logs",
        logging_steps=100,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        report_to="none",
    )

    # トレーナーの作成
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
    )

    return trainer

def setup_wandb_tracking(trainer, eval_dataset, tokenizer, model, mode="lora"):
    """W&Bトラッキングのセットアップを行う"""
    # W&Bコールバックの追加
    wandb_callback = WandbCallback(
        trainer=trainer,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        model=model,
        log_prefix=f"{mode}/"
    )

    # トレーニングハンドラーを上書き
    original_train = trainer.train
    original_evaluate = trainer.evaluate

    def train_with_wandb(*args, **kwargs):
        result = original_train(*args, **kwargs)
        return result

    def evaluate_with_wandb(*args, **kwargs):
        metrics = original_evaluate(*args, **kwargs)
        wandb_callback.on_evaluate(trainer.args, trainer.state, None, metrics)
        return metrics

    trainer.train = train_with_wandb
    trainer.evaluate = evaluate_with_wandb

    # カスタムコールバックの設定
    original_step_end = trainer._maybe_log_save_evaluate

    def step_end_with_wandb(*args, **kwargs):
        result = original_step_end(*args, **kwargs)
        wandb_callback.on_step_end(trainer.args, trainer.state, None)
        return result

    trainer._maybe_log_save_evaluate = step_end_with_wandb

    return trainer 