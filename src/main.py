"""メインスクリプト"""

import os
import wandb
from huggingface_hub import login
from src.config.config import (
    OUTPUT_DIR,
    WANDB_PROJECT,
    WANDB_ENTITY
)
from src.models.base_model import create_base_model
from src.models.lora import create_lora_model
from src.models.qlora import create_qlora_model
from src.data.data_processor import load_and_preprocess_data
from src.training.trainer import setup_trainer, setup_wandb_tracking
from src.evaluation.evaluator import evaluate_model, compare_models

def train_lora():
    """LoRAを使用してLLaMA-3.2-1Bをファインチューニング"""
    print("Starting LoRA fine-tuning...")

    # W&B実行の初期化
    wandb.init(
        project=WANDB_PROJECT,
        entity=WANDB_ENTITY,
        name="lora-finetune",
        config={
            "method": "LoRA"
        }
    )

    # モデルとトークナイザーの作成
    model, tokenizer = create_lora_model()

    # データセットの準備
    train_dataset, eval_dataset, data_collator = load_and_preprocess_data(tokenizer)

    # トレーナーのセットアップ
    trainer = setup_trainer(model, train_dataset, eval_dataset, data_collator, mode="lora")
    trainer = setup_wandb_tracking(trainer, eval_dataset, tokenizer, model, mode="lora")

    # トレーニングの実行
    trainer.train()

    # モデルの保存
    model.save_pretrained(f"{OUTPUT_DIR}/lora/final")
    tokenizer.save_pretrained(f"{OUTPUT_DIR}/lora/final")

    return model, tokenizer

def train_qlora():
    """QLoRAを使用してLLaMA-3.2-1Bをファインチューニング"""
    print("Starting QLoRA fine-tuning...")

    # W&B実行の初期化
    wandb.init(
        project=WANDB_PROJECT,
        entity=WANDB_ENTITY,
        name="qlora-finetune",
        config={
            "method": "QLoRA"
        }
    )

    # モデルとトークナイザーの作成
    model, tokenizer = create_qlora_model()

    # データセットの準備
    train_dataset, eval_dataset, data_collator = load_and_preprocess_data(tokenizer)

    # トレーナーのセットアップ
    trainer = setup_trainer(model, train_dataset, eval_dataset, data_collator, mode="qlora")
    trainer = setup_wandb_tracking(trainer, eval_dataset, tokenizer, model, mode="qlora")

    # トレーニングの実行
    trainer.train()

    # モデルの保存
    model.save_pretrained(f"{OUTPUT_DIR}/qlora/final")
    tokenizer.save_pretrained(f"{OUTPUT_DIR}/qlora/final")

    return model, tokenizer

def main():
    """メイン関数"""
    # Hugging Face Hubにログイン
    login()

    # 出力ディレクトリの作成
    os.makedirs(f"{OUTPUT_DIR}/lora", exist_ok=True)
    os.makedirs(f"{OUTPUT_DIR}/qlora", exist_ok=True)

    # ベースモデルの作成
    base_model, base_tokenizer = create_base_model()
    base_accuracy = evaluate_model(base_model, base_tokenizer, mode="base")

    # LoRAでのファインチューニング
    # lora_model, lora_tokenizer = train_lora()
    # lora_accuracy = evaluate_model(lora_model, lora_tokenizer, mode="lora")

    # QLoRAでのファインチューニング
    #qlora_model, qlora_tokenizer = train_qlora()
    #qlora_accuracy = evaluate_model(qlora_model, qlora_tokenizer, mode="qlora")

    # 結果の比較
    # results = compare_models(lora_accuracy, qlora_accuracy)
    print("\n=== Performance Comparison ===")
    print(f"Base Accuracy: {base_accuracy:.4f}")
    # print(f"LoRA Accuracy: {results['lora_accuracy']:.4f}")
    # print(f"QLoRA Accuracy: {results['qlora_accuracy']:.4f}")
    # print(f"Difference: {results['difference']:.4f}")

if __name__ == "__main__":
    main() 