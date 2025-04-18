"""設定値の管理モジュール"""

import os

# W&B設定
WANDB_PROJECT = "lora_vs_qlora"
WANDB_ENTITY = "ranomia-team"

# モデル設定
MODEL_NAME = "meta-llama/Llama-3.2-1B"
OUTPUT_DIR = "./results"
MAX_LENGTH = 512
BATCH_SIZE = 16
EPOCHS = 3
LEARNING_RATE = 2e-4

# LoRAの設定
LORA_R = 8
LORA_ALPHA = 16
LORA_DROPOUT = 0.05

# 出力ディレクトリの作成
os.makedirs(f"{OUTPUT_DIR}/lora", exist_ok=True)
os.makedirs(f"{OUTPUT_DIR}/qlora", exist_ok=True) 