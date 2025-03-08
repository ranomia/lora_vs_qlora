"""LoRAモデルの実装"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import get_peft_model, LoraConfig, TaskType
from ..config.config import (
    MODEL_NAME,
    LORA_R,
    LORA_ALPHA,
    LORA_DROPOUT
)

def create_lora_model():
    """LoRAモデルを作成する"""
    # トークナイザーの読み込み
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token

    # モデルの読み込み
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16,
        device_map="auto"
    )

    # LoRA設定
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        bias="none",
    )

    # LoRAモデルの作成
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    return model, tokenizer 