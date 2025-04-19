import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from src.config.config import MODEL_NAME

def create_base_model():
    """ベースモデル（ファインチューニングなし）を作成する"""
    # トークナイザーの読み込み
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token

    # モデルの読み込み
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16,
        device_map="auto"
    )

    return model, tokenizer