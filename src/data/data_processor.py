"""データ前処理モジュール"""

from datasets import load_dataset
from transformers import DataCollatorForLanguageModeling
from src.data.config.config import MAX_LENGTH

def preprocess_squad_data(examples, tokenizer):
    """SQuADデータセットの前処理関数"""
    prompts = []
    targets = []

    for context, question, answers in zip(examples["context"], examples["question"], examples["answers"]):
        prompt = f"コンテキスト: {context}\n\n質問: {question}\n\n回答:"
        prompts.append(prompt)

        # 最初の回答を選択
        if len(answers["text"]) > 0:
            targets.append(answers["text"][0])
        else:
            targets.append("")

    # プロンプトとターゲットを結合してエンコード
    inputs = tokenizer(
        prompts,
        text_target=targets,
        max_length=MAX_LENGTH,
        truncation=True,
        padding="max_length",
        return_tensors="pt"
    )

    return inputs

def load_and_preprocess_data(tokenizer):
    """データセットの読み込みと前処理を行う"""
    dataset = load_dataset("squad")

    # 訓練データセットの前処理
    train_dataset = dataset["train"].map(
        lambda examples: preprocess_squad_data(examples, tokenizer),
        batched=True,
        remove_columns=dataset["train"].column_names
    )

    # 評価データセットの前処理
    eval_dataset = dataset["validation"].map(
        lambda examples: preprocess_squad_data(examples, tokenizer),
        batched=True,
        remove_columns=dataset["validation"].column_names
    )

    # データコレーターの作成
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False
    )

    return train_dataset, eval_dataset, data_collator 