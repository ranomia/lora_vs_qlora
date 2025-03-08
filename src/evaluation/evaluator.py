"""モデル評価モジュール"""

import torch
import wandb
from datasets import load_dataset
from ..config.config import WANDB_PROJECT, WANDB_ENTITY

def evaluate_model(model, tokenizer, mode="lora"):
    """ファインチューニングされたモデルの評価"""
    print(f"Evaluating {mode} model...")

    # W&B実行の初期化
    wandb.init(
        project=WANDB_PROJECT,
        entity=WANDB_ENTITY,
        name=f"{mode}-evaluation",
        config={
            "method": mode,
            "evaluation_samples": 100
        }
    )

    # 評価データセットの読み込み
    eval_dataset = load_dataset("squad", split="validation")

    correct = 0
    total = 0
    eval_results = []

    for i in range(min(100, len(eval_dataset))):  # 最初の100サンプルのみ評価
        example = eval_dataset[i]
        context = example["context"]
        question = example["question"]
        ground_truth = example["answers"]["text"][0] if len(example["answers"]["text"]) > 0 else ""

        # プロンプトの作成
        prompt = f"コンテキスト: {context}\n\n質問: {question}\n\n回答:"

        # トークナイズ
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        # 生成時間の計測
        start_time = torch.cuda.Event(enable_timing=True)
        end_time = torch.cuda.Event(enable_timing=True)

        start_time.record()
        # 生成
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=50,
                num_beams=4,
                early_stopping=True
            )
        end_time.record()

        # 時間計測の同期
        torch.cuda.synchronize()

        # ミリ秒から秒に変換
        inference_time = start_time.elapsed_time(end_time) / 1000

        # デコード
        predicted_answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
        predicted_answer = predicted_answer.replace(prompt, "").strip()

        # 評価
        is_correct = ground_truth.lower() in predicted_answer.lower()
        if is_correct:
            correct += 1
        total += 1

        # 結果の格納
        eval_results.append({
            "id": i,
            "context": context[:100] + "...",  # 長いコンテキストを省略
            "question": question,
            "ground_truth": ground_truth,
            "prediction": predicted_answer,
            "is_correct": is_correct,
            "inference_time": inference_time
        })

        if i % 10 == 0:
            print(f"Progress: {i}/{min(100, len(eval_dataset))}")
            # 中間結果のログ
            current_accuracy = correct / (i + 1)
            wandb.log({
                f"{mode}/current_accuracy": current_accuracy,
                f"{mode}/samples_evaluated": i + 1
            })

    # 最終精度の計算
    accuracy = correct / total
    print(f"{mode} Accuracy: {accuracy:.4f}")

    # 結果をW&Bにログ
    wandb.log({
        f"{mode}/final_accuracy": accuracy,
        f"{mode}/total_samples": total
    })

    # 評価結果のテーブルを作成
    results_table = wandb.Table(columns=["id", "question", "ground_truth", "prediction", "is_correct", "inference_time"])
    for result in eval_results:
        results_table.add_data(
            result["id"],
            result["question"],
            result["ground_truth"],
            result["prediction"],
            result["is_correct"],
            result["inference_time"]
        )

    # テーブルをログ
    wandb.log({f"{mode}/evaluation_results": results_table})

    # 推論時間の統計をログ
    inference_times = [result["inference_time"] for result in eval_results]
    wandb.log({
        f"{mode}/avg_inference_time": sum(inference_times) / len(inference_times),
        f"{mode}/max_inference_time": max(inference_times),
        f"{mode}/min_inference_time": min(inference_times)
    })

    return accuracy

def compare_models(lora_accuracy, qlora_accuracy):
    """LoRAとQLoRAの性能比較"""
    # 最終比較結果をW&Bにログ
    wandb.init(
        project=WANDB_PROJECT,
        entity=WANDB_ENTITY,
        name="lora-qlora-comparison",
        config={
            "lora_accuracy": lora_accuracy,
            "qlora_accuracy": qlora_accuracy,
            "accuracy_difference": abs(lora_accuracy - qlora_accuracy)
        }
    )

    # 比較グラフ用のデータ
    comparison_data = [
        ["LoRA", lora_accuracy],
        ["QLoRA", qlora_accuracy]
    ]

    # 比較テーブルの作成とログ
    comparison_table = wandb.Table(columns=["method", "accuracy"])
    for row in comparison_data:
        comparison_table.add_data(row[0], row[1])

    # テーブルをログ
    wandb.log({"comparison_results": comparison_table})

    return {
        "lora_accuracy": lora_accuracy,
        "qlora_accuracy": qlora_accuracy,
        "difference": abs(lora_accuracy - qlora_accuracy)
    } 