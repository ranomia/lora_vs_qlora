"""トレーニングコールバックモジュール"""

import torch
import wandb

class WandbCallback:
    """W&Bへのログ記録を行うカスタムコールバック"""
    def __init__(self, trainer, eval_dataset, tokenizer, model, log_prefix=""):
        self.trainer = trainer
        self.eval_dataset = eval_dataset
        self.tokenizer = tokenizer
        self.model = model
        self.log_prefix = log_prefix

    def on_step_end(self, args, state, control, **kwargs):
        """各ステップ終了時にログを記録"""
        # 学習メトリクスをログ
        if state.global_step % 100 == 0:
            wandb.log({
                f"{self.log_prefix}train/loss": state.log_history[-1].get("loss", 0),
                f"{self.log_prefix}train/learning_rate": state.log_history[-1].get("learning_rate", 0),
                "global_step": state.global_step
            })

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        """評価時にメトリクスをログ"""
        if metrics:
            # 評価メトリクスをログ
            eval_metrics = {f"{self.log_prefix}eval/{k}": v for k, v in metrics.items()}
            wandb.log(eval_metrics)

            # サンプル予測をログ（最初の5つのサンプルのみ）
            self.log_sample_predictions(num_samples=5)

    def log_sample_predictions(self, num_samples=5):
        """サンプル予測結果をログ"""
        samples = []
        for i in range(min(num_samples, len(self.eval_dataset))):
            # サンプルの取得とテンソルへの変換
            sample = self.eval_dataset[i]
            if isinstance(sample["input_ids"], list):
                input_ids = torch.tensor(sample["input_ids"]).unsqueeze(0).to(self.model.device)
            else:
                input_ids = sample["input_ids"].unsqueeze(0).to(self.model.device)

            # 予測生成
            with torch.no_grad():
                outputs = self.model.generate(
                    input_ids,
                    max_new_tokens=50,
                    num_beams=2,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id
                )

            # 入力と出力をデコード
            input_text = self.tokenizer.decode(input_ids[0], skip_special_tokens=True)
            generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

            # 入力を取り除いて予測部分のみを抽出
            prediction = generated_text.replace(input_text, "").strip()

            samples.append({
                "input": input_text,
                "prediction": prediction
            })

        # サンプル予測をテーブルとしてログ
        predictions_table = wandb.Table(columns=["input", "prediction"])
        for sample in samples:
            predictions_table.add_data(sample["input"], sample["prediction"])

        wandb.log({f"{self.log_prefix}sample_predictions": predictions_table}) 