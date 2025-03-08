# LoRA vs QLoRA Comparison

このプロジェクトは、LLaMA-3.2-1Bモデルに対するLoRAとQLoRAの性能比較を行うための実装です。

## 特徴

- LoRAとQLoRAの両方の実装
- SQuADデータセットを使用した質問応答タスク
- Weights & Biasesを使用したトレーニングの追跡
- 詳細な評価メトリクスと比較

## セットアップ

1. 依存関係のインストール:
```bash
pip install -r requirements.txt
```

2. Weights & Biasesのセットアップ:
```bash
wandb login
```

3. Hugging Face Hubのセットアップ:
- Hugging Face Hubのアカウントを作成
- アクセストークンを取得

## 使用方法

```bash
python -m src.main
```

## プロジェクト構成

```
src/
├── config/
│   └── config.py          # 設定値の管理
├── data/
│   └── data_processor.py  # データ前処理関連
├── models/
│   ├── lora.py           # LoRAの実装
│   └── qlora.py          # QLoRAの実装
├── training/
│   ├── trainer.py        # トレーニング関連の共通コード
│   └── callbacks.py      # コールバック実装
├── evaluation/
│   └── evaluator.py      # モデル評価用コード
└── main.py               # メインスクリプト
```

## 結果

トレーニング結果とモデルの比較は、Weights & Biasesのダッシュボードで確認できます。
