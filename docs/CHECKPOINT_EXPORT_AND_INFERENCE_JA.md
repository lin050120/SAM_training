# Checkpoint エクスポートと推論（日本語版）

> 言語: [English](CHECKPOINT_EXPORT_AND_INFERENCE.md) | [中文](CHECKPOINT_EXPORT_AND_INFERENCE_CN.md) | **日本語**

生成日時: 2026-07-03（日本語版は英語版と同期して維持）

## 3 種類の checkpoint（ファイル名から決して推測しない）

| 種別 | 例 | トップレベル構造 | ロード方法 |
|---|---|---|---|
| **Base** | `/home/book/sam301/sam3.pt` | フラットな dict、ラッパーなし、key は `detector.*` / `tracker.*` 前綴（1156 tensor） | `sam3.model_builder.build_sam3_image_model(checkpoint_path=...)`（sam301 原版、未変更） |
| **Trainer/resume** | `<run_dir>/checkpoints/checkpoint.pt` | `{"model": {...}, "optimizer": {...}, "epoch": int, "loss": {...}, "steps": {...}, "scaler": {...}}` | 推論に直接ロードしてはならない。完全な再開状態を含む。 |
| **Inference** | `<run_dir>/checkpoints/inference_model.pt` | `{"format": "sam3_inference", "format_version": 1, "model": {...}, "metadata": {...}}` | `core.checkpoint_export.load_inference_checkpoint()` |

`core.checkpoint_export.identify_checkpoint(path)` はファイルをロードして実際の key を検査して分類する——拡張子やファイル名では決してない。`Sam3Adapter`（`core/sam3_adapter.py`）は構築時にこれを呼び、正しいロードパスを選ぶか、拒否する。

## trainer checkpoint を推論に直接渡せない理由（根本原因）

`sam301/sam3/model_builder.py::_load_checkpoint()`（本プロジェクトは未変更）は次を行う：

```python
if "model" in ckpt and isinstance(ckpt["model"], dict):
    ckpt = ckpt["model"]
sam3_image_ckpt = {k.replace("detector.", ""): v for k, v in ckpt.items() if "detector" in k}
```

これは **base** checkpoint には正しい（トップレベル key が最初から `detector.` 前綴を持つ）。しかし **trainer** checkpoint の `ckpt["model"]` サブ dict の key は `backbone.vision_backbone.trunk.pos_embed` のような形で、どこにも `"detector"` 部分文字列がない。`trainer.model` は推論と*同じ* `build_sam3_image_model` target で構築され、`.detector` 属性でラップされることがないからだ（`Sam3Image.__init__` は `self.backbone`、`self.transformer` 等を直接設定する）。そのため `if "detector" in k` フィルタは trainer checkpoint の 1134 個の key に**ゼロ**マッチし、`load_state_dict({}, strict=False)` は何もロードせず、モデルはランダム/デフォルト初期化のまま静かに動く——これがまさに、以前の人手受け入れ推論スモークで記録された 4/4 画像ゼロ予測の原因である。

実 run `2026-07-03_14-42-27` に対して実証済み（2026-07-03）：

- 新品の `build_sam3_image_model(checkpoint_path=None).state_dict()`：1134 tensor。
- trainer checkpoint の `ckpt["model"]`：1134 tensor、**keyset 完全一致**、shape 不一致 0。`model.load_state_dict(ckpt["model"], strict=True)` は直接成功——正しいマッピングは**恒等関数**であり、推測ではない。
- base `sam3.pt` の `detector.` 前綴除去後：1156 key（厳密なスーパーセット——純粋な画像検出器には属さない tracker/ビデオ専用バッファが 22 個多い）。

## エクスポート方法

```bash
conda run -n sam301 python scripts/export_sam3_inference_checkpoint.py \
  --input <run_dir>/checkpoints/checkpoint.pt \
  --output <run_dir>/checkpoints/inference_model.pt
```

オプション：`--base-checkpoint <path>`（デフォルト `/home/book/sam301/sam3.pt`、後述の「重みは本当に変わったか」チェックに使用）、`--no-base-diff` でスキップ、`--overwrite` で既存出力ファイルを置換、`--json` で機械可読出力。

決して上書きしない：ソース trainer checkpoint、base checkpoint のパス、既存の出力ファイル（`--overwrite` を除く）。

## Key マッピング規則（`core.checkpoint_export.build_key_mapping`）

- *実際の*ソース key に対して閉じた小さな候補前綴集合（`""`、`"module."`、`"detector."`）を試し、*実際の*ターゲットモデルの `state_dict()` と最も多くマッチするものを選ぶ。
- 推測を拒否：2 つの異なる前綴が同点で**異なる**マッピングを生む場合、恣意的に選ばずエクスポート自体を失敗させる。
- 2 つのソース key が同じターゲット key にマップしてはならない（衝突 → ハードエラー）。
- 全マッピングペアで tensor の shape と dtype を検査；shape 不一致は全てハードエラー（静かに捨てない）。
- `matched_tensors`、`matched_parameters`、`coverage_ratio`、`missing_keys`、`unexpected_keys`、`shape_mismatch`、トップレベルモジュール（`backbone`/`transformer`/`segmentation_head`/`dot_prod_scoring`/`geometry_encoder`）毎のマッチ/総数を報告。

## Fail-loud 規則（エクスポート時）

以下の場合エクスポートは raise する（ファイルを書かない）：

- 入力が **trainer** checkpoint と識別されない。
- `matched_tensors == 0`。
- `coverage_ratio < 0.98`（`MIN_COVERAGE_RATIO`）。
- shape 不一致が 1 つでも存在。
- いずれかのトップレベルモジュールのマッチ数がゼロ。
- 出力パスが既に存在して `--overwrite` がない、またはソース/base checkpoint を上書きしてしまう。
- **エクスポートされる重みが全共通 tensor で base checkpoint とビット単位で同一**——これは「エクスポート」がファインチューニング済みモデルを全く表していないことを意味する。

書き込み後、エクスポートは直ちに、書いたばかりのファイルを**新品の**モデルインスタンス（学習プロセスのモデルオブジェクトではない）へ strict-load して自己検証する——これは任意ではなくスキップ不可。

## strict / coverage 規則（ロード時）

`core.checkpoint_export.load_inference_checkpoint(path)`：

- **trainer** checkpoint を、エクスポートコマンドを示すメッセージ付きで拒否。
- **base** checkpoint を、既存の `build_sam3_image_model(checkpoint_path=...)` パスを示すメッセージ付きで拒否。
- 新品モデル（`build_sam3_image_model(checkpoint_path=None)`）を構築して `model.load_state_dict(state_dict, strict=True)` を呼ぶ——実アーキテクチャではこれが missing/unexpected key **ゼロ**で成功する（検証済み）。`ALLOWED_MISSING_KEYS` / `ALLOWED_UNEXPECTED_KEYS` は、将来の正当なアーキテクチャ変更のための明示的な、現在は空の許可リスト機構——この loader は決して `strict=False` に静かにフォールバックしない。
- `strict=True` が raise しなくても、パラメータをゼロ個ロードした checkpoint を拒否する（多層防御）。

## 実際の受け入れ結果（2026-07-03、run `2026-07-03_14-42-27`）

- `inference_model.pt` をエクスポート、SHA256 `e3aea4edbadc684f7807aed0d981a481f8650e01dff3041df21d29fe09afd222`。
- `matched_tensors=1134`、`matched_parameters=841689398`、`coverage_ratio=1.0`、`missing=[]`、`unexpected=[]`。
- base との比較：385 tensor が変化、749 が同一（backbone 凍結の学習設定と一致——学習されたのは `transformer`/`segmentation_head`/`dot_prod_scoring` のみ）。
- 実スモーク画像 4 枚（通常の背表紙、漫画/複雑柄、傾いた背表紙、薄い/密集棚）：base checkpoint とエクスポート済み inference checkpoint の両方が 4 枚全てで**非空**の予測を出し、両者のスコアと mask 数に実質的な差があった（エクスポート重みが実際に使われ、静かに無視されていない証拠）。生の trainer checkpoint の直接ロードは、推論が始まる前に正しく**拒否**された。
- 完全な数値結果：`runs/review_artifacts/checkpoint_inference_acceptance/checkpoint_inference_acceptance_results.json`（`runs/` 配下、git 管理外）。

完全な受け入れ記録は `docs/P1_REMEDIATION_AND_INFERENCE_ACCEPTANCE.md`。

## よくあるエラー

- `ValueError: ... is a TRAINER checkpoint ... Export it first: ...` —— 推論を `checkpoints/checkpoint.pt` に直接向けた；エクスポーターを実行すること。
- `ValueError: ... is a BASE checkpoint ...` —— inference-checkpoint loader を `sam3.pt` に向けた；そのパスは既存の `build_sam3_image_model(checkpoint_path=...)` 呼び出しで既に機能する。エクスポーター/loader を経由させないこと。
- `coverage ratio ... below the minimum` / `N tensor shape mismatches` / `critical module ... has zero matched tensors` —— trainer checkpoint のモデルが現在の `build_sam3_image_model` アーキテクチャと一致しなくなった（例：SAM3 コードのアップグレード後）；何かを強行する前に調査すること。
- `exported weights are BIT-IDENTICAL to the base checkpoint` —— エクスポートしようとした「trainer」checkpoint は実際には一度も学習されていない（例：optimizer step が 0 回）；これは run 自体の実問題であり、エクスポーターのバグではない。
