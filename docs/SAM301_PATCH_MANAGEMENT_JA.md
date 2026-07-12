# SAM301 パッチ管理（trainer 勾配累積 loss スケーリング）

> 言語: [中文](SAM301_PATCH_MANAGEMENT.md) | **日本語**

生成日時: 2026-07-03（日本語版は中文版と同期して維持）

## 1. なぜパッチが必要か

`/home/book/sam301/sam3/train/trainer.py` の `_run_step` は `gradient_accumulation_steps > 1` のとき各 micro-batch に対して直接 `backward(loss)` し、勾配は**総和**であって平均ではない——学習率を変えなければ真の大 batch と等価にならない。パッチ（唯一の変更点）：

```python
backward_loss = loss / accum_steps if accum_steps > 1 else loss
self.scaler.scale(backward_loss).backward()
```

backward への入力のみをスケールし、ログ/isfinite は元の loss を使う；optimizer/scheduler/clipping のタイミングは不変；accum=1 はビット単位で不変。数値等価性は `tests/test_grad_accum_numerics.py` が実物の `_run_step`（float64、許容誤差 1e-12）で継続検証している。

**問題**：`/home/book/sam301` は git リポジトリではなく、再インストール/ソースツリーの再コピーでパッチが静かに消える。そのため git 管理の manifest + 完全 SHA256 + 三層起動ガードで fail-closed に固定する。

## 2. Manifest フィールド（`config/sam301_patch_manifest.json`）

| フィールド | 意味 |
|---|---|
| `patch_id` / `version` | パッチ識別子とバージョン |
| `purpose` | パッチ目的の完全な説明 |
| `target_file` | 対象ファイルの正規絶対パス；loader は `Path.resolve()` 後にそれが `expected_sam3_root` 内にあることを要求し、`/home/book/sam3` を指す target や symlink で逃げる target を拒否する |
| `patch_file` | パッチ diff（book01 ルートからの相対）；解決後は `/home/book/book01/patches` 内にあることが必須で、`patch_file_sha256` により完全性を検証 |
| `original_sha256` | 未パッチ trainer.py の**完全** SHA256（`9c9c4159…248d`、三者独立検証：パッチ前バックアップ、`patch -R` 逆再構築、Codex レビュー記録） |
| `patched_sha256` | パッチ後の**完全** SHA256（`bcf5d8d6…9ec2`、Codex 独立再現） |
| `expected_sam3_root` | 期待される SAM3 import ルート `/home/book/sam301` |

## 3. 4 つのコマンド（全て book01 ルートで実行）

```bash
# 状態：PATCHED / UNPATCHED / UNKNOWN / MISSING
conda run -n sam301 python scripts/manage_sam301_patch.py status
conda run -n sam301 python scripts/manage_sam301_patch.py --json status   # 機械可読

# 検証（正式学習のゲート）：PATCHED のみ exit code 0、それ以外は全て非ゼロ
conda run -n sam301 python scripts/manage_sam301_patch.py verify

# 適用：現在の hash == original の場合のみ許可；先に dry-run、同ディレクトリの一時ファイルに patch、hash == patched を確認後アトミック置換
conda run -n sam301 python scripts/manage_sam301_patch.py apply

# ロールバック：現在の hash == patched の場合のみ許可；同ディレクトリの一時ファイルに逆 patch、hash == original を確認後アトミック置換
conda run -n sam301 python scripts/manage_sam301_patch.py revert
```

**UNKNOWN 状態（hash が既知の 2 値のどちらとも一致しない）での強制上書きは絶対禁止**——apply/revert とも拒否し、このツールに --force はない。この場合は人手でファイルを検査し（保持すべき第三者の変更があるかもしれない）、確認後に既知状態へ手動復旧してから操作すること。

apply/revert は対象ファイルと同ディレクトリのロックファイルを保持し、元ファイルを同ディレクトリの一時ファイルにコピーし、一時ファイルに patch または逆 patch を実行し、最終 SHA256 を検証し、元ファイルの権限を保持し、一時ファイルを fsync し、`os.replace` でアトミック置換し、親ディレクトリを fsync する。検出可能な失敗があれば本物の trainer は決して置換されない。

## 4. 環境再構築後の正しい手順

1. `/home/book/sam301` ソースツリーと `sam301` conda 環境を復旧/再インストール（editable install は `/home/book/sam301` を指す）；
2. `manage_sam301_patch.py status` —— 期待は `UNPATCHED`；
3. `manage_sam301_patch.py apply`；
4. `manage_sam301_patch.py verify` —— exit code 0 必須；
5. `conda run -n sam301 python -m pytest tests/test_grad_accum_numerics.py tests/test_sam301_patch.py -v`；
6. 通常どおり学習プリフライトへ。

## 5. 正式学習前に必ず実行

```bash
conda run -n sam301 python scripts/manage_sam301_patch.py verify   # exit 0 / PATCHED 必須
env -u PYTHONPATH conda run -n sam301 python -c "from pathlib import Path; import sam3; print(Path(sam3.__file__).resolve())"
# /home/book/sam301/sam3/__init__.py を出力すること（/home/book/sam3 を指し返してはならない）
sha256sum /home/book/sam301/sam3/train/trainer.py
# manifest の patched_sha256 と一致すること: bcf5d8d6970ffd8fda609c92b14f9beda475576180262d5e5483523020609ec2
```

手動実行を忘れても、学習チェーンは fail closed する——三層の自動ガード：

| 層 | 位置 | 挙動 |
|---|---|---|
| 1. preflight | `core/training_runner.py::inspect_training_config`（prepare_runtime 時、runtime YAML 書き込み前） | PATCHED 以外 → プリフライト error。YAML を書かず、起動可能な run を作らず、UI は token を発行しない |
| 2. launcher | `ui/training_preflight_page.py::_consume_preflight_for_launch`（token 消費前） | プリフライト後にファイルが置換された → BLOCKED、**token は消費されず**、サブプロセスなし |
| 3. 学習サブプロセス | `scripts/launch_sam3_training.py::run_training`（公式 main 呼び出し前、stdlib 自己検証） | 最終防衛線。hash 不一致なら非ゼロで即終了 |

三層は既存の sam3 import guard（プリフライト + 起動時に import が `/home/book/sam301/sam3/__init__.py` に解決されること、editable binding が `/home/book/sam3` を指し返していないことを検証）と並行して機能する。

起動可能な preflight と正式 launcher は毎回、機械可読の provenance を記録する：

- `provenance.json`
- `dataset_info.json` の `training_provenance`
- `training_config_summary.json` の `training_provenance`
- `training_summary.json` の `training_provenance`

フィールドには book01 の Git commit/dirty 状態、manifest SHA256、patch SHA256、trainer.py SHA256、SAM301 root、sam3 import path、Python executable、runtime YAML SHA256 が含まれる。launcher は起動前にこれらの値を再計算し、preflight の結果を単にコピーしない。
