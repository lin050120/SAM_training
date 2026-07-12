"""UI language switch (中文 / 日本語).

Design: pages are written with Chinese strings as the source of truth. After the
whole Blocks tree is built, wire_language_switch() walks every component once,
snapshots any translatable attribute (label / info / placeholder, Markdown and
Button values, Radio/Dropdown choice labels, Tab labels) whose exact (stripped)
Chinese text appears in the JA table, and wires the top language radio to emit
gr.update() for all of them. Strings missing from the table simply stay Chinese —
a missed translation is visible but can never break the app.

Runtime-generated texts (status boxes, JSON outputs, log tails) are data, not
chrome; they are deliberately not translated here.
"""

from __future__ import annotations

from typing import Any, Callable

import gradio as gr

from ui.cvat_page import DISCLAIMER as _CVAT_DISCLAIMER
from ui.history_page import INFERENCE_RUNS_ROOT as _INFERENCE_RUNS_ROOT
from ui.training_preflight_page import (
    BANNER as _BANNER,
    STAGE_A_NOTE as _STAGE_A_NOTE,
    STAGE_B_NOTE as _STAGE_B_NOTE,
)

LANGUAGE_CHOICES = [("中文", "zh"), ("日本語", "ja")]
DEFAULT_LANGUAGE = "zh"

JA: dict[str, str] = {
    # ---- app.py: header and tabs ----
    "# SAM3 Fine-tuning 工具\n"
    "阶段 D1 本地 Web UI：把现有命令行流程可视化，不重写推理/NMS/COCO/CVAT/训练预检逻辑。":
        "# SAM3 Fine-tuning ツール\n"
        "ステージ D1 ローカル Web UI：既存のコマンドラインフローを可視化するもので、"
        "推論/NMS/COCO/CVAT/学習プリフライトのロジックは再実装しません。",
    "推理任务配置": "推論タスク設定",
    "历史运行记录": "実行履歴",
    "结果查看": "結果表示",
    "CVAT 导出": "CVAT エクスポート",
    "训练预检": "学習プリフライト",
    "数据集登记": "データセット登録",
    "Checkpoint 评估": "Checkpoint 評価",

    # ---- inference page ----
    "调用现有 `scripts/run_unified_inference.py`，不重写推理/NMS/COCO 逻辑。"
    "device=cuda 且 CUDA 不可用时会拒绝启动，不做静默 CPU 回退。":
        "既存の `scripts/run_unified_inference.py` を呼び出します。推論/NMS/COCO の"
        "ロジックは再実装しません。device=cuda で CUDA が利用できない場合は起動を拒否し、"
        "暗黙の CPU フォールバックは行いません。",
    "### 模型权重选择": "### モデル重みの選択",
    "模型来源": "モデルソース",
    "默认原始 SAM3": "デフォルトのオリジナル SAM3",
    "已发现 inference models": "検出済み inference models",
    "手动输入目录和文件名": "ディレクトリとファイル名を手動入力",
    "手动输入完整绝对路径": "完全な絶対パスを手動入力",
    "刷新模型列表": "モデル一覧を更新",
    "检查模型": "モデルを検査",
    "已发现模型": "検出済みモデル",
    "模型目录": "モデルディレクトリ",
    "模型文件名": "モデルファイル名",
    "完整绝对路径": "完全な絶対パス",
    "模型检查结果": "モデル検査結果",
    "当前选中模型信息": "現在選択中のモデル情報",
    "limit (留空 = 处理全部图片)": "limit（空欄 = 全画像を処理）",
    "开始运行": "実行開始",
    "停止": "停止",
    "最终命令": "最終コマンド",
    "任务状态": "タスク状態",

    # ---- history page ----
    f"读取 `{_INFERENCE_RUNS_ROOT}` 下的推理 run。优先读取 run_config.json / manifest.json / "
    "errors.json / acceptance_report.json / cvat validation report；旧 run 缺字段时显示 "
    "unknown/unavailable，不会导致页面崩溃。":
        f"`{_INFERENCE_RUNS_ROOT}` 配下の推論 run を読み込みます。run_config.json / "
        "manifest.json / errors.json / acceptance_report.json / cvat validation report を"
        "優先的に読み、古い run でフィールドが欠けている場合は unknown/unavailable と"
        "表示します（ページはクラッシュしません）。",
    "刷新": "更新",

    # ---- results page ----
    "选择 run 和图片查看原图、raw/NMS 可视化、实例信息、NMS 删除详情、manifest、run_config 和各类报告。"
    "本阶段不支持逐像素编辑/拆分/合并/手绘 mask。":
        "run と画像を選択して、元画像、raw/NMS 可視化、インスタンス情報、NMS 除去の詳細、"
        "manifest、run_config、各種レポートを表示します。このステージではピクセル単位の"
        "編集/分割/結合/手描き mask はサポートしません。",
    "刷新 run 列表": "run 一覧を更新",
    "原图": "元画像",
    "raw/NMS 可视化": "raw/NMS 可視化",
    "实例信息 (来自 COCO + NMS npz)": "インスタンス情報（COCO + NMS npz 由来）",
    "### 附加报告": "### 追加レポート",

    # ---- CVAT page ----
    _CVAT_DISCLAIMER.strip():
        "**Polygon**: CVAT 互換フローで使用できます。エクスポート時に最終 NMS mask から"
        "約 8 点の低頂点 polygon を再生成するため、CVAT 上で点をドラッグして境界を修正する"
        "用途に適しています。これは非可逆変換であり、ピクセル単位の厳密な評価には向きません。"
        "ZIP 生成時の `annotations/instances_default.json` もこの約 8 点 polygon です。\n\n"
        "**RLE**: 最終 NMS bool mask とピクセル単位で一致します。本プロジェクトのローカル検証では "
        "19/19 exact（実 run `2026-07-02_12-28-40` に対して）。現行バージョンの CVAT への"
        "**実際の手動インポート**はユーザーによる確認待ちであり、このページは「RLE は CVAT への"
        "インポートに成功した」といった結論を表示しません。\n\n"
        "このページは既存の inference run に対する CVAT パッケージの再エクスポート/検証のみを行い、"
        "SAM3 を再実行することはありません。",
    "生成 ZIP": "ZIP を生成",
    "计算 polygon fidelity": "polygon fidelity を計算",
    "生成 NMS pair review": "NMS pair review を生成",
    "NMS kept/removed pair (自动来自 manifest，无需手填 ID)":
        "NMS kept/removed pair（manifest から自動取得、ID の手入力は不要）",
    "该 run 已有的 NMS review 信息": "この run の既存 NMS review 情報",
    "导出 / 重新导出": "エクスポート / 再エクスポート",
    "导出结果 (images/annotations/errors/exact_match/IoU/zip_path)":
        "エクスポート結果 (images/annotations/errors/exact_match/IoU/zip_path)",

    # ---- training preflight page ----
    _BANNER: "## 先に学習プリフライトを完了する必要があります。プリフライト合格後にのみ学習を開始できます。",
    _STAGE_A_NOTE:
        "空欄の数値フィールドは権威ベース YAML の値にフォールバックし、0・NaN・空文字列は"
        "書き込みません。`learning_rate` は `scratch.lr_transformer`（学習可能な "
        "transformer/decoder の学習率）に対応します。"
        "`scratch.lr_vision_backbone`/`scratch.lr_language_backbone` はベース YAML の設計どおり "
        "0.0 に固定凍結されており、このページでは上書きできません。"
        "`num_workers` は学習用 dataloader（`scratch.num_train_workers`）のみを上書きし、"
        "検証用はベース YAML の値（現在 0、検証セットは小さい）を使い続けます。",
    _STAGE_B_NOTE:
        "プリフライト合格、runtime YAML/checkpoint/データが全て存在、他にアクティブな学習タスクが"
        "ない、CUDA が利用可能、かつユーザーが確認チェックボックスをオンにした場合にのみ、"
        "学習サブプロセスを実際に起動します。",
    "**数据身份**：预检会按当前 train/val COCO 路径查询 dataset registry。未注册或未经人工审核的"
    "数据集只能用于 smoke test，不构成正式微调效果证据。详见 `docs/E3_DATASET_IDENTITY_ERRATUM.md`。":
        "**データ識別**：プリフライトは現在の train/val COCO パスで dataset registry を照会します。"
        "未登録または人手レビューされていないデータセットは smoke test にのみ使用でき、正式な"
        "ファインチューニング効果の証拠にはなりません。詳細は `docs/E3_DATASET_IDENTITY_ERRATUM.md`。",
    "### 数据集自动划分: 标注数据 → train/val，test 数据 → test":
        "### データセット自動分割: アノテーションデータ → train/val、test データ → test",
    "输入两个已标注 COCO 数据文件夹：标注数据文件夹会随机切分为 train/val，"
    "test 数据文件夹会整体写入 test。默认 val ratio=0.10，约为标注数据总量的 1/10。":
        "アノテーション済み COCO データフォルダを 2 つ指定します。アノテーションデータフォルダは"
        "ランダムに train/val に分割され、test データフォルダは丸ごと test に書き込まれます。"
        "デフォルトの val ratio=0.10 は、アノテーションデータ全体の約 1/10 です。",
    "标注数据文件夹 (自动切 train/val)": "アノテーションデータフォルダ（自動で train/val に分割）",
    "test 数据文件夹 (全部进入 test)": "test データフォルダ（全て test に入る）",
    "输出数据集目录": "出力データセットディレクトリ",
    "允许覆盖输出目录 (旧目录会先改名为 backup)":
        "出力ディレクトリの上書きを許可（旧ディレクトリは先に backup にリネーム）",
    "自动生成 train/val/test 数据集": "train/val/test データセットを自動生成",
    "数据集划分结果": "データセット分割結果",
    "生成的 test images": "生成された test images",
    "生成的 test COCO": "生成された test COCO",
    "### 阶段 A: 生成并验证训练配置": "### ステージ A: 学習設定の生成と検証",
    "max_epochs (留空=基础YAML)": "max_epochs（空欄=ベース YAML）",
    "train batch size (留空=基础YAML)": "train batch size（空欄=ベース YAML）",
    "gradient accumulation steps (留空=基础YAML)": "gradient accumulation steps（空欄=ベース YAML）",
    "learning rate (留空=基础YAML)": "learning rate（空欄=ベース YAML）",
    "num_workers (留空=基础YAML, 只作用于训练集)": "num_workers（空欄=ベース YAML、学習セットのみに適用）",
    "smoke (max_epochs<=1, 未审核数据默认)": "smoke（max_epochs<=1、未レビューデータのデフォルト）",
    "formal (需要人工审核 GT)": "formal（人手レビュー済み GT が必要）",
    "运行训练预检 (不会启动训练)": "学習プリフライトを実行（学習は開始しません）",
    "预检状态": "プリフライト状態",
    "预检结果 (resolved paths / max_epochs / batch size / gradient accumulation / learning rate / "
    "num_workers / effective batch size / prompt / runtime YAML / 最终训练命令)":
        "プリフライト結果 (resolved paths / max_epochs / batch size / gradient accumulation / "
        "learning rate / num_workers / effective batch size / prompt / runtime YAML / 最終学習コマンド)",
    "### 阶段 B: 启动训练": "### ステージ B: 学習開始",
    "我确认这将启动 GPU 训练任务。": "これが GPU 学習タスクを開始することを確認しました。",
    "启动训练": "学習開始",
    "停止训练": "学習停止",
    "训练任务状态": "学習タスク状態",
    "监控信息 (run_id/pid/pgid/started_at/elapsed/exit_code/output_dir/checkpoint_dir/"
    "discovered checkpoints/epoch/loss/lr/gpu memory — 指标解析为 best-effort, 解析不到时显示 unavailable)":
        "モニタリング情報 (run_id/pid/pgid/started_at/elapsed/exit_code/output_dir/checkpoint_dir/"
        "discovered checkpoints/epoch/loss/lr/gpu memory — 指標解析は best-effort、"
        "解析できない場合は unavailable と表示)",
    "training_summary.json (训练结束后生成)": "training_summary.json（学習終了後に生成）",

    # ---- dataset registry page ----
    "登记已经人工审核的数据集。登记后，训练预检会按 train/val COCO 路径匹配 registry；"
    "只有 `allowed_for_formal_training=true` 的登记数据集才能运行 formal 或多 epoch 训练。":
        "人手レビュー済みのデータセットを登録します。登録後、学習プリフライトは train/val COCO の"
        "パスで registry を照合します。`allowed_for_formal_training=true` で登録された"
        "データセットのみ formal または複数 epoch の学習を実行できます。",
    "annotation source evidence（必填：写清谁在什么时候审核了哪些内容，不要用模板句）":
        "annotation source evidence（必須：誰がいつ何をレビューしたかを具体的に書く。テンプレ文は不可）",
    "例: 2026-07-12 由 <姓名> 在 CVAT 中逐张检查并修正了 train/val 全部 52 张图的所有实例边界":
        "例: 2026-07-12 に <氏名> が CVAT で train/val 全 52 枚の全インスタンス境界を"
        "1 枚ずつ確認・修正した",
    "覆盖同 dataset_id 的既有登记和 manifest": "同じ dataset_id の既存登録と manifest を上書きする",
    "预览登记内容": "登録内容をプレビュー",
    "写入登记": "登録を書き込む",
    "登记预览 / 结果": "登録プレビュー / 結果",

    # ---- checkpoint evaluation page ----
    "## Checkpoint 评估：Validation 选最佳，Test 仅作诊断对照":
        "## Checkpoint 評価：Validation で最良を選択、Test は診断用の比較のみ",
    "评价指标为 mask 级（Mean IoU / Boundary F1 / 漏检 / 误检 / 面积偏差），"
    "不使用 bbox AP，也不默认最后一个 epoch 最好。验证集必须已在 "
    "`data_manifests/dataset_identity_registry.json` 登记为人工修正 GT，否则评价会被服务端阻止。"
    "Test 结果不会改变 validation 选出的 best checkpoint。"
    "详见 `docs/CHECKPOINT_EVALUATION_CN.md`。":
        "評価指標は mask レベル（Mean IoU / Boundary F1 / 見逃し / 誤検出 / 面積偏差）です。"
        "bbox AP は使わず、最後の epoch が最良だという前提も置きません。検証セットは "
        "`data_manifests/dataset_identity_registry.json` に人手修正 GT として登録済みである"
        "必要があり、未登録の場合サーバー側で評価がブロックされます。"
        "Test の結果が validation で選ばれた best checkpoint を変えることはありません。"
        "詳細は `docs/CHECKPOINT_EVALUATION_CN.md`。",
    "训练 run（自动列出含 checkpoint 的 run）": "学習 run（checkpoint を含む run を自動リスト）",
    "刷新状态/结果": "状態/結果を更新",
    "依次评价 Validation 和 Test": "Validation と Test を順に評価",
    "重新评价 Validation 和 Test（忽略缓存）": "Validation と Test を再評価（キャッシュ無視）",
    "导出最佳推理模型": "最良の推論モデルをエクスポート",
    "评价 Validation 全部 Checkpoint": "Validation で全 Checkpoint を評価",
    "重新评价 Validation": "Validation を再評価",
    "评价 Test 全部 Checkpoint": "Test で全 Checkpoint を評価",
    "重新评价 Test": "Test を再評価",
    "选择 run 后点击“刷新状态/结果”。": "run を選択して「状態/結果を更新」をクリックしてください。",
    "### Validation 排名（唯一用于 best selection）":
        "### Validation ランキング（best selection に使われる唯一の根拠）",
    "### Test 结果（diagnostic only，不参与 best selection）":
        "### Test 結果（diagnostic only、best selection には関与しない）",
    "评价进度": "評価進捗",
    "评价日志 (stdout/stderr)": "評価ログ (stdout/stderr)",
    "### 手动导出任意 Trainer Checkpoint": "### 任意の Trainer Checkpoint を手動エクスポート",
    "Run 目录": "Run ディレクトリ",
    "刷新 Checkpoint 列表": "Checkpoint 一覧を更新",
    "输出目录": "出力ディレクトリ",
    "输出文件名": "出力ファイル名",
    "允许覆盖已有输出（谨慎）": "既存出力の上書きを許可（注意）",
    "选中 checkpoint 详情": "選択中 checkpoint の詳細",
    "导出所选 Checkpoint": "選択した Checkpoint をエクスポート",
    "导出状态和日志": "エクスポート状態とログ",
    "导出模型 metadata / 检查结果": "エクスポートモデル metadata / 検査結果",
}

# Normalize keys once: match on stripped text so surrounding whitespace in
# triple-quoted constants can never cause a silent miss.
_JA_NORMALIZED = {key.strip(): value for key, value in JA.items()}

_TEXT_ATTRS = ("label", "info", "placeholder")
# Only static display components get their VALUE translated. Textbox/Code values
# are runtime data and must never be overwritten by a language switch.
_VALUE_TYPES = (gr.Markdown, gr.Button)


def _ja_text(text: Any) -> str | None:
    if not isinstance(text, str):
        return None
    return _JA_NORMALIZED.get(text.strip())


def build_language_radio() -> gr.Radio:
    return gr.Radio(
        choices=LANGUAGE_CHOICES,
        value=DEFAULT_LANGUAGE,
        label="Language / 语言 / 言語",
        container=False,
        scale=0,
    )


def wire_language_switch(demo: gr.Blocks, lang_radio: gr.Radio) -> Callable[[str], list[dict]]:
    """Snapshot all translatable components in `demo` and wire `lang_radio`.

    Must be called inside the Blocks context, after every page has been built.
    Returns the switch function (exposed for tests).
    """
    tracked: list[tuple[Any, dict[str, tuple[Any, Any]]]] = []
    for block in demo.blocks.values():
        if block is lang_radio:
            continue
        attrs: dict[str, tuple[Any, Any]] = {}
        for attr in _TEXT_ATTRS:
            original = getattr(block, attr, None)
            translated = _ja_text(original)
            if translated is not None:
                attrs[attr] = (original, translated)
        if isinstance(block, _VALUE_TYPES):
            original = getattr(block, "value", None)
            translated = _ja_text(original)
            if translated is not None:
                attrs["value"] = (original, translated)
        choices = getattr(block, "choices", None)
        if choices and any(_ja_text(label) is not None for label, _ in choices):
            ja_choices = [(_ja_text(label) or label, value) for label, value in choices]
            attrs["choices"] = (list(choices), ja_choices)
        if attrs:
            tracked.append((block, attrs))

    def switch(language: str) -> list[dict]:
        use_ja = language == "ja"
        return [
            gr.update(**{attr: pair[1] if use_ja else pair[0] for attr, pair in attrs.items()})
            for _, attrs in tracked
        ]

    lang_radio.change(
        fn=switch,
        inputs=[lang_radio],
        outputs=[block for block, _ in tracked],
        show_progress="hidden",
    )
    # Exposed for tests: lets them exercise the switch without re-entering the
    # Blocks context or double-wiring the event.
    demo.i18n_switch = switch
    demo.i18n_tracked = tracked
    return switch
