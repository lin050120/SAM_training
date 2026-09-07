from __future__ import annotations

from collections import Counter
from datetime import datetime

import gradio as gr
from PIL import Image, ImageDraw

from core.config import BOOK_ROOT
from core.manual_dataset_split import (
    add_sources,
    assign,
    export_dataset,
    restore_plan,
    split_spanning_groups,
)

PAGE_SIZE = 24


def _view(state, query, split_filter, page):
    records = state.get("records", [])
    filtered = [r for r in records if (split_filter == "all" or r["split"] == split_filter)
                and query.casefold() in (r["path"] + " " + r["group"]).casefold()]
    pages = max(1, (len(filtered) + PAGE_SIZE - 1) // PAGE_SIZE)
    page = min(max(1, int(page or 1)), pages)
    shown = filtered[(page - 1) * PAGE_SIZE:page * PAGE_SIZE]
    gallery, choices = [], []
    for n, r in enumerate(shown, 1):
        try:
            with Image.open(r["path"]) as im:
                thumb = im.convert("RGB")
                thumb.thumbnail((420, 280))
        except OSError as exc:
            # The source tree is not owned by this page; a file can move between
            # loading and browsing. Show the gap instead of failing the gallery.
            thumb = Image.new("RGB", (420, 280), "#3a1f1f")
            ImageDraw.Draw(thumb).text((12, 12), f"读取失败 / 読み込み失敗\n{exc}")
            gallery.append((thumb, f"{n}. [{r['split']}] {r['image']['file_name']} · 文件不可读"))
            if not r["locked"]:
                choices.append((f"{n}. {r['image']['file_name']} · 文件不可读", r["key"]))
            continue
        # Draw annotations in thumbnail coordinates so the user can inspect grouping context.
        draw = ImageDraw.Draw(thumb)
        sx, sy = thumb.width / r["image"]["width"], thumb.height / r["image"]["height"]
        for ann in r["annotations"]:
            seg = ann.get("segmentation")
            if isinstance(seg, list):
                for poly in seg:
                    if len(poly) >= 6:
                        points = [(poly[i] * sx, poly[i + 1] * sy) for i in range(0, len(poly), 2)]
                        draw.line(points + points[:1], fill="#00ff80", width=1)
            elif "bbox" in ann:
                x, y, w, h = ann["bbox"]
                draw.rectangle((x*sx, y*sy, (x+w)*sx, (y+h)*sy), outline="#00ff80")
        label = f"{n}. [{r['split']}] {r['image']['file_name']} · {r['group']} · {len(r['annotations'])}标注"
        gallery.append((thumb, label))
        if not r["locked"]:
            choices.append((label, r["key"]))
    counts = Counter(r["split"] for r in records)
    status = f"共 {len(records)} 张 | Train {counts['train']} · Val {counts['val']} · Test(锁定) {counts['test']} · 待分配 {counts['unassigned']} | 筛选 {len(filtered)} 张 | 第 {page}/{pages} 页"
    # Grouping exists to keep one shooting scene on one side of the train/val
    # boundary, but nothing enforces it — and `group` defaults to the source
    # directory name, so this is a prompt to review, not an error.
    spanning = split_spanning_groups(state)
    if spanning:
        status += f" | ⚠ 同时出现在 Train 和 Val 的场景组：{'、'.join(spanning)}"
    return gallery, gr.update(choices=choices, value=[]), page, status


def build_dataset_grouping_tab():
    gr.Markdown("按图片手动划分 Train / Val。标准数据集目录只加载 train/val/test 下的 annotations.json，忽略旁边的备份；其他目录递归查找 COCO。也可每行指定一个 COCO 文件。"
                "现有 train/val/test 子目录会继承分组，Test锁定。导出为新目录，不更改训练路径；导出后请到「数据集登记」审核登记。")
    state = gr.State({"records": []})
    paths = gr.Textbox(label="图片＋COCO来源（服务器路径，每行一个，可分批追加）", lines=3,
                       value=str(BOOK_ROOT / "data/book_spine_sam3_dataset"))
    initial = gr.Dropdown(["unassigned", "train", "val", "test"], value="unassigned", label="新批次初始分组（已知子目录优先）")
    with gr.Row():
        load = gr.Button("加载 / 追加图片", variant="primary")
        clear = gr.Button("清空当前选择池（不删除文件）")
    with gr.Row():
        query = gr.Textbox(label="筛选文件名 / 路径 / 场景组（回车应用）", value="")
        split_filter = gr.Dropdown(["all", "unassigned", "train", "val", "test"], value="all", label="显示分组")
        page = gr.Number(value=1, precision=0, minimum=1, label="页码（每页24张）")
        refresh = gr.Button("刷新 / 跳页")
    status = gr.Textbox(label="分组统计", interactive=False)
    gallery = gr.Gallery(label="图片与标注预览（多边形轮廓；RLE显示框）", columns=4, height=650)
    selected = gr.CheckboxGroup(label="勾选本页图片（编号与预览一致）", choices=[])
    with gr.Row():
        all_page = gr.Button("全选本页")
        group = gr.Textbox(label="场景组名（可选，例如：书架A_上排）")
        target = gr.Dropdown(["train", "val", "unassigned"], value="val", label="分配到")
        move = gr.Button("应用到勾选图片")
    gr.Markdown("同一拍摄场景尽量放同一组。跨页操作前先应用本页选择；筛选或翻页会清空勾选。分配结果可在导出的 manual_split.json 中恢复。")
    with gr.Row():
        plan = gr.Textbox(label="恢复划分名单路径（先加载相同来源）")
        restore = gr.Button("恢复名单")
    output = gr.Textbox(label="新数据集输出目录（必须不存在）", value=str(
        BOOK_ROOT / "data" / f"book_spine_manual_{datetime.now():%Y%m%d_%H%M%S}"))
    export = gr.Button("导出图片及 COCO（保留原数据）", variant="primary")
    result = gr.Textbox(label="导出结果", lines=5, interactive=False)
    view_outputs = [gallery, selected, page, status]

    def load_cb(s, p, init):
        try:
            return add_sources(s, p, init)
        except Exception as exc:
            raise gr.Error(str(exc)) from exc

    def move_cb(s, keys, dest, g):
        try:
            return assign(s, keys, dest, g)
        except Exception as exc:
            raise gr.Error(str(exc)) from exc

    def restore_cb(s, p):
        try:
            return restore_plan(s, p)
        except Exception as exc:
            raise gr.Error(str(exc)) from exc

    def export_cb(s, p):
        try:
            r = export_dataset(s, p)
            warning = ""
            if r["spanning_groups"]:
                warning = ("\n⚠ 以下场景组同时出现在 Train 和 Val，可能造成同场景泄漏，"
                           f"请确认是否有意为之：{'、'.join(r['spanning_groups'])}")
            return (f"已导出：{r['output']}\n图片数：{r['counts']}\n固定名单：{r['plan']}\n"
                    f"尚未切换训练数据，也未自动授权数据集登记。{warning}")
        except Exception as exc:
            raise gr.Error(str(exc)) from exc

    load.click(load_cb, [state, paths, initial], state).success(_view, [state, query, split_filter, page], view_outputs)
    clear.click(lambda: {"records": []}, outputs=state).then(_view, [state, query, split_filter, page], view_outputs)
    move.click(move_cb, [state, selected, target, group], state).success(_view, [state, query, split_filter, page], view_outputs)
    restore.click(restore_cb, [state, plan], state).success(_view, [state, query, split_filter, page], view_outputs)
    refresh.click(_view, [state, query, split_filter, page], view_outputs)
    # The dropdown changes discretely, so live updates are cheap. The textbox
    # fires .change on every keystroke, and each redraw decodes and annotates 24
    # thumbnails — so it applies on Enter (or the refresh button) instead.
    split_filter.change(_view, [state, query, split_filter, page], view_outputs)
    query.submit(_view, [state, query, split_filter, page], view_outputs)

    def select_page(s, q, f, p):
        rows = [r for r in s["records"] if (f == "all" or r["split"] == f)
                and q.casefold() in (r["path"] + " " + r["group"]).casefold()]
        p = min(max(1, int(p or 1)), max(1, (len(rows) + PAGE_SIZE - 1)//PAGE_SIZE))
        return [r["key"] for r in rows[(p-1)*PAGE_SIZE:p*PAGE_SIZE] if not r["locked"]]
    all_page.click(select_page, [state, query, split_filter, page], selected)
    export.click(export_cb, [state, output], result)
