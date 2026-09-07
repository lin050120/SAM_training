"""Manual COCO assignment; source datasets are never edited."""
from __future__ import annotations

import copy
import json
import os
import shutil
import tempfile
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image

from core.dataset_split_builder import _collect_coco_batches, _locate_image, _sha256_file


def _source_batches(entry: Path):
    if entry.is_file():
        return [(entry, json.loads(entry.read_text()))]
    # A built dataset can retain old split backups beside the active splits.
    # Its canonical files are authoritative; never ingest those backups.
    active = [entry / split / "annotations.json" for split in ("train", "val", "test")]
    if any(p.is_file() for p in active):
        return [(p, json.loads(p.read_text())) for p in active if p.is_file()]
    return _collect_coco_batches(entry)


def add_sources(state: dict, paths: str, initial_split: str = "unassigned") -> dict:
    result = copy.deepcopy(state or {"records": []})
    known = {r["key"] for r in result["records"]}
    hashes = {r["sha256"] for r in result["records"]}
    if initial_split not in {"unassigned", "train", "val", "test"}:
        raise ValueError("无效分组")
    entries = [Path(s.strip()).expanduser().resolve() for s in paths.splitlines() if s.strip()]
    if not entries:
        raise ValueError("请输入 COCO 文件或数据集目录")
    for entry in entries:
        batches = _source_batches(entry)
        for source, coco in batches:
            categories = {c["id"]: c for c in coco["categories"]}
            images = {im["id"]: im for im in coco["images"]}
            if len(images) != len(coco["images"]) or len(categories) != len(coco["categories"]):
                raise ValueError(f"重复 image/category ID: {source}")
            annotations = defaultdict(list)
            ids = set()
            for ann in coco["annotations"]:
                if ann["image_id"] not in images or ann["category_id"] not in categories or ann["id"] in ids:
                    raise ValueError(f"标注关联无效或 ID 重复: {source}")
                ids.add(ann["id"])
                annotations[ann["image_id"]].append(ann)
            source_hash = _sha256_file(source)
            inferred = source.parent.name
            split = inferred if inferred in {"train", "val", "test"} else initial_split
            for ident, im in images.items():
                key = f"{source}::{ident}"
                if key in known:
                    continue
                path = _locate_image(source.parent, im["file_name"])
                if path is None:
                    raise ValueError(f"找不到图片: {source}: {im['file_name']}")
                with Image.open(path) as image:
                    image.load()
                    if image.size != (im["width"], im["height"]):
                        raise ValueError(f"图片尺寸与 COCO 不一致: {path}")
                digest = _sha256_file(path)
                if digest in hashes:
                    raise ValueError(f"发现重复图片内容，请先去重，避免跨组泄漏: {path}")
                hashes.add(digest)
                known.add(key)
                result["records"].append(dict(
                    key=key, source=str(source), source_sha256=source_hash,
                    path=str(path), sha256=digest, image=im,
                    annotations=annotations[ident], categories=coco["categories"],
                    split=split, locked=split == "test", group=source.parent.name,
                ))
    return result


def split_spanning_groups(state: dict) -> dict[str, list[str]]:
    """Scene groups that appear in BOTH train and val, i.e. probable leakage.

    Grouping exists so one shooting scene lands entirely on one side of the
    train/val boundary; nothing else in this module enforces that, and `group`
    defaults to the source directory name, so a default grouping spanning both
    splits is normal and not by itself an error. Callers surface this as a
    warning rather than blocking the export.
    """
    seen: dict[str, set[str]] = defaultdict(set)
    for r in state.get("records", []):
        if r["split"] in {"train", "val"}:
            seen[r["group"]].add(r["split"])
    return {g: sorted(v) for g, v in sorted(seen.items()) if len(v) > 1}


def assign(state: dict, keys: list[str], split: str, group: str = "") -> dict:
    if split not in {"train", "val", "unassigned"}:
        raise ValueError("只能调整 Train / Val / 待分配")
    result = copy.deepcopy(state)
    selected = set(keys)
    records = {r["key"]: r for r in result["records"]}
    if not selected or not selected <= records.keys():
        raise ValueError("请先选择图片")
    if any(records[k]["locked"] for k in selected):
        raise ValueError("Test 图片已锁定，不能移动到 Train/Val")
    for k in selected:
        records[k]["split"] = split
        if group.strip():
            records[k]["group"] = group.strip()
    return result


def restore_plan(state: dict, plan_path: str) -> dict:
    plan = json.loads(Path(plan_path).expanduser().read_text())
    rows = plan["assignments"]
    if len({r["key"] for r in rows}) != len(rows):
        raise ValueError("划分名单存在重复项")
    mapping = {r["key"]: r for r in rows}
    result = copy.deepcopy(state)
    if set(mapping) != {r["key"] for r in result["records"]}:
        raise ValueError("请加载与名单完全相同的源数据")
    for r in result["records"]:
        p = mapping[r["key"]]
        if p["sha256"] != r["sha256"] or p["source_sha256"] != r["source_sha256"]:
            raise ValueError("图片或标注已变化，不能恢复旧名单")
        if p["split"] not in {"train", "val", "unassigned", "test"}:
            raise ValueError("名单包含无效分组")
        if (p["split"] == "test") != r["locked"]:
            raise ValueError("不能通过名单更改 Test 身份")
        r.update(split=p["split"], group=p.get("group", r["group"]))
    return result


def export_dataset(state: dict, destination: str) -> dict:
    records = state.get("records", [])
    counts = Counter(r["split"] for r in records)
    if counts["unassigned"] or not counts["train"] or not counts["val"]:
        raise ValueError("请分配全部图片，并确保 Train 和 Val 都非空")
    out = Path(destination).expanduser().resolve()
    if out.exists():
        raise ValueError("输出目录已存在，请选择新的版本目录；不会覆盖现有数据")
    for r in records:
        source_dir = Path(r["source"]).parent
        if out == source_dir or out.is_relative_to(source_dir):
            raise ValueError("输出目录不能放在输入 COCO 目录内部")
    for source in {r["source"] for r in records}:
        expected = {r["source_sha256"] for r in records if r["source"] == source}
        if expected != {_sha256_file(Path(source))}:
            raise ValueError(f"加载后标注发生变化，请重新加载: {source}")
    # Match category definitions, not numeric IDs from different COCO batches.
    categories, category_map = [], {}
    for r in records:
        for cat in r["categories"]:
            definition = {k: v for k, v in cat.items() if k != "id"}
            name = cat["name"]
            if name in category_map:
                if definition != category_map[name][1]:
                    raise ValueError(f"同名类别定义冲突: {name}")
            else:
                idx = len(categories) + 1
                category_map[name] = (idx, definition)
                categories.append(dict(definition, id=idx))
    out.parent.mkdir(parents=True, exist_ok=True)
    temp = Path(tempfile.mkdtemp(prefix=".manual_split_", dir=out.parent))
    assignments = []
    try:
        for split in ("train", "val", "test"):
            chosen = [r for r in records if r["split"] == split]
            if not chosen:
                continue
            image_dir = temp / split / "images"
            image_dir.mkdir(parents=True)
            images, anns = [], []
            for i, r in enumerate(chosen, 1):
                path = Path(r["path"])
                name = f"im_{i:06d}{path.suffix.lower()}"
                copied = image_dir / name
                shutil.copy2(path, copied)
                if _sha256_file(copied) != r["sha256"]:
                    raise ValueError(f"加载后图片发生变化: {path}")
                im = copy.deepcopy(r["image"])
                im.update(id=i, file_name=name)
                images.append(im)
                cats = {c["id"]: category_map[c["name"]][0] for c in r["categories"]}
                for annotation in r["annotations"]:
                    ann = copy.deepcopy(annotation)
                    ann.update(id=len(anns) + 1, image_id=i, category_id=cats[ann["category_id"]])
                    anns.append(ann)
                assignments.append({k: r[k] for k in ("key", "sha256", "source_sha256", "split", "group")}
                                   | {"export_file": f"{split}/images/{name}"})
            (temp / split / "annotations.json").write_text(json.dumps(
                dict(images=images, annotations=anns, categories=categories), ensure_ascii=False, indent=2))
        (temp / "manual_split.json").write_text(json.dumps(
            dict(version=1, assignments=assignments), ensure_ascii=False, indent=2))
        # Reserve destination exclusively, even if another request exports concurrently.
        out.mkdir()
        try:
            os.replace(temp, out)
        except Exception:
            out.rmdir()
            raise
    finally:
        if temp.exists():
            shutil.rmtree(temp)
    return {
        "output": str(out),
        "counts": dict(counts),
        "plan": str(out / "manual_split.json"),
        "spanning_groups": split_spanning_groups(state),
    }
