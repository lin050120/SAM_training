import json
from pathlib import Path

import pytest
from PIL import Image

from core.manual_dataset_split import add_sources, assign, export_dataset, restore_plan


def batch(root, name, color, category=7):
    folder = root / name
    (folder / "images").mkdir(parents=True)
    Image.new("RGB", (12, 10), color).save(folder / "images/a.png")
    coco = dict(images=[dict(id=1, file_name="a.png", width=12, height=10)],
                categories=[dict(id=category, name="book spine")],
                annotations=[dict(id=1, image_id=1, category_id=category,
                                  bbox=[1, 1, 4, 5], area=20, iscrowd=0,
                                  segmentation={"size": [10, 12], "counts": "abc"}, custom="keep")])
    p = folder / "annotations.json"
    p.write_text(json.dumps(coco))
    return p


def test_export_preserves_annotations_and_remaps_batches(tmp_path):
    a = batch(tmp_path, "batchA", "red")
    b = batch(tmp_path, "batchB", "blue", 42)
    s = add_sources({}, f"{a}\n{b}")
    s = assign(s, [s['records'][0]['key']], 'train', 'shelfA')
    s = assign(s, [s['records'][1]['key']], 'val', 'shelfB')
    out = tmp_path / 'export'
    export_dataset(s, str(out))
    for split in ['train', 'val']:
        c = json.loads((out / split / 'annotations.json').read_text())
        assert c['annotations'][0]['custom'] == 'keep'
        assert c['annotations'][0]['segmentation']['counts'] == 'abc'
        assert c['annotations'][0]['category_id'] == c['categories'][0]['id'] == 1
        assert (out / split / 'images' / c['images'][0]['file_name']).is_file()
    assert restore_plan(add_sources({}, f"{a}\n{b}"), str(out/'manual_split.json')) == s
    with pytest.raises(ValueError, match='已存在'):
        export_dataset(s, str(out))
    assert a.exists() and b.exists()


def test_test_locked_and_duplicates_atomic(tmp_path):
    a = batch(tmp_path, 'test', 'red')
    s = add_sources({}, str(a))
    with pytest.raises(ValueError, match='锁定'):
        assign(s, [s['records'][0]['key']], 'train')
    b = batch(tmp_path, 'other', 'red')
    with pytest.raises(ValueError, match='重复图片'):
        add_sources(s, str(b))
    assert len(s['records']) == 1


def test_changed_sources_and_unassigned_block_export(tmp_path):
    a = batch(tmp_path, 'train', 'red')
    b = batch(tmp_path, 'val', 'blue')
    s = add_sources({}, f'{a}\n{b}')
    c = assign(s, [s['records'][0]['key']], 'unassigned')
    with pytest.raises(ValueError, match='分配全部'):
        export_dataset(c, str(tmp_path/'out'))
    a.write_text(a.read_text() + '\n')
    with pytest.raises(ValueError, match='标注发生变化'):
        export_dataset(s, str(tmp_path/'out'))
    assert not (tmp_path/'out').exists()


def test_preview_pagination(tmp_path):
    from ui.dataset_grouping_page import _view
    a = batch(tmp_path, 'train', 'red')
    s = add_sources({}, str(a))
    gallery, choices, page, status = _view(s, '', 'all', 99)
    assert page == 1 and len(gallery) == 1
    assert choices['choices'][0][1] == s['records'][0]['key']
    assert 'Train 1' in status


def test_dataset_root_ignores_backup_splits(tmp_path):
    batch(tmp_path, 'train', 'red')
    batch(tmp_path, 'val', 'blue')
    batch(tmp_path, 'test', 'green')
    batch(tmp_path, 'test.pre_0725_merge', 'green')
    s = add_sources({}, str(tmp_path))
    assert len(s['records']) == 3
    assert {r['split'] for r in s['records']} == {'train', 'val', 'test'}
    assert len(add_sources(s, str(tmp_path))['records']) == 3
