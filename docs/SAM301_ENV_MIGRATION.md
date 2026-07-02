# SAM301 Environment Migration

Generated: 2026-07-02

## Purpose

`sam301` was created by cloning the older `sam3` Conda environment so `book01` can run against the target SAM3 source tree at `/home/book/sam301` without changing the old environment. The clone initially preserved the old editable install metadata, so `import sam3` still resolved to `/home/book/sam3`.

No real SAM3 training was started during this migration.

## Pre-Fix Audit

- New Conda environment: `sam301`
- Clone source: `sam3`
- Pre-fix Python: `/home/book/anaconda3/envs/sam301/bin/python`
- Pre-fix `sam3` import path from neutral cwd and with `PYTHONPATH` unset: `/home/book/sam3/sam3/__init__.py`
- Old binding source:
  - `pip show sam3` reported `Editable project location: /home/book/sam3`
  - `sam3-0.1.0.dist-info/direct_url.json` contained `file:///home/book/sam3`
  - `__editable___sam3_0_1_0_finder.py` mapped `sam3` to `/home/book/sam3/sam3`
- `/home/book/sam301/pyproject.toml` exists and supports editable install.

## Environment Fix

Commands executed only against `sam301`:

```bash
conda run -n sam301 python -m pip uninstall -y sam3
conda run -n sam301 python -m pip install -e /home/book/sam301 --no-deps
```

- Old editable binding uninstalled from `sam301`: yes.
- `/home/book/sam301` installed editable into `sam301`: yes.
- `--no-deps` used: yes, to avoid changing the cloned dependency stack.
- Old Conda environment `sam3` was not modified.
- `/home/book/sam3` was not modified.
- `/home/book/sam301` source files were not modified.

## Post-Fix Verification

From neutral cwd with `PYTHONPATH` unset:

- Python: `/home/book/anaconda3/envs/sam301/bin/python`
- `sam3`: `/home/book/sam301/sam3/__init__.py`
- `from sam3.model_builder import build_sam3_image_model`: OK
- Old `sam3` environment still imports: `/home/book/sam3/sam3/__init__.py`

Dependency checks in `sam301`:

- Python: `3.12.13`
- PyTorch: `2.10.0+cu128`
- CUDA build: `12.8`
- CUDA available in Codex sandbox: `False`
- GPU in Codex sandbox: `NONE`
- Gradio: `5.50.0`
- huggingface_hub: `1.19.0`
- timm: `1.0.27`
- `pip check`: non-zero because `decord 0.6.0 is not supported on this platform`; this was recorded and not changed.

Codex sandbox GPU visibility remains limited. Do not infer that the physical machine has no GPU from this result.

## Project Migration

- Project default Conda environment is now `sam301`.
- Expected SAM3 root is `/home/book/sam301`.
- Training command builder now emits `conda run -n sam301 ...`.
- Preflight writes `conda_environment`, `expected_sam3_root`, `resolved_sam3_import_path`, import guard status, and effective `PYTHONPATH`.
- `training_config_summary.json` is written for new preflight runs.
- `training_summary.json` records the environment/import metadata used at launch.

## Import Guard

Real training launch now performs a lightweight guard before `subprocess.Popen`:

- Uses `conda run -n sam301 python -c ...`.
- Runs from the same project cwd and with the same child environment used for the trainer.
- Resolves `Path(sam3.__file__).resolve()`.
- Requires `/home/book/sam301/sam3/__init__.py`.
- Uses canonical `Path` comparison, not string prefix checks.
- Fails before creating the trainer process if import resolves to `/home/book/sam3` or any other external path.

The project uses both the dedicated `sam301` environment and a defensive child-process `PYTHONPATH` prefix:

- `sam301` now imports correctly even when `PYTHONPATH` is unset.
- Training children receive `PYTHONPATH=/home/book/sam301[:existing]`.
- The existing `PYTHONPATH` value is preserved after the canonical prefix.
- The global `os.environ` is not modified.

## Tests

- Targeted: `conda run -n sam301 python -m pytest tests/test_e1_training.py tests/test_ui.py -v`
  - Result: `142 passed, 3 warnings, 15 subtests passed`
- Full: `conda run -n sam301 python -m pytest tests/ -v`
  - Sandbox run failed only at Gradio smoke due socket permission.
  - Re-run with socket permission: `144 passed, 5 warnings, 15 subtests passed`
- Gradio startup smoke test: passed.
- Residual trainer process check: no matches for `[s]am3/train/train.py|[r]untime_config.yaml`.

## E2 Preflight

- Old E2 run `2026-07-02_19-17-28` is superseded due to Conda environment migration.
- New E2 preflight run: `2026-07-02_19-46-53`
- New command uses `sam301`.
- New import guard result: `/home/book/sam301/sam3/__init__.py`
- New run was not started.

## Rollback

If the migration needs to be reverted, do not touch the old `sam3` environment. In `sam301`, uninstall the editable package and reinstall the desired source explicitly:

```bash
conda run -n sam301 python -m pip uninstall -y sam3
conda run -n sam301 python -m pip install -e /home/book/sam301 --no-deps
```

To intentionally point `sam301` elsewhere, replace only the final path and verify `import sam3` from a neutral cwd with `PYTHONPATH` unset.

## Still Required Before Real Training

Run these in a normal terminal:

```bash
env -u PYTHONPATH conda run -n sam301 python -c "from pathlib import Path; import sys, torch, sam3; print('python:', sys.executable); print('torch:', torch.__version__); print('torch cuda:', torch.version.cuda); print('cuda:', torch.cuda.is_available()); print('gpu:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NONE'); print('sam3:', Path(sam3.__file__).resolve())"
conda run -n sam301 python -m pip check
pgrep -af "[s]am3/train/train.py|[r]untime_config.yaml"
```

Expected normal-terminal GPU result for this machine: CUDA `True`, GPU `NVIDIA GeForce RTX 5090`, SAM3 import `/home/book/sam301/sam3/__init__.py`.
