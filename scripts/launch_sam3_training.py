"""Launch SAM3 training with a runtime config that lives outside the sam3.train package.

The official entrypoint `sam3/train/train.py` initializes Hydra with
`initialize_config_module("sam3.train")`, so its `-c` argument is a Hydra config
*name* searched inside that package (`pkg://sam3.train`) — an absolute filesystem
path can never be found there (E2 failure: `MissingConfigException: Cannot find
primary config 'home/book/book01/...'`). Per-run runtime YAMLs must stay under the
run directory and must not be copied into /home/book/sam301, so this wrapper
initializes Hydra from the runtime config's own directory via
`initialize_config_dir` and then hands off to the unmodified official
`sam3.train.train.main()`, keeping all launcher semantics (submitit config,
single_node_runner, --num-gpus override) official.

`--validate-only` composes the config and exits without importing torch or
creating any trainer, so preflight can prove Hydra loadability without training.
"""

from __future__ import annotations

import argparse
import sys
from argparse import Namespace
from pathlib import Path


LEGACY_BOOK_ROOT = Path("/home/book/book01")
LEGACY_SAM301_ROOT = Path("/home/book/sam301")
FORBIDDEN_SAM3_ROOT = Path("/home/book/sam3")


def resolve_config_target(config_path: str | Path) -> tuple[str, str]:
    """Map an absolute runtime YAML path to (hydra config_dir, config_name)."""
    resolved = Path(config_path).expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"runtime config does not exist: {resolved}")
    if resolved.suffix not in {".yaml", ".yml"}:
        raise ValueError(f"runtime config must be a YAML file: {resolved}")
    return str(resolved.parent), resolved.stem


def compose_runtime_config(config_path: str | Path):
    """Compose the runtime config exactly the way the launch path will.

    Hydra-only: does not import torch/sam3 and does not resolve lazy
    interpolations (SAM3's custom `times` resolver is only registered in the
    real launch path, matching train.py's own __main__ flow).
    """
    from hydra import compose, initialize_config_dir

    config_dir, config_name = resolve_config_target(config_path)
    with initialize_config_dir(config_dir=config_dir, version_base="1.2"):
        cfg = compose(config_name=config_name)
    if cfg.get("trainer") is None:
        raise ValueError(f"composed config has no 'trainer' section: {config_path}")
    if cfg.get("launcher") is None:
        raise ValueError(f"composed config has no 'launcher' section: {config_path}")
    if cfg.get("submitit") is None:
        raise ValueError(f"composed config has no 'submitit' section: {config_path}")
    return cfg


def _configured_sam301_root(project_root: Path) -> Path:
    """Resolve SAM301 without importing book01 modules inside the train process."""
    import json

    local_config = project_root / "config" / "local_paths.json"
    if not local_config.exists():
        if project_root == LEGACY_BOOK_ROOT.resolve(strict=False):
            return LEGACY_SAM301_ROOT.resolve(strict=False)
        raise ValueError(
            f"no config/local_paths.json for checkout {project_root}; "
            "run: conda run -n sam301 python scripts/migrate_environment.py"
        )
    data = json.loads(local_config.read_text(encoding="utf-8"))
    if data.get("schema_version") != 1:
        raise ValueError(f"unsupported local machine-path config version: {local_config}")
    configured_book_root = Path(str(data["book_root"])).expanduser()
    configured_sam301_root = Path(str(data["sam301_root"])).expanduser()
    if not configured_book_root.is_absolute() or not configured_sam301_root.is_absolute():
        raise ValueError(f"local machine paths must be absolute: {local_config}")
    if configured_book_root.resolve(strict=False) != project_root:
        raise ValueError(
            f"local machine-path config points to {configured_book_root.resolve(strict=False)}, "
            f"but this checkout is {project_root}"
        )
    return configured_sam301_root.resolve(strict=False)


def _resolve_guard_target(project_root: Path, manifest: dict) -> tuple[Path, Path]:
    """Resolve both v1 absolute and v2 SAM301-relative manifest targets."""
    raw_target = Path(str(manifest["target_file"])).expanduser()
    if raw_target.is_absolute():
        if not manifest.get("expected_sam3_root"):
            raise ValueError("absolute manifest target requires expected_sam3_root")
        expected_root = Path(str(manifest["expected_sam3_root"])).expanduser().resolve(strict=False)
        target = raw_target.resolve(strict=False)
    else:
        expected_root = _configured_sam301_root(project_root)
        target = (expected_root / raw_target).resolve(strict=False)
    target.relative_to(expected_root)
    try:
        target.relative_to(FORBIDDEN_SAM3_ROOT.resolve(strict=False))
    except ValueError:
        pass
    else:
        raise ValueError(f"target points into forbidden old SAM3 root: {target}")
    return expected_root, target


def _verify_sam301_patch_or_die(project_root: Path | None = None) -> None:
    """Last-line fail-closed guard inside the actual training subprocess.

    Stdlib-only re-implementation of the manifest hash check (this script must not
    depend on book01 imports): the trainer file must match the manifest's patched
    SHA256 exactly, otherwise training is refused.
    """
    import hashlib
    import json

    root = (project_root or Path(__file__).resolve().parent.parent).expanduser().resolve(strict=False)
    manifest_path = root / "config" / "sam301_patch_manifest.json"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        _expected_root, target = _resolve_guard_target(root, manifest)
        patch_root = (root / "patches").resolve(strict=False)
        patch_file = Path(manifest["patch_file"])
        if not patch_file.is_absolute():
            patch_file = root / patch_file
        patch_file = patch_file.expanduser().resolve(strict=False)
        patch_file.relative_to(patch_root)
        expected_patch = manifest.get("patch_file_sha256")
        if expected_patch and hashlib.sha256(patch_file.read_bytes()).hexdigest() != expected_patch:
            raise ValueError(f"patch file sha256 mismatch: {patch_file}")
        expected = manifest["patched_sha256"]
        actual = hashlib.sha256(target.read_bytes()).hexdigest()
    except Exception as exc:
        raise SystemExit(f"sam301 patch guard: cannot evaluate {manifest_path}: {exc!r}; refusing to train")
    if actual != expected:
        raise SystemExit(
            f"sam301 patch guard: {target} sha256={actual} != expected patched {expected}; "
            "refusing to train. Run: conda run -n sam301 python scripts/manage_sam301_patch.py verify"
        )


def run_training(
    config_path: str | Path,
    num_gpus: int | None,
    num_nodes: int | None,
    use_cluster: int | None,
    wait_free_gb: float = 29.3,
    wait_check_interval: float = 30.0,
) -> None:
    _verify_sam301_patch_or_die()

    from hydra import initialize_config_dir

    # Same pre-main setup as train.py's own __main__ block.
    from sam3.train.train import main as sam3_train_main
    from sam3.train.utils.train_utils import register_omegaconf_resolvers

    config_dir, config_name = resolve_config_target(config_path)
    train_args = Namespace(
        config=config_name,
        use_cluster=bool(use_cluster) if use_cluster is not None else None,
        partition=None,
        account=None,
        qos=None,
        num_gpus=num_gpus,
        num_nodes=num_nodes,
        wait_free_gb=wait_free_gb,
        wait_check_interval=wait_check_interval,
    )
    register_omegaconf_resolvers()
    with initialize_config_dir(config_dir=config_dir, version_base="1.2"):
        sam3_train_main(train_args)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-c", "--config", required=True, help="absolute path to the per-run runtime_config.yaml")
    parser.add_argument("--use-cluster", type=int, default=None, help="0: run locally, 1: run on a cluster")
    parser.add_argument("--num-gpus", type=int, default=None, help="number of GPUs per node")
    parser.add_argument("--num-nodes", type=int, default=None, help="number of nodes")
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="compose the config with Hydra and exit without starting training",
    )
    parser.add_argument(
        "--wait-free-gb",
        type=float,
        default=29.3,
        help="wait until every visible GPU has at least this much free memory (GiB) "
        "before starting training (29.3 GiB = 30000 MiB, past training peak); "
        "set 0 to disable",
    )
    parser.add_argument(
        "--wait-check-interval",
        type=float,
        default=30.0,
        help="seconds between free GPU memory checks while waiting",
    )
    args = parser.parse_args(argv)

    if args.validate_only:
        try:
            compose_runtime_config(args.config)
        except Exception as exc:
            print(f"hydra validation failed: {exc!r}", file=sys.stderr)
            return 1
        print(f"hydra validation ok: {args.config}")
        return 0

    run_training(
        args.config,
        num_gpus=args.num_gpus,
        num_nodes=args.num_nodes,
        use_cluster=args.use_cluster,
        wait_free_gb=args.wait_free_gb,
        wait_check_interval=args.wait_check_interval,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
