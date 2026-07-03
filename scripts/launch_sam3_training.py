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


def _verify_sam301_patch_or_die() -> None:
    """Last-line fail-closed guard inside the actual training subprocess.

    Stdlib-only re-implementation of the manifest hash check (this script must not
    depend on book01 imports): the trainer file must match the manifest's patched
    SHA256 exactly, otherwise training is refused.
    """
    import hashlib
    import json

    manifest_path = Path(__file__).resolve().parent.parent / "config" / "sam301_patch_manifest.json"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        target = Path(manifest["target_file"])
        expected = manifest["patched_sha256"]
        actual = hashlib.sha256(target.read_bytes()).hexdigest()
    except Exception as exc:
        raise SystemExit(f"sam301 patch guard: cannot evaluate {manifest_path}: {exc!r}; refusing to train")
    if actual != expected:
        raise SystemExit(
            f"sam301 patch guard: {target} sha256={actual} != expected patched {expected}; "
            "refusing to train. Run: conda run -n sam301 python scripts/manage_sam301_patch.py verify"
        )


def run_training(config_path: str | Path, num_gpus: int | None, num_nodes: int | None, use_cluster: int | None) -> None:
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
    args = parser.parse_args(argv)

    if args.validate_only:
        try:
            compose_runtime_config(args.config)
        except Exception as exc:
            print(f"hydra validation failed: {exc!r}", file=sys.stderr)
            return 1
        print(f"hydra validation ok: {args.config}")
        return 0

    run_training(args.config, num_gpus=args.num_gpus, num_nodes=args.num_nodes, use_cluster=args.use_cluster)
    return 0


if __name__ == "__main__":
    sys.exit(main())
