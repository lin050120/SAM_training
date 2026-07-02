from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

logger = logging.getLogger("book_spine_ui")
if not logger.handlers:
    logger.setLevel(logging.INFO)
    _handler = logging.StreamHandler()
    _handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s ui.%(message)s"))
    logger.addHandler(_handler)


@dataclass(frozen=True)
class CudaStatus:
    available: bool
    torch_version: str | None
    cuda_build_version: str | None
    device_name: str | None
    device_count: int
    note: str


def detect_cuda() -> CudaStatus:
    """Detect CUDA availability from inside the UI process itself.

    The UI process is expected to be launched with `conda run -n sam301 python app.py`
    from a normal terminal, so this reflects the terminal's real GPU visibility,
    not any restricted coding-agent sandbox.
    """
    try:
        import torch

        available = bool(torch.cuda.is_available())
        device_name = torch.cuda.get_device_name(0) if available else None
        device_count = torch.cuda.device_count() if available else 0
        return CudaStatus(
            available=available,
            torch_version=torch.__version__,
            cuda_build_version=torch.version.cuda,
            device_name=device_name,
            device_count=device_count,
            note="Detected from the UI process. Launch the UI from a normal terminal, "
            "not a restricted coding-agent sandbox, for this to reflect real GPU availability.",
        )
    except Exception as exc:  # pragma: no cover - defensive
        logger.exception("cuda_detection_failed")
        return CudaStatus(
            available=False,
            torch_version=None,
            cuda_build_version=None,
            device_name=None,
            device_count=0,
            note=f"torch import or CUDA query failed: {exc!r}",
        )


def safe_read_json(path: Path) -> tuple[Any | None, str | None]:
    """Read a JSON file, returning (data, error). Never raises."""
    if not path.exists():
        return None, f"file not found: {path}"
    try:
        return json.loads(path.read_text(encoding="utf-8")), None
    except Exception as exc:
        logger.exception("json_read_failed path=%s", path)
        return None, f"failed to parse {path}: {exc!r}"


def format_json(data: Any) -> str:
    if data is None:
        return "{}"
    try:
        return json.dumps(data, ensure_ascii=False, indent=2, default=_json_default)
    except Exception as exc:  # pragma: no cover - defensive
        return f"<failed to render JSON: {exc!r}>"


def _json_default(obj: Any) -> Any:
    try:
        import numpy as np

        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
    except Exception:
        pass
    return str(obj)


def check_input_path(path_str: str, must_exist: bool = True, must_be_dir: bool | None = None) -> str | None:
    """Validate a user-supplied path. Returns an error string, or None if valid."""
    if not path_str or not path_str.strip():
        return "path is empty"
    try:
        path = Path(path_str).expanduser()
    except Exception as exc:
        return f"invalid path: {exc!r}"
    if must_exist and not path.exists():
        return f"path does not exist: {path}"
    if must_exist and must_be_dir is True and not path.is_dir():
        return f"path is not a directory: {path}"
    if must_exist and must_be_dir is False and not path.is_file():
        return f"path is not a file: {path}"
    return None


def parse_optional_positive_int(value: Any, allow_zero: bool = False) -> tuple[int | None, str | None]:
    """Parse an optional positive-integer UI field into (value, error).

    Gradio widgets deliver unreliable payloads for empty numeric fields (None, "",
    or NaN depending on component and version), so all optional integer inputs go
    through this single adapter instead of trusting the widget:

    - None / empty or whitespace-only string -> (None, None): field not set.
    - Positive whole number (int, float like 10.0, or numeric string) -> (int, None).
    - 0, negatives, non-whole floats, NaN/inf, or unparseable text -> (None, error).

    allow_zero=True permits 0 (e.g. num_workers=0 is PyTorch DataLoader's documented
    "load in the main process" setting, not an invalid value).
    """
    if value is None:
        return None, None
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None, None
        try:
            value = float(text)
        except ValueError:
            return None, f"not a number: {text!r}"
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None, f"unsupported value type: {type(value).__name__}"
    number = float(value)
    if not math.isfinite(number):
        return None, "not a finite number"
    if number != int(number):
        return None, f"must be a whole number, got {number}"
    minimum_ok = int(number) >= 0 if allow_zero else int(number) > 0
    if not minimum_ok:
        bound = "non-negative" if allow_zero else "positive"
        return None, f"must be a {bound} integer, got {int(number)}"
    return int(number), None


def parse_optional_positive_float(value: Any) -> tuple[float | None, str | None]:
    """Same contract as parse_optional_positive_int, but allows fractional values.

    Used for fields like learning_rate where the base YAML value is not a whole
    number. None / empty string means "no override, fall back to the base config".
    """
    if value is None:
        return None, None
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None, None
        try:
            value = float(text)
        except ValueError:
            return None, f"not a number: {text!r}"
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None, f"unsupported value type: {type(value).__name__}"
    number = float(value)
    if not math.isfinite(number):
        return None, "not a finite number"
    if number <= 0:
        return None, f"must be a positive number, got {number}"
    return number, None


def is_finite_number(value: Any) -> bool:
    """True when value is a real, finite number (guards against NaN from gr.Number)."""
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def user_error(message: str, exc: Exception | None = None) -> str:
    """Convert an exception into a short user-readable message; log full detail."""
    if exc is not None:
        logger.exception(message)
    else:
        logger.error(message)
    return message
