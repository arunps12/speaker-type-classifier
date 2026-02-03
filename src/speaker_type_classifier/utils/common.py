from __future__ import annotations

from pathlib import Path
from datetime import datetime
from typing import Optional, Any, Dict, Union

import yaml





def make_run_dir(
    run_root: Path,
    stage_name: str,
    prefix: str = "run",
    ts: Optional[str] = None,
) -> Path:
    """
    Create a unique run directory:
      artifacts/runs/<stage_name>/<prefix>_YYYYmmdd_HHMMSS
    """
    run_root = Path(run_root)
    stamp = ts or datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = run_root / stage_name / f"{prefix}_{stamp}"
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir




def read_yaml(path: Union[str, Path]) -> Dict[str, Any]:
    """
    Read a YAML file and return its contents as a dictionary.

    Args:
        path (str | Path): Path to YAML file

    Returns:
        dict: Parsed YAML content

    Raises:
        FileNotFoundError: If file does not exist
        ValueError: If YAML is empty or invalid
    """
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"YAML file not found: {path}")

    with path.open("r", encoding="utf-8") as f:
        try:
            content = yaml.safe_load(f)
        except yaml.YAMLError as e:
            raise ValueError(f"Invalid YAML format in {path}: {e}") from e

    if content is None:
        raise ValueError(f"YAML file is empty: {path}")

    if not isinstance(content, dict):
        raise ValueError(f"YAML root must be a mapping (dict): {path}")

    return content


def write_yaml(
    path: Union[str, Path],
    data: Dict[str, Any],
    sort_keys: bool = False,
) -> None:
    """
    Write a dictionary to a YAML file.

    Args:
        path (str | Path): Output YAML path
        data (dict): Data to write
        sort_keys (bool): Sort keys alphabetically (default False)
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(
            data,
            f,
            default_flow_style=False,
            sort_keys=sort_keys,
            allow_unicode=True,
        )
