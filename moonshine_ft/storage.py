"""
Storage and cache helpers for Moonshine workflows.
"""

import os
from pathlib import Path
from typing import Any, Dict, Optional


def resolve_path(path_value: Optional[str], base_dir: Optional[Path] = None) -> Optional[str]:
    """
    Resolve a path string to an absolute path.

    If ``base_dir`` is provided, relative paths are interpreted relative to it.
    Otherwise, relative paths are left to the current working directory.
    """
    if path_value is None:
        return None

    path = Path(path_value).expanduser()
    if path.is_absolute() or base_dir is None:
        return str(path.resolve(strict=False))

    return str((base_dir / path).resolve(strict=False))


def _configured_path(
    configured_value: Optional[str],
    storage_root: Optional[Path],
    default_relative: Optional[str] = None
) -> Optional[str]:
    """
    Resolve a configured path, or derive a default under the storage root.
    """
    if configured_value:
        return resolve_path(configured_value, storage_root)

    if storage_root is None or default_relative is None:
        return None

    return resolve_path(default_relative, storage_root)


def configure_storage(config: Dict[str, Any]) -> Dict[str, Optional[str]]:
    """
    Apply storage-root aware path resolution and runtime cache configuration.

    Supported behavior:
    - ``storage.root_dir`` or ``MOONSHINE_STORAGE_ROOT`` makes relative storage
      paths resolve under a shared scratch directory.
    - Hugging Face model, datasets, and evaluate caches are redirected away
      from ``~/.cache`` when a storage root is configured.
    - Training output and logging directories are resolved under the storage
      root when they are relative.
    """
    storage_config = config.setdefault("storage", {})
    model_config = config.setdefault("model", {})
    dataset_config = config.setdefault("dataset", {})
    training_config = config.setdefault("training", {})

    raw_storage_root = storage_config.get("root_dir") or os.environ.get("MOONSHINE_STORAGE_ROOT")
    storage_root = Path(resolve_path(raw_storage_root)) if raw_storage_root else None

    if storage_root is not None:
        storage_config["root_dir"] = str(storage_root)

    model_cache_dir = _configured_path(
        model_config.get("cache_dir"),
        storage_root,
        default_relative="cache/models"
    )
    dataset_cache_dir = _configured_path(
        dataset_config.get("cache_dir"),
        storage_root,
        default_relative="cache/datasets"
    )
    evaluate_cache_dir = _configured_path(
        storage_config.get("evaluate_cache_dir"),
        storage_root,
        default_relative="cache/evaluate"
    )
    hf_home = _configured_path(
        storage_config.get("hf_home"),
        storage_root,
        default_relative="cache/huggingface"
    )
    hf_hub_cache = _configured_path(
        storage_config.get("hf_hub_cache"),
        storage_root,
        default_relative="cache/huggingface/hub"
    )
    hf_modules_cache = _configured_path(
        storage_config.get("hf_modules_cache"),
        storage_root,
        default_relative="cache/huggingface/modules"
    )
    hf_assets_cache = _configured_path(
        storage_config.get("hf_assets_cache"),
        storage_root,
        default_relative="cache/huggingface/assets"
    )

    if model_cache_dir is not None:
        model_config["cache_dir"] = model_cache_dir
    if dataset_cache_dir is not None:
        dataset_config["cache_dir"] = dataset_cache_dir
    if evaluate_cache_dir is not None:
        storage_config["evaluate_cache_dir"] = evaluate_cache_dir
    if hf_home is not None:
        storage_config["hf_home"] = hf_home
    if hf_hub_cache is not None:
        storage_config["hf_hub_cache"] = hf_hub_cache
    if hf_modules_cache is not None:
        storage_config["hf_modules_cache"] = hf_modules_cache
    if hf_assets_cache is not None:
        storage_config["hf_assets_cache"] = hf_assets_cache

    if storage_root is not None and training_config.get("output_dir"):
        training_config["output_dir"] = resolve_path(training_config["output_dir"], storage_root)

    if storage_root is not None and training_config.get("logging_dir"):
        training_config["logging_dir"] = resolve_path(training_config["logging_dir"], storage_root)

    dirs_to_create = [
        storage_root,
        Path(model_cache_dir) if model_cache_dir else None,
        Path(dataset_cache_dir) if dataset_cache_dir else None,
        Path(evaluate_cache_dir) if evaluate_cache_dir else None,
        Path(hf_home) if hf_home else None,
        Path(hf_hub_cache) if hf_hub_cache else None,
        Path(hf_modules_cache) if hf_modules_cache else None,
        Path(hf_assets_cache) if hf_assets_cache else None,
        Path(training_config["output_dir"]) if training_config.get("output_dir") else None,
        Path(training_config["logging_dir"]) if training_config.get("logging_dir") else None,
    ]

    for directory in dirs_to_create:
        if directory is not None:
            try:
                directory.mkdir(parents=True, exist_ok=True)
            except OSError as exc:
                raise OSError(
                    f"Could not create storage directory '{directory}'. "
                    "Update storage.root_dir or choose a writable path."
                ) from exc

    env_updates = {
        "HF_HOME": hf_home,
        "HF_HUB_CACHE": hf_hub_cache,
        "HUGGINGFACE_HUB_CACHE": hf_hub_cache,
        "HF_DATASETS_CACHE": dataset_cache_dir,
        "TRANSFORMERS_CACHE": model_cache_dir,
        "HF_MODULES_CACHE": hf_modules_cache,
        "HF_ASSETS_CACHE": hf_assets_cache,
        "HF_EVALUATE_CACHE": evaluate_cache_dir,
    }
    for env_name, env_value in env_updates.items():
        if env_value is not None:
            os.environ[env_name] = env_value

    try:
        from datasets import config as datasets_config

        if dataset_cache_dir is not None:
            datasets_config.HF_DATASETS_CACHE = dataset_cache_dir
    except ImportError:
        pass

    return {
        "storage_root": str(storage_root) if storage_root is not None else None,
        "model_cache_dir": model_cache_dir,
        "dataset_cache_dir": dataset_cache_dir,
        "evaluate_cache_dir": evaluate_cache_dir,
        "hf_home": hf_home,
        "output_dir": training_config.get("output_dir"),
        "logging_dir": training_config.get("logging_dir"),
    }
