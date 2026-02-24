from __future__ import annotations

import importlib.util
import sys
import tomllib
from dataclasses import dataclass
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from .module_api import MODULE_API_VERSION, ModulePlugin


class ModuleEntrypoint(BaseModel):
    model_config = ConfigDict(extra="forbid")

    module_file: str = "module.py"
    class_name: str


class ModuleManifest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    module_id: str = Field(min_length=1)
    module_api_version: int = 1
    name: str = Field(min_length=1)
    version: str = Field(min_length=1)
    priority: int = 100
    enabled: bool = True
    intents: list[str] = Field(default_factory=list)
    actions: list[str] = Field(default_factory=list)
    permissions: list[str] = Field(default_factory=list)
    entrypoint: ModuleEntrypoint


@dataclass(frozen=True)
class LoadedModule:
    manifest: ModuleManifest
    plugin: ModulePlugin
    root: Path


class ModuleRegistry:
    def __init__(self, loaded: list[LoadedModule]):
        ordered = sorted(loaded, key=lambda item: item.manifest.priority)
        self._ordered = ordered
        self._by_id = {item.manifest.module_id: item for item in ordered}

    def all(self) -> list[LoadedModule]:
        return list(self._ordered)

    def get(self, module_id: str) -> LoadedModule | None:
        return self._by_id.get(module_id)


class ModuleLoadError(RuntimeError):
    pass


def _load_manifest(module_toml_path: Path) -> ModuleManifest:
    try:
        data = tomllib.loads(module_toml_path.read_text(encoding="utf-8"))
        manifest = ModuleManifest.model_validate(data)
        if manifest.module_api_version != MODULE_API_VERSION:
            raise ModuleLoadError(
                (
                    f"Module '{manifest.module_id}' has module_api_version="
                    f"{manifest.module_api_version}, expected {MODULE_API_VERSION}"
                )
            )
        return manifest
    except (OSError, ValidationError, tomllib.TOMLDecodeError) as exc:
        raise ModuleLoadError(f"Invalid module manifest at {module_toml_path}: {exc}") from exc


def _import_module_from_file(module_name: str, module_file: Path):
    spec = importlib.util.spec_from_file_location(module_name, module_file)
    if spec is None or spec.loader is None:
        raise ModuleLoadError(f"Unable to load module file: {module_file}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _load_plugin(manifest: ModuleManifest, module_root: Path):
    module_file = (module_root / manifest.entrypoint.module_file).resolve()
    if not module_file.exists():
        raise ModuleLoadError(
            f"Module '{manifest.module_id}' entrypoint file not found: {module_file}"
        )

    module_name = f"jsonllm_module_{manifest.module_id}"
    module = _import_module_from_file(module_name, module_file)

    if not hasattr(module, manifest.entrypoint.class_name):
        raise ModuleLoadError(
            f"Module '{manifest.module_id}' missing class '{manifest.entrypoint.class_name}'"
        )

    cls = getattr(module, manifest.entrypoint.class_name)
    plugin = cls()

    module_id = getattr(plugin, "module_id", None)
    if module_id != manifest.module_id:
        raise ModuleLoadError(
            f"Module id mismatch for '{manifest.module_id}': plugin reports '{module_id}'"
        )

    return plugin


def load_module_registry(modules_dir: Path) -> ModuleRegistry:
    if not modules_dir.exists():
        raise ModuleLoadError(f"Modules directory does not exist: {modules_dir}")

    loaded: list[LoadedModule] = []

    for module_root in sorted(path for path in modules_dir.iterdir() if path.is_dir()):
        module_toml = module_root / "module.toml"
        if not module_toml.exists():
            continue

        manifest = _load_manifest(module_toml)
        if not manifest.enabled:
            continue

        plugin = _load_plugin(manifest, module_root)
        loaded.append(LoadedModule(manifest=manifest, plugin=plugin, root=module_root))

    ids = [item.manifest.module_id for item in loaded]
    if len(ids) != len(set(ids)):
        raise ModuleLoadError("Duplicate module_id found in loaded modules")

    return ModuleRegistry(loaded)
