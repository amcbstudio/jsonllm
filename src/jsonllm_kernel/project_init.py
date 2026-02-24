from __future__ import annotations

from importlib.resources import files
from pathlib import Path


def _copy_tree(src, dst: Path, force: bool, written: list[str]) -> None:
    dst.mkdir(parents=True, exist_ok=True)
    for item in src.iterdir():
        target = dst / item.name
        if item.is_dir():
            _copy_tree(item, target, force=force, written=written)
            continue

        if target.exists() and not force:
            continue

        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(item.read_bytes())
        written.append(str(target))


def init_project_workspace(target_root: Path, force: bool = False) -> dict[str, object]:
    root = target_root.resolve()
    root.mkdir(parents=True, exist_ok=True)

    template_root = files("jsonllm_kernel").joinpath("templates")

    written: list[str] = []

    _copy_tree(template_root.joinpath("catalog"), root / "catalog", force=force, written=written)
    _copy_tree(template_root.joinpath("modules"), root / "modules", force=force, written=written)

    data_dir = root / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    events_file = data_dir / "events.jsonl"
    if not events_file.exists() or force:
        events_file.write_text("", encoding="utf-8")
        written.append(str(events_file))

    outputs_dir = data_dir / "outputs"
    outputs_dir.mkdir(parents=True, exist_ok=True)

    gitignore_file = root / ".gitignore"
    default_ignores = ["data/events.jsonl", "data/outputs/", "__pycache__/", "*.pyc"]
    if not gitignore_file.exists():
        gitignore_file.write_text("\n".join(default_ignores) + "\n", encoding="utf-8")
        written.append(str(gitignore_file))
    else:
        current = gitignore_file.read_text(encoding="utf-8").splitlines()
        changed = False
        for line in default_ignores:
            if line not in current:
                current.append(line)
                changed = True
        if changed:
            gitignore_file.write_text("\n".join(current).rstrip() + "\n", encoding="utf-8")
            written.append(str(gitignore_file))

    return {
        "workspace": str(root),
        "written_files": written,
    }
