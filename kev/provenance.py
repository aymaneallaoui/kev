"""What a kev.train run was made from, written as <out>/provenance.json: the kev commit, the sha256 of its data files, the
base revision actually loaded, the warm start, the library versions and the hardware, and the records the context filter
rejected. training_config.json keeps the arguments; this file does not repeat them."""
import os, platform, re, subprocess
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path
from .suite import digest

PACKAGE = Path(__file__).resolve().parent


def code_commit(root=PACKAGE):
    """{"commit", "dirty"} of the checkout kev runs from; dirty counts tracked files only. Inside a container without .git,
    KEV_GIT_COMMIT (as kev.experiment.git_commit) with dirty unknown; both None outside a git checkout."""
    if os.environ.get("KEV_GIT_COMMIT"):
        return {"commit": os.environ["KEV_GIT_COMMIT"], "dirty": None}
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True, stderr=subprocess.DEVNULL).strip()
        status = subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=no"], cwd=root, text=True, stderr=subprocess.DEVNULL)
    except (subprocess.CalledProcessError, OSError):
        return {"commit": None, "dirty": None}
    return {"commit": commit, "dirty": bool(status.strip())}


def file_record(path):
    """{"path", "bytes", "sha256"} of one data file, or None when no file was given."""
    if not path: return None
    p = Path(path).resolve()
    return {"path": str(p), "bytes": p.stat().st_size, "sha256": digest(p)}


def resolved_revision(repo, revision=None, cache=None):
    """(the full commit sha `repo` loads from at `revision`, None) or (None, why not), from the local HF cache only: a full
    sha as given, else the cached ref (main when no revision), else the one cached snapshot a short sha prefixes."""
    if os.path.isdir(repo):
        return None, "a local directory, not a Hub repo"
    if revision and re.fullmatch(r"[0-9a-f]{40}", revision):
        return revision, None
    if cache is None:
        from huggingface_hub import constants
        cache = constants.HF_HUB_CACHE
    root = Path(cache) / f"models--{repo.replace('/', '--')}"
    ref = root / "refs" / (revision or "main")
    if ref.is_file():
        return ref.read_text(encoding="utf-8").strip(), None
    if revision and re.fullmatch(r"[0-9a-f]{4,39}", revision):
        matches = [p.name for p in (root / "snapshots").glob(f"{revision}*") if p.is_dir()]
        if len(matches) == 1: return matches[0], None
    return None, f"revision {revision or 'main'} of {repo} is not resolvable from the local HF cache ({root})"


def init_record(init_source):
    """The warm start (Checkpoint.warm_start's provenance) plus the Hub commit its files came from, when it was a Hub id."""
    if not init_source: return None
    resolved = Path(init_source["resolved"])
    return {**init_source, "revision": resolved.name if resolved.parent.name == "snapshots" else None}


def versions():
    out = {"python": platform.python_version()}
    for name in ("torch", "transformers", "peft"):
        try: out[name] = metadata.version(name)
        except metadata.PackageNotFoundError: out[name] = None
    return out


def device_record(dev, world=1):
    from .device import device_name
    return {"type": dev, "name": device_name(dev), "world_size": world}


def now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def run_provenance(a, dev, world, revision, init_source, rejected, started, ended=None):
    """The provenance.json of one kev.train run. rejected: {"train": {reason: count}, "val": ...} from the context filter."""
    base_sha, unresolved = resolved_revision(a.base, revision)
    return {"kev": code_commit(),
            "data": {"data": file_record(a.data), "val_data": file_record(a.val_data)},
            "base": {"id": a.base, "requested_revision": revision, "revision": base_sha, "unresolved": unresolved},
            "init": init_record(init_source), "seed": a.seed, "versions": versions(), "device": device_record(dev, world),
            "rejected_records": rejected, "started": started, "ended": ended}
