"""Runtime adapter: the ONLY module that knows which runtime it is running on.

Detection order: explicit RUN_ENV (set by the submission scripts) -> Kaggle markers -> local.
Optional overrides: OUTPUT_ROOT, DATA_ROOT, HAS_INTERNET (0/1), MAX_RUNTIME_S.
"""
import functools
import os
import shlex
import socket
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

KAGGLE_WORKING = Path("/kaggle/working")
KAGGLE_MAX_RUNTIME_S = 9 * 3600


class Runtime(StrEnum):
    LOCAL = "local"
    KAGGLE = "kaggle"
    HF_JOBS = "hf_jobs"


@dataclass(frozen=True)
class Env:
    runtime: Runtime
    output_root: Path       # durable; survives job termination
    data_root: str          # local path OR remote URI
    has_internet: bool
    max_runtime_s: int | None

    def secret(self, name: str) -> str | None:
        """Runtime-appropriate secret lookup. Never raises on absence."""
        if self.runtime is Runtime.KAGGLE:
            try:
                from kaggle_secrets import UserSecretsClient

                return UserSecretsClient().get_secret(name)
            except Exception:
                pass
        return os.environ.get(name) or None

    def resolve_data_uri(self, uri: str) -> str:
        """A relative data URI is resolved against this runtime's data root; absolute paths,
        URIs with a scheme, and ./ or ../ paths (cwd-relative) are returned unchanged."""
        if "://" in uri or Path(uri).is_absolute() or uri.startswith(("./", "../", ".\\")):
            return uri
        root = self.data_root
        if "://" in root:
            return f"{root.rstrip('/')}/{uri}"
        return str(Path(root).resolve() / uri)

    def hydra_argv(self, argv: list[str]) -> list[str]:
        """Hydra overrides arrive as CLI args, except on Kaggle (single code file), where the
        submitter puts them in HYDRA_OVERRIDES."""
        if self.runtime is Runtime.KAGGLE:
            return [argv[0], *shlex.split(os.environ.get("HYDRA_OVERRIDES", ""))]
        return list(argv)


def _detect_runtime() -> Runtime:
    if explicit := os.environ.get("RUN_ENV"):
        return Runtime(explicit.strip().lower())
    if os.environ.get("KAGGLE_KERNEL_RUN_TYPE") or KAGGLE_WORKING.exists():
        return Runtime.KAGGLE
    return Runtime.LOCAL


@functools.lru_cache(maxsize=1)
def _probe_internet(timeout: float = 2.0) -> bool:
    try:
        socket.create_connection(("huggingface.co", 443), timeout=timeout).close()
        return True
    except OSError:
        return False


def detect() -> Env:
    runtime = _detect_runtime()
    if runtime is Runtime.KAGGLE:
        output_root, data_root, max_s = KAGGLE_WORKING, "/kaggle/input", KAGGLE_MAX_RUNTIME_S
    elif runtime is Runtime.HF_JOBS:
        output_root, data_root, max_s = Path("/ckpt"), "hf://datasets", None
    else:
        output_root, data_root, max_s = Path("outputs").resolve(), "data", None

    if "HAS_INTERNET" in os.environ:
        has_internet = os.environ["HAS_INTERNET"].strip().lower() in ("1", "true", "yes")
    else:
        # Kaggle has internet only if the kernel enables it; the others always do.
        has_internet = _probe_internet() if runtime is Runtime.KAGGLE else True

    return Env(
        runtime=runtime,
        output_root=Path(os.environ.get("OUTPUT_ROOT", output_root)),
        data_root=os.environ.get("DATA_ROOT", data_root),
        has_internet=has_internet,
        max_runtime_s=int(os.environ["MAX_RUNTIME_S"]) if "MAX_RUNTIME_S" in os.environ else max_s,
    )


def get_output_root(output_dir: str | None) -> Path:
    """Durable output root: explicit config override, else the runtime's root."""
    root = Path(output_dir).resolve() if output_dir else detect().output_root
    root.mkdir(parents=True, exist_ok=True)
    return root
