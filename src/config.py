import json
import types
from pathlib import Path

_ROOT = Path(__file__).parent.parent


def _to_namespace(d: dict) -> types.SimpleNamespace:
    ns = types.SimpleNamespace()
    for k, v in d.items():
        setattr(ns, k, _to_namespace(v) if isinstance(v, dict) else v)
    return ns


class Config:
    _instance: "Config | None" = None

    def __init__(self, data: dict) -> None:
        self._ns = _to_namespace(data)
        for k, v in vars(self._ns.paths).items():
            if isinstance(v, str):
                setattr(self._ns.paths, k, _ROOT / v)

    def __getattr__(self, name: str):
        return getattr(self._ns, name)

    @classmethod
    def get(cls) -> "Config":
        if cls._instance is None:
            config_path = _ROOT / "config.json"
            if not config_path.exists():
                raise FileNotFoundError(
                    f"config.json tidak ditemukan di: {config_path.resolve()}"
                )
            with open(config_path, encoding="utf-8") as f:
                data = json.load(f)
            cls._instance = cls(data)
            cls._instance._apply_cuda_fallback()
        return cls._instance

    def _apply_cuda_fallback(self) -> None:
        if self._ns.sam.device == "cuda":
            try:
                import torch
                if not torch.cuda.is_available():
                    self._ns.sam.device = "cpu"
            except ImportError:
                self._ns.sam.device = "cpu"

    @classmethod
    def force_cpu(cls) -> None:
        cls.get()._ns.sam.device = "cpu"
