from typing import Any, Dict, Union
from dataclasses import dataclass, field
import yaml
from pathlib import Path

DEFAULTS: Dict[str, Any] = {
    "alphabet": "abcdefghijklmnopqrstuvwxyz0123456789-_.",
    "max_length": 256,
    "pad_token": "[PAD]",
    "unk_token": "[UNK]",
    "cls_token": "[CLS]",
    "sep_token": "[SEP]",
    "mask_token": "[MASK]",
    "padding": True,
    "truncation": True,
}


@dataclass
class CharTokConfig:

    config: Union[str, Path, Dict[str, Any]]
    _cfg: Dict[str, Any] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if isinstance(self.config, dict):
            object.__setattr__(self, "_cfg", self.config)
            return

        path = Path(self.config)
        if not path.exists():
            raise FileNotFoundError(f"Config file {path} does not exist.")
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise ValueError(f"Config file {path} is not a valid YAML file.")
        object.__setattr__(self, "_cfg", data)

    def __getattr__(self, name: str) -> Any:
        if name == "special_tokens":
            base = {
                k: getattr(self, k)
                for k in (
                    "pad_token",
                    "unk_token",
                    "cls_token",
                    "sep_token",
                    "mask_token",
                )
            }
            override = self._cfg.get("special_tokens", {})
            return {**base, **override}

        if name in DEFAULTS:
            return self._cfg.get(name, DEFAULTS[name])
        raise AttributeError(f"Attribute {name} not found in CharTokConfig.")

    def __dir__(self):
        default = set(super().__dir__())
        keys = set(self._cfg.keys())
        return default | keys

    def to_file(self, path: Union[str, Path]) -> None:
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(yaml.safe_dump(self._cfg), encoding="utf-8")

    @classmethod
    def from_yaml(cls, config_path: Union[str, Path]) -> "CharTokConfig":
        return cls(Path(config_path))


def get_config_for_char_tok(
    config_src: Union[str, Path, Dict[str, Any]],
) -> CharTokConfig:
    if isinstance(config_src, dict):
        return CharTokConfig(config_src)

    return CharTokConfig.from_yaml(config_src)


if __name__ == "__main__":
    DIR = (
        Path(__file__)
        .resolve()
        .parent.parent.parent.parent.parent.parent.parent
    )
    config_path = DIR / "configs" / "tokenizer_char.yaml"
    cfg = get_config_for_char_tok(config_path)
    print(cfg.special_tokens["pad_token"])
    print(cfg.padding)
    print(dir(cfg))
    print("Done")
