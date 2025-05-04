from typing import Any, Dict, Union
from dataclasses import dataclass, field
import yaml
from pathlib import Path
from jsonschema import validate as js_validate, ValidationError
from omegaconf import DictConfig, OmegaConf

SCHEMA_PATH = Path(
    "C:/Users/tomic/ml/projects/dns_exfil_mbert" "/configs/bpe_tok_schema.json"
)
_schema = yaml.safe_load(SCHEMA_PATH.read_text(encoding="utf-8"))

DEFAULTS: Dict[str, Any] = {
    "alphabet": "abcdefghijklmnopqrstuvwxyz0123456789-._",
    "vocab_size": 8000,
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
class BpeTokConfig:

    config: Union[str, Path, Dict[str, Any]]
    _cfg: Dict[str, Any] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if isinstance(self.config, DictConfig):
            data = OmegaConf.to_container(self.config, resolve=True)
        elif isinstance(self.config, dict):
            data = self.config
        else:
            path = Path(self.config)
            if not path.exists():
                raise FileNotFoundError(f"Config file {path} does not exist.")
            data = yaml.safe_load(path.read_text(encoding="utf-8"))
            if not isinstance(data, dict):
                raise ValueError(
                    f"Config file {path} is not a valid YAML file."
                )
        try:
            js_validate(data, _schema)
        except ValidationError as e:
            source = path if 'path' in locals() else "in-memory dictConfig"
            raise ValueError(f"Config {source!r} failed schema validation: {e.message}")
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
        raise AttributeError(f"Attribute {name} not found in BpeTokConfig.")

    def __dir__(self):
        default = set(super().__dir__())
        keys = set(self._cfg.keys())
        return default | keys

    def __eq__(self, value):
        if isinstance(value, BpeTokConfig):
            return self._cfg == value._cfg
        if isinstance(value, dict):
            return self._cfg == value
        return False

    def resolve_defaults(self) -> Dict[str, Any]:
        out = {**DEFAULTS, **self._cfg}
        out["special_tokens"] = self.special_tokens
        return out

    def to_file(self, path: Union[str, Path]):
        data = self.resolve_defaults()
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(yaml.dump(data), encoding="utf-8")

    @classmethod
    def from_file(cls, path: Union[str, Path]) -> "BpeTokConfig":
        return cls(Path(path))


def get_config_for_bpe_tok(config_src: Union[str, Path, Dict[str, Any]]):
    if isinstance(config_src, dict):
        return BpeTokConfig(config_src)

    return BpeTokConfig.from_file(config_src)


if __name__ == "__main__":
    config_from_file_str = BpeTokConfig(
        "C:/Users/tomic/ml/projects/" "dns_exfil_mbert/configs/" "bpe_tok.yaml"
    )
    config_from_Path = BpeTokConfig(
        Path(
            "C:/Users/tomic/ml/projects/"
            "dns_exfil_mbert/configs/"
            "bpe_tok.yaml"
        )
    )
    yaml_dict = yaml.safe_load(
        Path(
            "C:/Users/tomic/ml/projects/"
            "dns_exfil_mbert/configs/"
            "bpe_tok.yaml"
        ).read_text(encoding="utf-8")
    )
    config_from_dict = BpeTokConfig(yaml_dict)
    config_from_get = get_config_for_bpe_tok(
        "C:/Users/tomic/ml/projects/" "dns_exfil_mbert/configs/" "bpe_tok.yaml"
    )
    print(config_from_file_str.max_length)
    assert config_from_file_str == config_from_Path
    assert config_from_get == config_from_dict
    assert config_from_get == config_from_file_str
    print("All tests passed.")
