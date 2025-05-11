import hydra
from hydra.core.hydra_config import HydraConfig
from pathlib import Path

if __name__ == "__main__":
    with hydra.initialize_config_dir(
        config_dir=str(Path.cwd() / "configs"),
        job_name="char_tok_test",
        version_base="1.3",
    ):
        cfg = hydra.compose(
            config_name="config",
            overrides=[
                "tokenizer=char",
            ],
            return_hydra_config=True,
        )
        HydraConfig().set_config(cfg)
        tokenizer = hydra.utils.instantiate(cfg.tokenizer)
        print(tokenizer)
