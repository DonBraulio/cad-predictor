from dynaconf import Dynaconf

settings = Dynaconf(
    envvar_prefix="DYNACONF",
    settings_files=[
        "params.yaml",
        "models.yaml",
        "diarization.yaml",
        "data/inputs.yaml",
        ".secrets.yaml",
    ],
)
