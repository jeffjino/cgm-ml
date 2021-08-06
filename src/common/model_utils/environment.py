
from pathlib import Path

from azureml.core import Environment, Workspace

from common.endpoints.constants import REPO_DIR  # noqa: E402


def cgm_environment(workspace: Workspace,
                    curated_env_name: str,
                    env_exist: bool,
                    fpath_env_yml: Path = None) -> Environment:
    if env_exist:
        return Environment.get(workspace=workspace, name=curated_env_name)
    fpath_env_yml = REPO_DIR / "environment_train.yml" if fpath_env_yml is None else Path(fpath_env_yml)
    cgm_env = Environment.from_conda_specification(name=curated_env_name, file_path=fpath_env_yml)
    cgm_env.docker.enabled = True
    cgm_env.docker.base_image = 'mcr.microsoft.com/azureml/openmpi4.1.0-cuda11.0.3-cudnn8-ubuntu18.04'
    return cgm_env
