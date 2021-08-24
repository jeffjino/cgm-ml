from pathlib import Path

from azureml.core import Environment, Workspace


def cgm_environment(workspace: Workspace,
                    curated_env_name: str,
                    env_exist: bool,
                    fpath_env_yml: Path = None) -> Environment:
    if env_exist:
        return Environment.get(workspace=workspace, name=curated_env_name)
    if fpath_env_yml is None:
        from cgmml.common.endpoints.constants import REPO_DIR  # noqa: E402
        fpath_env_yml = REPO_DIR / "environment_train.yml"
    else:
        fpath_env_yml = Path(fpath_env_yml)
    cgm_env = Environment.from_conda_specification(name=curated_env_name, file_path=fpath_env_yml)
    cgm_env.docker.enabled = True
    cgm_env.docker.base_image = 'mcr.microsoft.com/azureml/openmpi4.1.0-cuda11.0.3-cudnn8-ubuntu18.04'
    cgm_env.register(workspace)
    return cgm_env
