from azureml.core import Workspace
from azureml.core.model import Model
from cgmml.models.HRNET.code.config import cfg, update_config


def download_model(workspace, model_name, target_path):
    '''
    Download the pretrained model
    Input:
         workspace: workspace to access the experiment
         model_name: Name of the model in which model is registered
         target_path: Where model should download
    '''
    model = Model(workspace, name=model_name)
    model.download(target_dir=target_path, exist_ok=True, exists_ok=None)


if __name__ == '__main__':
    workspace = Workspace.from_config()

    args = 'cgmml/models/HRNET/inference-config-hrnet.yaml'
    update_config(cfg, args)
    download_model(workspace=workspace, model_name=cfg.MODEL.NAME, target_path='cgmml/models/HRNET')
