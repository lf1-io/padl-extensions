import os
import subprocess
import pathlib


def prepare(target_model, version='1.0', force=False):
    """Prepare and package model into archive `mar` file with TorchModelArchiver.

    :param target_model: PADL serialized model - `.padl`
    :param version: version of model
    :param force: if force=True, existing archive file is overwritten
    """

    target_model = pathlib.Path(target_model)
    model_parent = target_model.parents[0]
    model_name = target_model.stem
    current_directory = pathlib.Path(__file__).parent
    handler_dir = current_directory/'handler.py'

    print(f'handler is {handler_dir}')
    print(f'converting {model_name} to MAR format...')

    cmd = [
        'torch-model-archiver',
        '--model-name', model_name,
        '--version', version,
        '--export-path', model_parent,
        '--extra-files', target_model,
        '--handler', handler_dir,
    ]
    if force:
        cmd += ['--force']
    subprocess.run(cmd)


def serve(model_store,
          model,
          ncs=True,
          workflow_store=None,
          ts_config=None,
          log_config=None):
    """Serve model with TorchServe.

    :param model_store: directory where model is stored
    :param model: model file name - `.mar`
    :param ncs: Disable snapshot feature
    :param ts_config: Configuration file for TorchServe
    :param workflow_store: Workflow store location where workflow can be loaded. Defaults to model_store
    :param log_config: Log4j configuration file for TorchServe
    """
    cmd = [
        'torchserve',
        '--start',
        '--model-store', model_store,
        '--models', model,
    ]
    if ncs:
        cmd += ['--ncs']
    if workflow_store is not None:
        cmd += ['--workflow-store', workflow_store]
    if ts_config is not None:
        cmd += ['--ts-config', ts_config]
    if log_config is not None:
        cmd += ['--log-config', log_config]
    cmd += ['--foreground']
    subprocess.run(cmd)


def stop():
    """Stop TorchServe model-server."""
    subprocess.run(['torchserve', '--stop'])


def prepare_and_serve(target_model,
                      version='1.0',
                      force=False,
                      ncs=True,
                      workflow_store=None,
                      ts_config=None,
                      log_config=None):
    """Package model and serve with TorchServe.

    :param target_model: PADL serialized model - `.padl`
    :param version: version of model
    :param force: if force=True, existing archive file is overwritten
    :param ncs: Disable snapshot feature for TorchServe
    :param ts_config: Configuration file for TorchServe
    :param workflow_store: Workflow store location for TorchServe where workflow can be loaded. Defaults to model_store
    :param log_config: Log4j configuration file for TorchServe
    """
    target_model = pathlib.Path(target_model)
    model_parent = target_model.parent
    model_name = target_model.stem
    mar_name = pathlib.Path(f'{model_name}.mar')

    if force or (not os.path.exists(model_parent/mar_name)):
        prepare(target_model, version=version, force=force)
    else:
        print(f'Torch model archive already exists for {target_model}, skipping archiving...')

    print(f'converting {model_name} to MAR format...')
    print('Serving the model...')

    serve(model_parent, mar_name, ncs=ncs, workflow_store=workflow_store, ts_config=ts_config, log_config=log_config)
