import os
import subprocess
import pathlib


def prepare(target_model, version="1.0", force=False):
    """Prepare and package model into archive `mar` file with TorchModelArchiver.

    Note: When building the command double quotes needs to be used for compatibility with
    python 3.7 on Windows.

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
        "torch-model-archiver",
        "--model-name", str(model_name),
        "--version", str(version),
        "--export-path", str(model_parent),
        "--extra-files", str(target_model),
        "--handler", str(handler_dir),
    ]
    if force:
        cmd += ["--force"]
    subprocess.run(cmd, check=True)


def serve(model_store,
          model,
          ncs=True,
          workflow_store=None,
          ts_config=None,
          log_config=None,
          timeout=None):
    """Serve model with TorchServe.

    Note: When building the command double quotes needs to be used for compatibility with
    python 3.7 on Windows.

    :param model_store: directory where model is stored
    :param model: model file name - `.mar`
    :param ncs: Disable snapshot feature
    :param ts_config: Configuration file for TorchServe
    :param workflow_store: Workflow store location where workflow can be loaded. Defaults to model_store
    :param log_config: Log4j configuration file for TorchServe
    :param timeout: If provided a TimeoutExpired exception will be raised after the length
        specified.
    """
    cmd = [
        "torchserve",
        "--start",
        "--model-store", str(model_store),
        "--models", str(model),
    ]
    if ncs:
        cmd += ["--ncs"]
    if workflow_store is not None:
        cmd += ["--workflow-store", str(workflow_store)]
    if ts_config is not None:
        cmd += ["--ts-config", str(ts_config)]
    if log_config is not None:
        cmd += ["--log-config", str(log_config)]
    cmd += ["--foreground"]
    subprocess.run(cmd, timeout=timeout, check=True)


def stop():
    """Stop TorchServe model-server.

    Note: When building the command double quotes needs to be used for compatibility with
    python 3.7 on Windows.
    """
    subprocess.run(["torchserve", "--stop"], check=True)


def prepare_and_serve(target_model,
                      version="1.0",
                      force=False,
                      ncs=True,
                      workflow_store=None,
                      ts_config=None,
                      log_config=None,
                      timeout=None):
    """Package model and serve with TorchServe.

    :param target_model: PADL serialized model - `.padl`
    :param version: version of model
    :param force: if force=True, existing archive file is overwritten
    :param ncs: Disable snapshot feature for TorchServe
    :param ts_config: Configuration file for TorchServe
    :param workflow_store: Workflow store location for TorchServe where workflow can be loaded.
        Defaults to model_store
    :param log_config: Log4j configuration file for TorchServe
    :param timeout: If provided a TimeoutExpired exception will be raised after the length
        specified.
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

    serve(model_parent, mar_name, ncs=ncs, workflow_store=workflow_store, ts_config=ts_config,
          log_config=log_config, timeout=timeout)
