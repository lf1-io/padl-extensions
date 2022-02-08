import os
import subprocess


def prepare(target_model, version='1.0', force=False):
    """Prepare and package model into archive `mar` file with TorchModelArchiver.

    :param target_model: PADL serialized model - `.padl`
    :param version: version of model
    :param force: if force=True, existing archive file is overwritten
    :return:
    """
    current_directory = '/'.join(__file__.split('/')[:-1])
    handler_dir = os.path.join(current_directory, 'handler.py')
    print(f'handler is {handler_dir}')
    model_parent = '/'.join(target_model.split('/')[:-1])
    model_name = target_model.split('/')[-1].split('.padl')[0]
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


def serve(model_store, model, foreground=True):
    """Serve model with TorchServe.

    :param model_store: directory where model is stored
    :param model: model file name - `.mar`
    :param foreground: runs TorchServe in the foreground. If this option is disabled, TorchServe runs in the background
    :return:
    """
    cmd = [
        'torchserve',
        '--start', '--ncs',
        '--model-store', model_store,
        '--models', model,
    ]
    if foreground:
        cmd += ['--foreground']
    subprocess.run(cmd)


def prepare_and_serve(target_model, version='1.0', force=False, foreground=True):
    """Package model and serve with TorchServe.

    :param target_model: PADL serialized model - `.padl`
    :param version: version of model
    :param force: if force=True, existing archive file is overwritten
    :param foreground: runs TorchServe in the foreground. If this option is disabled, TorchServe runs in the background
    :return:
    """
    model_parent = '/'.join(target_model.split('/')[:-1])
    model_name = target_model.split('/')[-1].split('.padl')[0]
    mar_name = model_name + '.mar'

    if force:
        prepare(target_model, version=version, force=force)
    elif not os.path.exists(model_parent + '/' + mar_name):
        prepare(target_model, version=version, force=force)
    else:
        print(f'Torch model archive already exists for {target_model}, skipping archiving...')

    print(f'converting {model_name} to MAR format...')
    print('Serving the model...')

    serve(model_parent, mar_name, foreground=foreground)
