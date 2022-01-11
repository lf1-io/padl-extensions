import os
import subprocess


def prepare(checkpoint, version='1.0'):
    current_directory = '/'.join(__file__.split('/')[:-1])
    handler_dir = os.path.join(current_directory, 'handler.py')
    print(f'handler is {handler_dir}')
    model_parent = '/'.join(checkpoint.split('/')[:-1])
    model_name = checkpoint.split('/')[-1].split('.padl')[0]
    print('converting current transform to MAR format...')
    subprocess.run([
        'torch-model-archiver',
        '--model-name', model_name,
        '--version', version,
        '--export-path', model_parent,
        '--extra-files', checkpoint,
        '--handler', handler_dir,
    ])
