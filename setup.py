import setuptools

import pkg_resources
import pathlib
from distutils.util import convert_path

versions = {}
ver_path = convert_path('padl_ext/version.py')
with open(ver_path) as ver_file:
    exec(ver_file.read(), versions)

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read().split('\n')

long_description = ['# PADL-EXTENSIONS\n'] + [x for x in long_description if not x.strip().startswith('<img') and not x.strip().startswith('[!')]
long_description = '\n'.join(long_description)


def parse_requirements(filename):
    with pathlib.Path(filename).open() as requirements_txt:
        return [str(requirement) for requirement in pkg_resources.parse_requirements(requirements_txt)]


all_ = parse_requirements('requirements.txt')


trainer_extra = []


pytorch_lightning_extra = [
    "pytorch-lightning>=1.5.2",
]

torchserve_extra = [
    "torchserve",
    "torch-model-archiver",
]


setuptools.setup(
    name="padl-extensions",
    version=versions['__version__'],
    author="LF1",
    author_email="contact@lf1.io",
    description="Extensions for pytorch abstractions for deep learning",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/lf1-io/padl-extensions",
    packages=setuptools.find_packages(),
    setup_requires=[],
    license="Apache 2.0",
    classifiers=[
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        'Topic :: Scientific/Engineering',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
    ],
    python_requires='>=3.7',
    install_requires=all_,
    test_suite="tests",
    tests_require=all_ + parse_requirements('requirements-test.txt') + pytorch_lightning_extra
                  + torchserve_extra,
    package_data={'': ['requirements.txt']},
    include_package_data=True,
    extras_require={
        'all': all_
               + pytorch_lightning_extra
               + torchserve_extra
               + trainer_extra,
        'pytorch_lightning': pytorch_lightning_extra,
        'torchserve': torchserve_extra,
        'trainer': trainer_extra,
    }
)
