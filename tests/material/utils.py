import sys
from importlib.metadata import DistributionFinder


def _check_if_module_installed(module_name):
    """Check if a module is installed.

    :param module_name: Name of Module
    :return: Bool if module is installed or not
    """
    distribution_instance = filter(None, (getattr(finder, 'find_distributions', None) for finder in sys.meta_path))
    for res in distribution_instance:
        dists = res(DistributionFinder.Context(name=module_name))
        dist = next(iter(dists), None)
        if dist is not None:
            return True
    else:
        return False
