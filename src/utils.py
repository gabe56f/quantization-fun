import importlib.util


def is_available(pkg):
    return importlib.util.find_spec(pkg) is not None
