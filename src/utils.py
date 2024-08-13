import importlib.util


def is_available(pkg):
    return importlib.util.find_spec(pkg) is not None


def uintx_version() -> int:
    """
    1 - no uintx
    2 - uintx-prototype
    3 - uintx-dtype
    """
    uintx_proto = is_available("torchao.prototype.uintx")
    uintx_dtype = is_available("torchao.dtypes.uintx")
    return 1 if not (uintx_dtype or uintx_proto) else 2 if uintx_proto else 3
