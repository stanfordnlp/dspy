def check_package_available(package_name, caller_name, pip_name=None, extra=None):
    """
    Import a package and return the imported module if available, otherwise raises an ImportError.
    """
    try:
        import importlib

        return importlib.import_module(package_name)
    except ImportError as e:
        pip_name = pip_name or package_name
        if extra:
            install_cmd = f"`pip install 'dspy[{extra}]'`"
        else:
            install_cmd = f"`pip install {pip_name}`"

        raise ImportError(
            f"{caller_name} requires {package_name}. You can install it with {install_cmd}"
        ) from e


def is_package_available(package_name):
    try:
        import importlib

        importlib.import_module(package_name)
        return True
    except ImportError:
        return False


# Convenience functions for common optional dependencies
def check_numpy_support(caller_name):
    return check_package_available("numpy", caller_name, extra="retrieval")


def has_numpy():
    return is_package_available("numpy")
