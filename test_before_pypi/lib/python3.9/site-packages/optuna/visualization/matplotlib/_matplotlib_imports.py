from packaging import version

from optuna._imports import try_import


with try_import() as _imports:
    # TODO(ytknzw): Add specific imports.
    import matplotlib
    from matplotlib import __version__ as matplotlib_version
    from matplotlib import pyplot as plt
    from matplotlib.axes._axes import Axes
    from matplotlib.collections import LineCollection
    from matplotlib.collections import PathCollection
    from matplotlib.colors import Colormap
    from matplotlib.contour import ContourSet
    from matplotlib.figure import Figure

    # TODO(ytknzw): Set precise version.
    if version.parse(matplotlib_version) < version.parse("3.0.0"):
        raise ImportError(
            "Your version of Matplotlib is " + matplotlib_version + " . "
            "Please install Matplotlib version 3.0.0 or higher. "
            "Matplotlib can be installed by executing `$ pip install -U matplotlib>=3.0.0`. "
            "For further information, please refer to the installation guide of Matplotlib. ",
            name="matplotlib",
        )

__all__ = [
    "_imports",
    "matplotlib",
    "matplotlib_version",
    "plt",
    "Axes",
    "LineCollection",
    "PathCollection",
    "Colormap",
    "ContourSet",
    "Figure",
]
