from packaging import version

from optuna._imports import try_import


with try_import() as _imports:
    import plotly
    from plotly import __version__ as plotly_version
    import plotly.graph_objects as go
    from plotly.graph_objects import Contour
    from plotly.graph_objects import Scatter
    from plotly.subplots import make_subplots

    if version.parse(plotly_version) < version.parse("4.0.0"):
        raise ImportError(
            "Your version of Plotly is " + plotly_version + " . "
            "Please install plotly version 4.0.0 or higher. "
            "Plotly can be installed by executing `$ pip install -U plotly>=4.0.0`. "
            "For further information, please refer to the installation guide of plotly. ",
            name="plotly",
        )

__all__ = ["_imports", "plotly", "go", "Contour", "Scatter", "make_subplots"]
