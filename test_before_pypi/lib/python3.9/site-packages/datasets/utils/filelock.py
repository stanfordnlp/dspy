# deprecated, please use the `filelock` package instead

from filelock import (  # noqa: F401 # imported for backward compatibility TODO: remove in 3.0.0
    BaseFileLock,
    SoftFileLock,
    Timeout,
    UnixFileLock,
    WindowsFileLock,
)

from ._filelock import FileLock  # noqa: F401 # imported for backward compatibility. TODO: remove in 3.0.0
