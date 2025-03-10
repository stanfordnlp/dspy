"""Types for aiohappyeyeballs."""

import socket
from collections.abc import Callable
from typing import Tuple, Union

AddrInfoType = Tuple[
    Union[int, socket.AddressFamily],
    Union[int, socket.SocketKind],
    int,
    str,
    Tuple,  # type: ignore[type-arg]
]

SocketFactoryType = Callable[[AddrInfoType], socket.socket]
