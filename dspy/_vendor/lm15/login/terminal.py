"""
lm15.login.terminal — the terminal adapter for AUTH-16.

Prints notices to stderr, asks with ``input()``/``getpass``.  Opens a
browser only when the application says so (``open_browser=True``): a URL
is otherwise printed for the person to open, which is what a machine
reached over SSH needs.  Ctrl-C at a prompt cancels the login
(``LoginCancelled``); nothing is retried.
"""

from __future__ import annotations

import getpass
import sys
from typing import Any, TextIO

from .types import (
    AuthUrlNotice,
    DeviceCodeNotice,
    InfoNotice,
    ManualCodePrompt,
    Notice,
    ProgressNotice,
    Prompt,
    SecretPrompt,
    SelectPrompt,
    TextPrompt,
)

__all__ = ["TerminalUI"]


class TerminalUI:
    def __init__(self, *, out: TextIO | None = None, inp: Any = None, open_browser: bool = False) -> None:
        self._out = out if out is not None else sys.stderr
        self._input = inp if inp is not None else input
        self._open_browser = open_browser
        self._stale: set[int] = set()

    def _say(self, text: str) -> None:
        self._out.write(text + "\n")
        self._out.flush()

    def notify(self, notice: Notice) -> None:
        if isinstance(notice, AuthUrlNotice):
            self._say(f"\nOpen this link to sign in:\n  {notice.url}\n{notice.instructions}")
            if self._open_browser:
                self._open(notice.url)
        elif isinstance(notice, DeviceCodeNotice):
            self._say(f"\nOpen {notice.verification_url}\nand enter this code:  {notice.user_code}\n"
                      f"(the code is valid for about {int(notice.expires_in_s // 60)} minutes)")
            if self._open_browser:
                self._open(notice.verification_url)
        elif isinstance(notice, ProgressNotice):
            self._say(f"… {notice.message}")
        elif isinstance(notice, InfoNotice):
            links = "".join(f"\n  {label}: {url}" for label, url in notice.links)
            self._say(notice.message + links)

    def _open(self, url: str) -> None:
        if not url.startswith("https://"):
            return  # AUTH-18: never launch a scheme a provider chose
        try:
            import webbrowser

            webbrowser.open(url, new=2)
        except Exception:
            pass

    def prompt(self, prompt: Prompt) -> str:
        if isinstance(prompt, SecretPrompt):
            return getpass.getpass(f"{prompt.label}: ")
        if isinstance(prompt, SelectPrompt):
            self._say(f"\n{prompt.label}")
            for index, option in enumerate(prompt.options, 1):
                note = f"  — {option.description}" if option.description else ""
                self._say(f"  {index}. {option.label}{note}")
            while True:
                raw = self._input("Choose a number: ").strip()
                if raw.isdigit() and 1 <= int(raw) <= len(prompt.options):
                    return prompt.options[int(raw) - 1].id
                for option in prompt.options:
                    if raw == option.id:
                        return option.id
                self._say("Not one of the choices.")
        if isinstance(prompt, ManualCodePrompt):
            return self._input(f"{prompt.label}\n> ")
        if isinstance(prompt, TextPrompt):
            hint = f" [{prompt.placeholder}]" if prompt.placeholder else ""
            return self._input(f"{prompt.label}{hint}: ")
        raise TypeError(f"unknown prompt {type(prompt).__name__}")

    def dismiss(self, prompt: Prompt) -> None:
        self._say("\n(the browser finished — press Enter to continue)")
