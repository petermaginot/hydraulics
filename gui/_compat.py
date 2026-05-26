"""Compatibility shims that must run before NodeGraphQt is imported.

NodeGraphQt 0.6.44 still uses distutils.version.LooseVersion, which was
removed from the standard library in Python 3.12.  We inject a stub backed
by packaging.version.Version (already in our transitive deps via Pint)
*before* any NodeGraphQt import resolves the name.

Import this module exactly once, before importing NodeGraphQt.
"""

import sys
import types

from PySide6.QtCore import QtMsgType, qInstallMessageHandler


def _qt_message_handler(msg_type, context, message):
    # NodeGraphQt 0.6.44 uses QMouseEvent.pos() which Qt 6 deprecated in
    # favour of QMouseEvent.position().  The method still works; suppress the
    # noise until NodeGraphQt is updated upstream.
    if "QMouseEvent.pos() const" in message and "deprecated" in message:
        return
    if msg_type == QtMsgType.QtWarningMsg:
        print(f"Qt warning: {message}", file=sys.stderr)
    elif msg_type == QtMsgType.QtCriticalMsg:
        print(f"Qt critical: {message}", file=sys.stderr)
    elif msg_type == QtMsgType.QtFatalMsg:
        print(f"Qt fatal: {message}", file=sys.stderr)


qInstallMessageHandler(_qt_message_handler)


def _install_distutils_shim():
    if "distutils.version" in sys.modules:
        return
    try:
        # If the real distutils is still around (Python <= 3.11), use it.
        from distutils.version import LooseVersion  # noqa: F401
        return
    except ImportError:
        pass

    from packaging.version import Version

    distutils_mod = sys.modules.setdefault("distutils", types.ModuleType("distutils"))
    version_mod = types.ModuleType("distutils.version")
    # NodeGraphQt only uses LooseVersion for ordered comparison against the
    # Qt version string.  packaging.version.Version handles that with
    # stricter semantics, which is fine for the simple "Qt >= 5.10" check.
    version_mod.LooseVersion = Version
    distutils_mod.version = version_mod
    sys.modules["distutils.version"] = version_mod


_install_distutils_shim()
