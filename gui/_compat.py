"""Compatibility shims that must run before NodeGraphQt is imported.

Two shims live here:

1. NodeGraphQt 0.6.44 still uses distutils.version.LooseVersion, which was
   removed from the standard library in Python 3.12.  We inject a stub backed
   by packaging.version.Version (already in our transitive deps via Pint)
   *before* any NodeGraphQt import resolves the name.
2. NodeGraphQt's viewer calls QMouseEvent.pos(), deprecated in Qt 6 in favour
   of QMouseEvent.position().  PySide6 reports that as a Python
   DeprecationWarning (not a Qt log message), and the call sits inside
   mouseMoveEvent, so it fires on every mouse move over the node canvas.  We
   filter it out below.  Nothing in this repo calls the deprecated API — the
   fix belongs upstream.

Import this module exactly once, before importing NodeGraphQt.
"""

import sys
import types
import warnings

from PySide6.QtCore import QtMsgType, qInstallMessageHandler

def install_warning_filters():
    """Suppress the NodeGraphQt QMouseEvent.pos() DeprecationWarning.

    Called at import, but must be re-called inside any
    ``warnings.catch_warnings()`` block that runs ``simplefilter("always")``:
    that prepends a catch-all filter which shadows this one.  It matters
    because the compressible solver pumps ``QApplication.processEvents()``
    from its progress callback, so NodeGraphQt mouse handlers dispatch
    *inside* the solver's capture block.
    """
    # Narrow on purpose: a blanket DeprecationWarning filter would also hide
    # real signals from our own dependencies.
    warnings.filterwarnings(
        "ignore",
        message=r".*QMouseEvent\.pos\(\).*",
        category=DeprecationWarning,
    )


install_warning_filters()


def _qt_message_handler(msg_type, context, message):
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
