"""Figures from the Scope.

export_view() saves the panels of a tab as PNG, SVG or PDF, drawn in a light
theme (white background, darker trace colours) whatever the window's theme.
render_figure() draws a whole recording offscreen -- the <run>_traces.png saved
after every offline analysis (`python -m kinefly2.apps.scope --figure`).
"""
import contextlib
import os

import pyqtgraph as pg
import pyqtgraph.exporters  # noqa: F401  (makes pg.exporters available)
from PySide6 import QtCore, QtGui, QtSvg, QtWidgets

from . import data
from .view import LIGHT, ScopeView

FORMATS = ('.png', '.svg', '.pdf')


def _pdf(glw, path):
    """Vector PDF of the layout: the scene rendered onto a page of its own size."""
    rect = glw.ci.sceneBoundingRect()
    writer = QtGui.QPdfWriter(path)
    writer.setPageSize(QtGui.QPageSize(QtCore.QSizeF(rect.width(), rect.height()),
                                       QtGui.QPageSize.Unit.Point))
    writer.setPageMargins(QtCore.QMarginsF(0, 0, 0, 0))
    writer.setResolution(144)
    painter = QtGui.QPainter(writer)
    try:
        glw.scene().render(painter, QtCore.QRectF(0, 0, writer.width(), writer.height()), rect)
    finally:
        painter.end()


def _svg(glw, path):
    """Vector SVG of the layout, through Qt's own generator. (pyqtgraph's
    SVGExporter post-processes Qt's path data expecting "x,y" pairs, which Qt
    6.11 no longer writes, and fails.)"""
    rect = glw.ci.sceneBoundingRect()
    gen = QtSvg.QSvgGenerator()
    gen.setFileName(path)
    gen.setSize(QtCore.QSize(int(rect.width()), int(rect.height())))
    gen.setViewBox(QtCore.QRectF(0, 0, rect.width(), rect.height()))
    gen.setTitle('Kinefly2 Scope')
    painter = QtGui.QPainter(gen)
    try:
        glw.scene().render(painter, QtCore.QRectF(0, 0, rect.width(), rect.height()), rect)
    finally:
        painter.end()


def export_view(view, path, width=1800, light=True):
    """Save a tab's panels to `path` (.png, .svg or .pdf)."""
    ext = os.path.splitext(path)[1].lower()
    if ext not in FORMATS:
        raise ValueError('save the figure as .png, .svg or .pdf, not %r' % (ext or path))
    with (view.themed(LIGHT) if light else contextlib.nullcontext()):
        QtWidgets.QApplication.processEvents()
        if ext == '.png':
            exp = pg.exporters.ImageExporter(view.glw.ci)
            exp.parameters()['width'] = int(width)
            exp.export(path)
        elif ext == '.svg':
            _svg(view.glw, path)
        else:
            _pdf(view.glw, path)
    return path


def render_figure(path, out, width=1600, panel_px=150):
    """Draw every trace of a recording into `out`, offscreen: the panels that
    have data (default visibility, not this machine's remembered choices, so
    the figure depends only on the file), the whole recording wide, and each
    panel fitted to its data -- a figure must not clip what it summarises."""
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(['kinefly2-figure'])
    pg.setConfigOptions(antialias=True)
    rec = data.load_recording(path)
    view = ScopeView(rec, settings=None, figure=True)
    view.resize(int(width), int(70 + panel_px * max(1, len(view.panels))))
    view.show()
    for _ in range(3):
        app.processEvents()
    view.show_all()
    view.autoscale_all()
    app.processEvents()
    export_view(view, out, width=width, light=True)
    view.close()
    return out
