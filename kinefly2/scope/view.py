"""One Scope tab: stacked, time-linked panels for a saved recording or the
live stream, an overview strip to navigate, two measurement cursors, and
markers for stimulus onsets, recording start/stop and logged events.

Mouse, Axoscope-style:
    wheel / drag inside a panel      time: zoom / pan -- every panel together
    Shift + wheel / drag             this panel's amplitude instead
    wheel / drag on a y axis         this panel's amplitude
    double-click a panel             fit its amplitude to what is visible
    right-click                      pyqtgraph's menu (view all, export, ...)
Amplitude ranges otherwise stay where they are put -- a chart recorder, not an
autoscaling plot, so a value now compares with the same value a minute ago --
and are remembered per channel group.

Curves are drawn with connect='finite': consecutive samples are always joined
and only a missing value (NaN) breaks the line, however far you zoom in.
(The browser dashboard used to lift the pen at every pixel column without a
sample, so zooming in made every trace vanish.)
"""
import contextlib
import math

import numpy as np
import pyqtgraph as pg
from PySide6 import QtCore, QtGui, QtWidgets

from . import data

Qt = QtCore.Qt
DARK = {'bg': '#141414', 'fg': '#b8b8b8', 'title': '#eeeeee', 'dim': '#8a8a8a', 'grid': 0.13}
LIGHT = {'bg': '#ffffff', 'fg': '#303030', 'title': '#111111', 'dim': '#666666', 'grid': 0.20}
MARKERS = {'stim': ('#ffa060', Qt.PenStyle.DashLine), 'rec': ('#ff5050', Qt.PenStyle.SolidLine),
           'event': ('#5fb8ff', Qt.PenStyle.DotLine), 'reset': ('#9a9a9a', Qt.PenStyle.SolidLine)}
CURSOR_COLORS = ('#ffe14d', '#4dd2ff')
SPANS = (2, 5, 10, 30, 60, 120, 300)
AXIS_W = 58
SHIFT_OR_CTRL = Qt.KeyboardModifier.ShiftModifier | Qt.KeyboardModifier.ControlModifier


class TimeViewBox(pg.ViewBox):
    """Wheel and drag inside a panel act on TIME, which every panel shares;
    with Shift (or Ctrl) they act on this panel's amplitude. Wheel and drag on
    the y axis reach here with axis=1 and scale the amplitude as they are."""

    def _axis(self, ev, axis):
        if axis is not None:
            return axis
        return 1 if ev.modifiers() & SHIFT_OR_CTRL else 0

    def wheelEvent(self, ev, axis=None):
        super().wheelEvent(ev, axis=self._axis(ev, axis))

    def mouseDragEvent(self, ev, axis=None):
        super().mouseDragEvent(ev, axis=self._axis(ev, axis))


def nice(v):
    if v is None or not np.isfinite(v):
        return '–'
    a = abs(v)
    return '%.0f' % v if a >= 100 else '%.1f' % v if a >= 10 else '%.2f' % v


def pen_color(color, light):
    """A trace colour readable on the theme's background: the dark theme's
    bright yellows vanish on white, so a light figure darkens them."""
    return QtGui.QColor(color).darker(165) if light else QtGui.QColor(color)


class Panel:
    """One channel group: a plot, its curves and its per-panel items."""

    def __init__(self, group, traces, plot):
        self.group = group
        self.traces = traces              # [(label, color, key, unit)]
        self.plot = plot
        self.unit = traces[0][3] if traces else group.unit
        self.curves = []                  # PlotDataItem per trace
        self.cursors = []                 # the two measurement lines
        self.markers = []                 # InfiniteLines for markers
        self.hover = None


class ScopeView(QtWidgets.QWidget):
    """One tab. `source` is a data.Recording or a data.LiveBuffer; `settings`
    a QSettings for remembered ranges/visibility (None: defaults, for figures)."""

    rangesChanged = QtCore.Signal()

    def __init__(self, source, settings=None, parent=None, figure=False):
        super().__init__(parent)
        self.source = source
        self.live = isinstance(source, data.LiveBuffer)
        self.settings = settings
        self.figure = figure
        self.theme = LIGHT if figure else DARK
        self.follow = self.live
        self.window_s = float(self._get('window_s', 30.0)) if self.live else None
        self.cursors_on = False
        self.cursor_x = None              # [x1, x2]
        self.hover_x = None
        self.panels = []
        self._layout_key = None
        self._guard = False
        self._markers_seen = None
        self._rate = (0, QtCore.QElapsedTimer())
        self._rate[1].start()
        self._rate_text = ''
        self._ticks = 0
        if self.live:
            self.name = 'LIVE %s' % source.address
            self.t0 = None
        else:
            self.name = source.name
            self.t0 = float(source.t[0]) if len(source.t) else 0.0
            self.x_all = source.t - self.t0

        self.info = QtWidgets.QLabel()
        self.info.setTextFormat(Qt.TextFormat.RichText)
        self.info.setContentsMargins(8, 3, 8, 3)
        self.overview = pg.PlotWidget(background=self.theme['bg'])
        self.overview.setFixedHeight(52)
        self.overview.hideAxis('left')
        self.overview.hideAxis('bottom')
        self.overview.setMouseEnabled(x=False, y=False)
        self.overview.hideButtons()
        self.overview.setMenuEnabled(False)
        self.ov_curve = self.overview.plot(pen=pg.mkPen('#8a8a8a', width=1), connect='finite')
        self.ov_curve.setDownsampling(auto=True, method='peak')
        self.region = pg.LinearRegionItem(brush=pg.mkBrush(255, 255, 255, 28),
                                          pen=pg.mkPen('#5fd38a', width=1.5))
        self.region.setZValue(10)
        self.overview.addItem(self.region, ignoreBounds=True)
        self.region.sigRegionChanged.connect(self._region_moved)
        self.glw = pg.GraphicsLayoutWidget()
        self.glw.setBackground(self.theme['bg'])
        self.glw.ci.layout.setSpacing(0)
        self.table = QtWidgets.QTableWidget(0, 7)
        self.table.setHorizontalHeaderLabels(['channel', 'C1', 'C2', 'Δ (C2−C1)',
                                              'mean', 'min', 'max'])
        self.table.verticalHeader().setVisible(False)
        self.table.setEditTriggers(QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.horizontalHeader().setSectionResizeMode(
            0, QtWidgets.QHeaderView.ResizeMode.Stretch)
        self.table.hide()
        self.cursor_label = QtWidgets.QLabel()
        self.cursor_label.hide()
        split = QtWidgets.QSplitter(Qt.Orientation.Vertical)
        split.addWidget(self.glw)
        bottom = QtWidgets.QWidget()
        bl = QtWidgets.QVBoxLayout(bottom)
        bl.setContentsMargins(4, 2, 4, 2)
        bl.addWidget(self.cursor_label)
        bl.addWidget(self.table)
        split.addWidget(bottom)
        split.setStretchFactor(0, 5)
        split.setStretchFactor(1, 1)
        lay = QtWidgets.QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(0)
        if not figure:
            lay.addWidget(self.info)
            lay.addWidget(self.overview)
        lay.addWidget(split)
        self.glw.scene().sigMouseMoved.connect(self._hover)
        self.glw.scene().sigMouseClicked.connect(self._clicked)

        self.rebuild()
        if self.live:
            self.timer = QtCore.QTimer(self)
            self.timer.timeout.connect(self._tick)
            self.timer.start(33)
        else:
            self.show_all()
            self._update_overview()
            self._update_info()

    # ------------------------------------------------------------ settings

    def _get(self, key, default=None):
        if self.settings is None:
            return default
        v = self.settings.value(key)
        return default if v is None else v

    def _set(self, key, value):
        if self.settings is not None and not self.figure:
            self.settings.setValue(key, value)

    def group_range(self, g):
        v = self._get('groups/%s/range' % g.id)
        try:
            lo, hi = float(v[0]), float(v[1])
            if math.isfinite(lo) and math.isfinite(hi) and hi > lo:
                return lo, hi
        except (TypeError, ValueError, IndexError):
            pass
        return g.range

    def group_visible(self, g):
        v = self._get('groups/%s/visible' % g.id)
        if v in ('on', 'off'):
            return v == 'on'
        return not g.hidden

    def set_group_visible(self, gid, on):
        self._set('groups/%s/visible' % gid, 'on' if on else 'off')
        self.rebuild(force=True)

    # ------------------------------------------------------------ data

    def _columns(self):
        """Channel -> values, to decide which groups have data. Live: the last
        few seconds of the buffer."""
        if not self.live:
            return self.source.columns
        span = self.source.span()
        if span is None:
            return {}
        keys = self.source.keys()
        _, cols = self.source.window(keys, span[1] - 5.0, np.inf)
        return cols

    def _data(self, keys, lo=-np.inf, hi=np.inf):
        """(x, {key: values}) between display times lo and hi."""
        if self.live:
            origin = self.source.t_origin or 0.0
            t, cols = self.source.window(keys, lo + origin, hi + origin)
            return t - origin, cols
        i0 = np.searchsorted(self.x_all, lo, 'left')
        i1 = np.searchsorted(self.x_all, hi, 'right')
        cols = {}
        for k in keys:
            a = data.series(self.source.columns, k)
            if a is not None:
                cols[k] = a[i0:i1]
        return self.x_all[i0:i1], cols

    def all_groups(self):
        """[(group, traces)] for every group with data, extras included."""
        cols = self._columns()
        out = []
        for g in list(data.GROUPS) + data.extra_groups(cols):
            traces = data.resolve(g, cols)
            if traces:
                out.append((g, traces))
        return out

    # ------------------------------------------------------------ layout

    def rebuild(self, force=False):
        groups = [(g, tr) for g, tr in self.all_groups() if self.group_visible(g)]
        key = tuple((g.id, tuple(t[2] for t in tr)) for g, tr in groups)
        if key == self._layout_key and not force:
            return
        self._layout_key = key
        xr = self.panels[0].plot.vb.viewRange()[0] if self.panels else None
        self.glw.clear()
        self.panels = []
        if self.figure:
            self.glw.addLabel(self._figure_title(), row=0, col=0, justify='left')
        first = None
        for g, traces in groups:
            vb = TimeViewBox()
            plot = self.glw.addPlot(row=len(self.panels) + (1 if self.figure else 0), col=0,
                                    viewBox=vb)
            panel = Panel(g, traces, plot)
            self._style_plot(plot)
            plot.setMenuEnabled(True)
            plot.hideButtons()
            lo, hi = self.group_range(g)
            vb.setYRange(lo, hi, padding=0)
            for label, color, k, unit in traces:
                c = plot.plot(pen=pg.mkPen(pen_color(color, self.figure), width=1.4 if self.figure
                                           else 1.2), connect='finite')
                c.setDownsampling(auto=True, method='peak')
                c.setClipToView(True)
                panel.curves.append(c)
                if not self.live:
                    a = data.series(self.source.columns, k)
                    c.setData(self.x_all, a)
            panel.hover = pg.InfiniteLine(angle=90, movable=False,
                                          pen=pg.mkPen('#777777', style=Qt.PenStyle.DashLine))
            panel.hover.hide()
            plot.addItem(panel.hover, ignoreBounds=True)
            if first is None:
                first = plot
            else:
                plot.setXLink(first)
            vb.sigRangeChangedManually.connect(self._manual_range)
            vb.sigYRangeChanged.connect(lambda _vb, r, g=g: self._remember_range(g, r))
            self.panels.append(panel)
        if self.panels:
            for p in self.panels[:-1]:
                p.plot.getAxis('bottom').setStyle(showValues=False)
                p.plot.getAxis('bottom').setHeight(4)
            last = self.panels[-1].plot
            last.setLabel('bottom', 'time (s)')
            first.sigXRangeChanged.connect(self._x_changed)
            if xr is not None:
                first.setXRange(*xr, padding=0)
        self._markers_seen = None
        self._draw_markers()
        if self.cursors_on:
            self._make_cursors()
        self._update_titles()

    def _style_plot(self, plot):
        th = self.theme
        plot.showGrid(x=True, y=True, alpha=th['grid'])
        for side in ('left', 'bottom'):
            ax = plot.getAxis(side)
            ax.setPen(pg.mkPen(th['fg']))
            ax.setTextPen(pg.mkPen(th['fg']))
        plot.getAxis('left').setWidth(AXIS_W)

    def _figure_title(self):
        info = self.source.info if not self.live else {}
        src = info.get('source') or {}
        vids = ', '.join('%s: %s' % (c, v.rsplit('/', 1)[-1])
                         for c, v in (src.get('videos') or {}).items())
        frames = info.get('n', 0)
        return ('<span style="font-size:13pt;font-weight:600;color:#111">%s</span>'
                '<span style="font-size:9pt;color:#555">&nbsp;&nbsp; %s%d frames, %.1f s</span>'
                % (self.name, (vids + ' &nbsp;·&nbsp; ') if vids else '', frames,
                   info.get('span', 0.0)))

    @contextlib.contextmanager
    def themed(self, theme):
        """Draw in another theme for the duration (exporting a light figure
        from the dark window), then put everything back."""
        saved, was_figure = self.theme, self.figure
        self.theme = theme
        try:
            self._apply_theme(theme is LIGHT)
            yield
        finally:
            self.theme = saved
            self._apply_theme(was_figure)

    def _apply_theme(self, light):
        self.glw.setBackground(self.theme['bg'])
        for p in self.panels:
            self._style_plot(p.plot)
            for c, (label, color, k, unit) in zip(p.curves, p.traces):
                c.setPen(pg.mkPen(pen_color(color, light), width=1.4 if light else 1.2))
        self._update_titles()

    # ------------------------------------------------------------ time axis

    def x_range(self):
        if not self.panels:
            return 0.0, 1.0
        return tuple(self.panels[0].plot.vb.viewRange()[0])

    def set_x_range(self, lo, hi):
        if self.panels and hi > lo:
            self._guard = True
            try:
                self.panels[0].plot.setXRange(lo, hi, padding=0)
            finally:
                self._guard = False

    def data_span(self):
        if self.live:
            span = self.source.span()
            if span is None:
                return None
            o = self.source.t_origin or 0.0
            return span[0] - o, span[1] - o
        if not len(self.x_all):
            return None
        return float(self.x_all[0]), float(self.x_all[-1])

    def show_all(self):
        span = self.data_span()
        if span:
            self.set_x_range(span[0], max(span[1], span[0] + 1e-3))

    def set_window(self, seconds):
        seconds = float(seconds)
        if self.live:
            self.window_s = seconds
            self._set('window_s', seconds)
            if self.follow:
                self._follow_range()
                return
        lo, hi = self.x_range()
        c = (lo + hi) / 2 if not self.live else hi - seconds / 2
        self.set_x_range(c - seconds / 2, c + seconds / 2)

    def zoom_time(self, factor):
        lo, hi = self.x_range()
        if self.live and self.follow:
            self.window_s = max(0.2, (hi - lo) * factor)
            self._follow_range()
            return
        c, h = (lo + hi) / 2, (hi - lo) / 2 * factor
        self.set_x_range(c - h, c + h)

    def pan(self, fraction):
        lo, hi = self.x_range()
        d = (hi - lo) * fraction
        self.set_follow(False)
        self.set_x_range(lo + d, hi + d)

    def home(self):
        span = self.data_span()
        if span:
            lo, hi = self.x_range()
            self.set_follow(False)
            self.set_x_range(span[0], span[0] + (hi - lo))

    def end(self):
        if self.live:
            self.set_follow(True)
            return
        span = self.data_span()
        if span:
            lo, hi = self.x_range()
            self.set_x_range(span[1] - (hi - lo), span[1])

    def set_follow(self, on):
        if not self.live or on == self.follow:
            return
        self.follow = on
        if on:
            self._follow_range()
        self._update_info()

    def _follow_range(self):
        span = self.data_span()
        if not span:
            return
        first, last = span
        w = self.window_s or 30.0
        # Chart-recorder sweep: fill from the left until a window's worth has
        # arrived, then scroll with the newest sample at the right edge.
        if last - first < w:
            self.set_x_range(first, first + w)
        else:
            self.set_x_range(last - w, last)

    def _manual_range(self, *args):
        if self.live and not self._guard:
            self.set_follow(False)

    def _x_changed(self, *args):
        lo, hi = self.x_range()
        self._guard_region(lo, hi)
        if not self.live:
            self._update_titles()

    def _guard_region(self, lo, hi):
        self.region.blockSignals(True)
        self.region.setRegion((lo, hi))
        self.region.blockSignals(False)

    def _region_moved(self):
        lo, hi = self.region.getRegion()
        self.set_follow(False)
        self.set_x_range(lo, hi)

    def _remember_range(self, g, r):
        if not self.figure:
            self._set('groups/%s/range' % g.id, [float(r[0]), float(r[1])])

    # ------------------------------------------------------------ amplitude

    def autoscale(self, panel):
        lo, hi = self.x_range()
        _, cols = self._data([t[2] for t in panel.traces], lo, hi)
        vals = [a[np.isfinite(a)] for a in cols.values() if a is not None]
        vals = [v for v in vals if v.size]
        if not vals:
            return
        v = np.concatenate(vals)
        a, b = float(v.min()), float(v.max())
        pad = (b - a) * 0.08 or abs(b) * 0.1 or 1.0
        panel.plot.vb.setYRange(a - pad, b + pad, padding=0)

    def autoscale_all(self):
        for p in self.panels:
            self.autoscale(p)

    def reset_ranges(self):
        for p in self.panels:
            if self.settings is not None:
                self.settings.remove('groups/%s/range' % p.group.id)
            p.plot.vb.setYRange(*p.group.range, padding=0)

    # ------------------------------------------------------------ live

    def _tick(self):
        self._ticks += 1
        if self._ticks % 30 == 1:
            self.rebuild()
        if self.follow:
            self._follow_range()
        lo, hi = self.x_range()
        margin = (hi - lo) * 0.1
        keys = [t[2] for p in self.panels for t in p.traces]
        x, cols = self._data(keys, lo - margin, hi + margin)
        for p in self.panels:
            for c, (label, color, k, unit) in zip(p.curves, p.traces):
                a = cols.get(k)
                if a is None:
                    c.setData([], [])
                else:
                    c.setData(x, a)
        if self._ticks % 10 == 0:
            self._update_overview()
            self._draw_markers()
            self._update_info()
        if self.hover_x is None:
            self._update_titles()

    def _update_overview(self):
        if not self.panels:
            self.ov_curve.setData([], [])
            return
        label, color, key, unit = self.panels[0].traces[0]
        for p in self.panels:
            if p.group.id == 'lmr':
                key = p.traces[0][2]
                color = p.traces[0][1]
        x, cols = self._data([key])
        a = cols.get(key)
        if a is None or not len(x):
            return
        step = max(1, len(x) // 4000)
        self.ov_curve.setData(x[::step], a[::step])
        self.ov_curve.setPen(pg.mkPen(color, width=1))
        span = self.data_span()
        if span:
            self.overview.setXRange(span[0], max(span[1], span[0] + 1e-3), padding=0.01)
        lo, hi = self.x_range()
        self._guard_region(lo, hi)

    def _update_info(self):
        th = self.theme
        if self.live:
            st = self.source.status()
            n0, clock = self._rate
            if clock.elapsed() > 1500:
                self._rate_text = '%.0f states/s' % ((st['n'] - n0) * 1000.0 / clock.elapsed())
                self._rate = (st['n'], clock)
                clock.restart()
            bits = ['<b>LIVE</b> %s' % self.source.address, self._rate_text or 'waiting for data…']
            if st['error']:
                bits.append('<span style="color:#ff6060">%s</span>' % st['error'])
            if st['rec']:
                bits.append('<span style="background:#c33;color:white">&nbsp;REC %s&nbsp;</span>'
                            % st['rec'])
            if st['stale']:
                bits.append('<span style="background:#a22;color:white">&nbsp;%s STALE&nbsp;</span>'
                            % st['stale'])
            if not self.follow:
                bits.append('<span style="color:#ffd84a">paused — End or Live returns to now</span>')
        else:
            info = self.source.info
            bits = ['<b>%s</b>' % self.name,
                    '%d rows, %.1f s' % (info.get('n', 0), info.get('span', 0.0))]
            if info.get('cameras'):
                bits.append('cameras %s (primary %s%s)' % (
                    ', '.join(info['cameras']), info.get('primary'),
                    '' if info.get('primary_known', True) else ', guessed'))
            if info.get('code_version'):
                bits.append(info['code_version'])
        self.info.setText('<span style="color:%s">%s</span>'
                          % (th['dim'], ' &nbsp;·&nbsp; '.join(bits)))

    # ------------------------------------------------------------ markers

    def marker_list(self):
        """[(x, kind, label)] in display time."""
        if self.live:
            o = self.source.t_origin or 0.0
            return [(t - o, k, lbl) for t, k, lbl in self.source.status()['markers']]
        out = []
        s = self.source.columns.get('stim')
        if s is not None and len(s) > 1:
            prev = s[:-1]
            cur = s[1:]
            change = np.isfinite(cur) & ~((cur == prev) | (np.isnan(cur) & np.isnan(prev)))
            for i in np.nonzero(change)[0] + 1:
                out.append((float(self.x_all[i]), 'stim', '%g' % s[i]))
        for ev in self.source.events:
            out.append((ev['t'] - self.t0, 'event', str(ev['text'])[:40]))
        return out

    def _draw_markers(self):
        marks = self.marker_list()
        sig = (len(marks), marks[-1] if marks else None, len(self.panels))
        if sig == self._markers_seen:
            return
        self._markers_seen = sig
        for p in self.panels:
            for line in p.markers:
                p.plot.removeItem(line)
            p.markers = []
        for i, p in enumerate(self.panels):
            for x, kind, label in marks:
                color, style = MARKERS.get(kind, ('#888888', Qt.PenStyle.SolidLine))
                opts = {}
                if i == 0:
                    opts = {'label': label, 'labelOpts': {'position': 0.88, 'color': color,
                                                          'movable': False}}
                line = pg.InfiniteLine(pos=x, angle=90, movable=False,
                                       pen=pg.mkPen(color, style=style, width=1), **opts)
                line.setOpacity(0.75)
                p.plot.addItem(line, ignoreBounds=True)
                p.markers.append(line)

    # ------------------------------------------------------------ cursors

    def set_cursors(self, on):
        self.cursors_on = bool(on)
        if on and self.cursor_x is None:
            lo, hi = self.x_range()
            self.cursor_x = [lo + (hi - lo) / 3, lo + 2 * (hi - lo) / 3]
        self._make_cursors()
        self.table.setVisible(self.cursors_on)
        self.cursor_label.setVisible(self.cursors_on)
        self.update_readout()

    def _make_cursors(self):
        for p in self.panels:
            for line in p.cursors:
                p.plot.removeItem(line)
            p.cursors = []
            if not self.cursors_on:
                continue
            for i, x in enumerate(self.cursor_x):
                opts = {}
                if p is self.panels[0]:
                    opts = {'label': 'C%d' % (i + 1),
                            'labelOpts': {'position': 0.97, 'color': CURSOR_COLORS[i]}}
                line = pg.InfiniteLine(pos=x, angle=90, movable=True,
                                       pen=pg.mkPen(CURSOR_COLORS[i], width=1.5),
                                       hoverPen=pg.mkPen(CURSOR_COLORS[i], width=3), **opts)
                line.sigPositionChanged.connect(lambda ln, i=i: self._cursor_moved(i, ln))
                p.plot.addItem(line, ignoreBounds=True)
                p.cursors.append(line)

    def set_cursor(self, i, x):
        """Place cursor i (0 or 1) at display time x."""
        if not self.cursors_on:
            self.set_cursors(True)
        for p in self.panels:
            if len(p.cursors) > i:
                p.cursors[i].setValue(x)
        self.cursor_x[i] = float(x)
        self.update_readout()

    def _cursor_moved(self, i, line):
        if self._guard:
            return
        x = float(line.value())
        self.cursor_x[i] = x
        self._guard = True
        try:
            for p in self.panels:
                if len(p.cursors) > i and p.cursors[i] is not line:
                    p.cursors[i].setValue(x)
        finally:
            self._guard = False
        self.update_readout()

    def value_at(self, key, x):
        """The sample nearest display time x (within a frame or two), or NaN."""
        xs, cols = self._data([key], x - 0.1, x + 0.1)
        a = cols.get(key)
        if a is None or not len(xs):
            return math.nan
        i = int(np.argmin(np.abs(xs - x)))
        return float(a[i])

    def readout(self):
        """Rows of the cursor table: (name, unit, v1, v2, delta, mean, min, max)."""
        if not self.cursors_on or not self.cursor_x:
            return []
        x1, x2 = self.cursor_x
        lo, hi = min(x1, x2), max(x1, x2)
        keys = [t[2] for p in self.panels for t in p.traces]
        _, between = self._data(keys, lo, hi)
        rows = []
        for p in self.panels:
            for label, color, k, unit in p.traces:
                v1, v2 = self.value_at(k, x1), self.value_at(k, x2)
                seg = between.get(k)
                seg = seg[np.isfinite(seg)] if seg is not None else np.zeros(0)
                stats = (float(seg.mean()), float(seg.min()), float(seg.max())) if seg.size \
                    else (math.nan,) * 3
                name = p.group.title if len(p.traces) == 1 else '%s · %s' % (p.group.title, label)
                rows.append((name, unit, v1, v2, v2 - v1) + stats)
        return rows

    def update_readout(self):
        if not self.cursors_on:
            return
        x1, x2 = self.cursor_x
        self.cursor_label.setText('C1 %.3f s &nbsp; C2 %.3f s &nbsp; <b>Δt %.3f s</b>'
                                  % (x1, x2, x2 - x1))
        rows = self.readout()
        self.table.setRowCount(len(rows))
        for r, row in enumerate(rows):
            name, unit = row[0], row[1]
            cells = ['%s (%s)' % (name, unit) if unit else name] + [nice(v) for v in row[2:]]
            for c, text in enumerate(cells):
                item = QtWidgets.QTableWidgetItem(text)
                if c:
                    item.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
                self.table.setItem(r, c, item)

    # ------------------------------------------------------------ hover

    def _hover(self, pos):
        x = None
        for p in self.panels:
            vb = p.plot.vb
            if vb.sceneBoundingRect().contains(pos):
                x = float(vb.mapSceneToView(pos).x())
                break
        self.hover_x = x
        for p in self.panels:
            if x is None:
                p.hover.hide()
            else:
                p.hover.setPos(x)
                p.hover.show()
        self._update_titles()

    def _clicked(self, ev):
        if not ev.double():
            return
        for p in self.panels:
            if p.plot.vb.sceneBoundingRect().contains(ev.scenePos()):
                self.autoscale(p)
                ev.accept()
                return

    def _update_titles(self):
        """Each panel's title row: its name, the value of each trace (under the
        mouse, else at the right edge of the view), and the unit."""
        if not self.panels:
            return
        th = self.theme
        x = self.hover_x
        if x is None:
            lo, hi = self.x_range()
            span = self.data_span()
            x = min(hi, span[1]) if span else hi
        for p in self.panels:
            bits = ['<span style="color:%s;font-weight:600">%s</span>' % (th['title'], p.group.title)]
            for label, color, k, unit in p.traces:
                c = pen_color(color, th is LIGHT).name()
                if self.figure:         # a whole recording: name the traces, no readout
                    bits.append('<span style="color:%s">%s</span>' % (c, label))
                    continue
                v = self.value_at(k, x)
                bits.append('<span style="color:%s">%s %s</span>' % (c, label, nice(v)))
            if p.unit:
                bits.append('<span style="color:%s">%s</span>' % (th['dim'], p.unit))
            p.plot.setTitle(' &nbsp; '.join(bits), justify='left', size='9pt')

    # ------------------------------------------------------------ export

    def range_rows(self):
        """(header, x, columns) for the cursor range, else the visible range."""
        if self.cursors_on and self.cursor_x:
            lo, hi = sorted(self.cursor_x)
        else:
            lo, hi = self.x_range()
        keys = [t[2] for p in self.panels for t in p.traces]
        x, cols = self._data(keys, lo, hi)
        header = ['time_s'] + ['%s (%s)' % (k, u) if u else k
                               for p in self.panels for _, _, k, u in p.traces]
        return header, x, [cols.get(k, np.full(len(x), np.nan)) for k in keys]

    def info_text(self):
        if self.live:
            st = self.source.status()
            return 'Live stream %s\n%d states received\nrecording: %s' % (
                self.source.address, st['n'], st['rec'] or 'no')
        i = self.source.info
        lines = ['%s' % i.get('path'), '',
                 'rows: %d   span: %.2f s' % (i.get('n', 0), i.get('span', 0.0)),
                 'cameras: %s   primary: %s%s' % (', '.join(i.get('cameras') or []) or '-',
                                                 i.get('primary'),
                                                 '' if i.get('primary_known', True)
                                                 else ' (guessed: the file predates the rig stamp)'),
                 'code version: %s' % (i.get('code_version') or 'not stamped')]
        conv = i.get('conventions')
        if conv:
            lines.append('conventions: %s' % ', '.join('%s=%s' % kv for kv in sorted(conv.items())))
        if i.get('lever_px'):
            lines.append('lever_px: %s' % i['lever_px'])
        src = i.get('source') or {}
        for cam, v in (src.get('videos') or {}).items():
            lines.append('video %s: %s  (%s frames, %s fps)' % (
                cam, v, (src.get('n_frames') or {}).get(cam, '?'), (src.get('fps') or {}).get(cam, '?')))
        if i.get('time_column'):
            lines.append('time column: %s' % i['time_column'])
        return '\n'.join(lines)

    def close_source(self):
        if self.live:
            self.timer.stop()
            self.source.close()
