"""The Scope's main window: one tab per recording or live stream, the menus
and toolbar, drag-and-drop, and the single-instance server through which other
programs (analyze_videos after each analysis, `scope file.h5` from a terminal)
open files in the window that is already there instead of a new one."""
import getpass
import json
import os

import numpy as np
from PySide6 import QtCore, QtGui, QtNetwork, QtWidgets

from . import data
from .export import FORMATS, export_view
from .launch import DEFAULT_LIVE
from .view import SPANS, ScopeView

Qt = QtCore.Qt
SERVER = os.environ.get('KINEFLY2_SCOPE_SERVER') or 'kinefly2-scope-%s' % getpass.getuser()
MAX_RECENT = 10
WELCOME = """<div style="color:#9a9a9a;font-size:13pt;line-height:160%">
<p style="font-size:18pt;color:#dddddd">Kinefly2 Scope</p>
<p><b>File ▸ Open</b> (Ctrl+O) a recording — an <code>_flystate.h5</code> from a live
recording or an offline analysis, or one of the CSVs — or drop files here.</p>
<p><b>File ▸ Monitor live</b> (Ctrl+L) to watch run_rig as it runs.</p>
<p style="font-size:11pt">Wheel/drag in a panel: time, all panels together &nbsp;·&nbsp;
Shift+wheel or wheel on the y axis: amplitude &nbsp;·&nbsp; double-click: fit &nbsp;·&nbsp;
A: fit all &nbsp;·&nbsp; C: cursors &nbsp;·&nbsp; Space: pause/live</p></div>"""
SHORTCUTS = """Mouse
  wheel / drag in a panel      zoom / pan time (all panels together)
  Shift + wheel / drag         zoom / pan that panel's amplitude
  wheel / drag on a y axis     zoom / pan that panel's amplitude
  double-click a panel         fit its amplitude to what is visible
  drag the overview box        jump anywhere; drag its edges to zoom
  right-click                  view all, export, mouse mode

Keys
  A          fit every panel's amplitude
  C          two measurement cursors (drag them; the table shows values, Δ, mean/min/max)
  Space      live: pause / back to now
  ← →        pan half a window          + −   zoom time
  Home End   start / end (live: back to now)
  Ctrl+O     open      Ctrl+L  monitor live     Ctrl+E  export figure
  Ctrl+W     close tab Ctrl+I  file info"""


def send_to_running(paths, live=None, name=SERVER, timeout_ms=400):
    """Hand files (and/or a live address) to a Scope that is already open.
    True if one took them."""
    sock = QtNetwork.QLocalSocket()
    sock.connectToServer(name)
    if not sock.waitForConnected(timeout_ms):
        return False
    sock.write(json.dumps({'open': list(paths), 'live': live}).encode() + b'\n')
    sock.flush()
    sock.waitForBytesWritten(1000)
    sock.disconnectFromServer()
    return True


class InstanceServer(QtCore.QObject):
    """Listens for send_to_running() and opens what it is sent."""

    def __init__(self, window, name=SERVER):
        super().__init__(window)
        self.window = window
        self.buffers = {}
        QtNetwork.QLocalServer.removeServer(name)     # a socket left by a crashed Scope
        self.server = QtNetwork.QLocalServer(self)
        self.server.newConnection.connect(self._accept)
        self.listening = self.server.listen(name)

    def _accept(self):
        while self.server.hasPendingConnections():
            s = self.server.nextPendingConnection()
            self.buffers[s] = b''
            s.readyRead.connect(lambda s=s: self._read(s))
            s.disconnected.connect(lambda s=s: self._done(s))
            if s.bytesAvailable():
                self._read(s)

    def _read(self, s):
        self.buffers[s] = self.buffers.get(s, b'') + bytes(s.readAll())
        while b'\n' in self.buffers[s]:
            line, self.buffers[s] = self.buffers[s].split(b'\n', 1)
            try:
                msg = json.loads(line.decode())
            except ValueError:
                continue
            self.window.open_paths(msg.get('open') or [])
            if msg.get('live'):
                self.window.open_live(msg['live'])
            self.window.bring_to_front()

    def _done(self, s):
        self.buffers.pop(s, None)
        s.deleteLater()


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self, settings=None):
        super().__init__()
        self.settings = settings or QtCore.QSettings('Kinefly2', 'Scope')
        self.setWindowTitle('Kinefly2 Scope')
        self.setAcceptDrops(True)
        self.tabs = QtWidgets.QTabWidget()
        self.tabs.setTabsClosable(True)
        self.tabs.setMovable(True)
        self.tabs.setDocumentMode(True)
        self.tabs.tabCloseRequested.connect(self.close_tab)
        self.tabs.currentChanged.connect(self._tab_changed)
        welcome = QtWidgets.QLabel(WELCOME)
        welcome.setAlignment(Qt.AlignmentFlag.AlignCenter)
        welcome.setWordWrap(True)
        self.stack = QtWidgets.QStackedWidget()
        self.stack.addWidget(welcome)
        self.stack.addWidget(self.tabs)
        self.setCentralWidget(self.stack)
        self._build_actions()
        self._build_menus()
        self._build_toolbar()
        self.statusBar().showMessage('Open a recording, or monitor a live rig.')
        geo = self.settings.value('window/geometry')
        if geo is not None:
            self.restoreGeometry(geo)
        else:
            self.resize(1280, 900)

    # ------------------------------------------------------------ actions

    def _act(self, text, slot, shortcut=None, checkable=False, tip=None):
        a = QtGui.QAction(text, self)
        if shortcut:
            a.setShortcut(QtGui.QKeySequence(shortcut))
            a.setShortcutContext(Qt.ShortcutContext.WindowShortcut)
        a.setCheckable(checkable)
        if tip:
            a.setToolTip(tip)
            a.setStatusTip(tip)
        a.triggered.connect(slot)
        self.addAction(a)            # keyboard shortcuts work without the menus open
        return a

    def _build_actions(self):
        v = self.view
        self.a_open = self._act('&Open…', self.open_dialog, 'Ctrl+O', tip='Open .h5 / .csv recordings')
        self.a_live = self._act('Monitor &live…', self.live_dialog, 'Ctrl+L',
                                tip="Watch run_rig's fly-state stream")
        self.a_close = self._act('&Close tab', lambda: self.close_tab(self.tabs.currentIndex()),
                                 'Ctrl+W')
        self.a_export = self._act('&Export figure…', self.export_dialog, 'Ctrl+E',
                                  tip='Save the panels as PNG, SVG or PDF (light background)')
        self.a_csv = self._act('Export cursor range as &CSV…', self.export_csv)
        self.a_info = self._act('Recording &info', self.show_info, 'Ctrl+I')
        self.a_quit = self._act('&Quit', self.close, 'Ctrl+Q')
        self.a_fit = self._act('&Fit all amplitudes', lambda: v() and v().autoscale_all(), 'A',
                               tip='Fit every panel to what is visible (A)')
        self.a_cursors = self._act('&Cursors', self._toggle_cursors, 'C', checkable=True,
                                   tip='Two measurement cursors (C)')
        self.a_follow = self._act('&Live / pause', self._toggle_follow, 'Space', checkable=True,
                                  tip='Live: follow the newest data, or pause (Space)')
        self.a_zin = self._act('Zoom time in', lambda: v() and v().zoom_time(1 / 1.5), '+')
        self.a_zin2 = self._act('Zoom time in', lambda: v() and v().zoom_time(1 / 1.5), '=')
        self.a_zout = self._act('Zoom time out', lambda: v() and v().zoom_time(1.5), '-')
        self.a_left = self._act('Pan left', lambda: v() and v().pan(-0.5), 'Left')
        self.a_right = self._act('Pan right', lambda: v() and v().pan(0.5), 'Right')
        self.a_home = self._act('Start', lambda: v() and v().home(), 'Home')
        self.a_end = self._act('End / now', lambda: v() and v().end(), 'End')
        self.a_all = self._act('Show &whole recording', lambda: v() and v().show_all(), 'Ctrl+0')
        self.a_reset = self._act('&Reset amplitude ranges', lambda: v() and v().reset_ranges())
        self.a_keys = self._act('&Mouse and keys', self.show_shortcuts, 'F1')

    def _build_menus(self):
        m = self.menuBar().addMenu('&File')
        m.addAction(self.a_open)
        self.recent_menu = m.addMenu('Open &recent')
        self.recent_menu.aboutToShow.connect(self._fill_recent)
        m.addAction(self.a_live)
        m.addSeparator()
        for a in (self.a_export, self.a_csv, self.a_info):
            m.addAction(a)
        m.addSeparator()
        m.addAction(self.a_close)
        m.addAction(self.a_quit)
        m = self.menuBar().addMenu('&View')
        for a in (self.a_fit, self.a_cursors, self.a_follow, self.a_all, self.a_reset):
            m.addAction(a)
        self.channels_menu = m.addMenu('C&hannels')
        self.channels_menu.aboutToShow.connect(self._fill_channels)
        m = self.menuBar().addMenu('&Help')
        m.addAction(self.a_keys)

    def _build_toolbar(self):
        tb = self.addToolBar('main')
        tb.setObjectName('main')
        tb.setMovable(False)
        tb.addAction(self.a_open)
        tb.addAction(self.a_live)
        tb.addSeparator()
        tb.addWidget(QtWidgets.QLabel(' window '))
        self.span_box = QtWidgets.QComboBox()
        self.span_box.addItem('all', None)
        for s in SPANS:
            self.span_box.addItem('%d s' % s if s < 60 else '%d min' % (s // 60), s)
        self.span_box.setToolTip('How much time the panels show')
        self.span_box.activated.connect(self._span_chosen)
        tb.addWidget(self.span_box)
        tb.addSeparator()
        for a in (self.a_follow, self.a_fit, self.a_cursors):
            tb.addAction(a)
        tb.addSeparator()
        tb.addAction(self.a_export)

    # ------------------------------------------------------------ tabs

    def view(self):
        w = self.tabs.currentWidget()
        return w if isinstance(w, ScopeView) else None

    def views(self):
        return [self.tabs.widget(i) for i in range(self.tabs.count())]

    def _add(self, view, title, tip):
        i = self.tabs.addTab(view, title)
        self.tabs.setTabToolTip(i, tip)
        self.tabs.setCurrentIndex(i)
        self.stack.setCurrentIndex(1)
        return view

    def open_paths(self, paths):
        opened = []
        for p in paths:
            p = os.path.abspath(str(p))
            same = [i for i in range(self.tabs.count())
                    if getattr(self.tabs.widget(i).source, 'path', None) == p]
            if same:
                self.tabs.setCurrentIndex(same[0])
                continue
            try:
                rec = data.load_recording(p)
            except Exception as e:          # a bad file must not take the window down
                QtWidgets.QMessageBox.warning(self, 'Cannot open', '%s\n\n%s' % (p, e))
                continue
            opened.append(self._add(ScopeView(rec, self.settings), rec.name, p))
            self._remember(p)
        return opened

    def open_live(self, address=DEFAULT_LIVE):
        for i in range(self.tabs.count()):
            v = self.tabs.widget(i)
            if v.live and v.source.address == address:
                self.tabs.setCurrentIndex(i)
                return v
        self.settings.setValue('live/address', address)
        view = ScopeView(data.LiveBuffer(address), self.settings)
        return self._add(view, 'LIVE %s' % address.replace('tcp://', ''), address)

    def close_tab(self, index):
        if index < 0:
            return
        w = self.tabs.widget(index)
        self.tabs.removeTab(index)
        if isinstance(w, ScopeView):
            w.close_source()
            w.deleteLater()
        if not self.tabs.count():
            self.stack.setCurrentIndex(0)

    def _tab_changed(self, index):
        v = self.view()
        self.a_follow.setEnabled(bool(v and v.live))
        self.a_follow.setChecked(bool(v and v.live and v.follow))
        self.a_cursors.setChecked(bool(v and v.cursors_on))
        if v is None:
            self.statusBar().showMessage('Open a recording, or monitor a live rig.')
            return
        # The window box shows this tab's span: live tabs sweep a fixed window.
        i = self.span_box.findData(int(v.window_s)) if v.live and v.window_s else 0
        self.span_box.setCurrentIndex(max(0, i))
        self.setWindowTitle('Kinefly2 Scope — %s' % v.name)
        self.statusBar().showMessage(v.source.path if not v.live else v.source.address)

    def bring_to_front(self):
        self.showNormal()
        self.raise_()
        self.activateWindow()

    # ------------------------------------------------------------ commands

    def _toggle_cursors(self):
        v = self.view()
        if v:
            v.set_cursors(not v.cursors_on)
            self.a_cursors.setChecked(v.cursors_on)

    def _toggle_follow(self):
        v = self.view()
        if v and v.live:
            v.set_follow(not v.follow)
            self.a_follow.setChecked(v.follow)

    def _span_chosen(self, index):
        v = self.view()
        if not v:
            return
        s = self.span_box.itemData(index)
        if s is None:
            v.set_follow(False)
            v.show_all()
        else:
            v.set_window(s)

    def _fill_channels(self):
        self.channels_menu.clear()
        v = self.view()
        if not v:
            self.channels_menu.addAction('(no tab)').setEnabled(False)
            return
        for g, traces in v.all_groups():
            a = self.channels_menu.addAction(g.title)
            a.setCheckable(True)
            a.setChecked(v.group_visible(g))
            a.toggled.connect(lambda on, gid=g.id: v.set_group_visible(gid, on))

    def _remember(self, path):
        recent = [p for p in (self.settings.value('recent') or []) if p != path]
        self.settings.setValue('recent', ([path] + recent)[:MAX_RECENT])

    def _fill_recent(self):
        self.recent_menu.clear()
        recent = [p for p in (self.settings.value('recent') or []) if os.path.exists(p)]
        if not recent:
            self.recent_menu.addAction('(none yet)').setEnabled(False)
        for p in recent:
            self.recent_menu.addAction(p, lambda p=p: self.open_paths([p]))

    def open_dialog(self):
        start = self.settings.value('open/dir') or os.path.expanduser('~')
        paths, _ = QtWidgets.QFileDialog.getOpenFileNames(
            self, 'Open recordings', start,
            'Kinefly2 data (*.h5 *.hdf5 *.csv);;HDF5 (*.h5 *.hdf5);;CSV (*.csv);;All files (*)')
        if paths:
            self.settings.setValue('open/dir', os.path.dirname(paths[0]))
            self.open_paths(paths)

    def live_dialog(self):
        addr, ok = QtWidgets.QInputDialog.getText(
            self, 'Monitor live', "run_rig's fly-state stream (zmq_port in rig.yaml):",
            text=self.settings.value('live/address') or DEFAULT_LIVE)
        if ok and addr.strip():
            self.open_live(addr.strip())

    def export_dialog(self):
        v = self.view()
        if not v:
            return
        start = os.path.join(self.settings.value('export/dir') or os.path.expanduser('~'),
                             v.name.replace(' ', '_').replace(':', '_') + '.png')
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, 'Export figure', start, 'PNG (*.png);;SVG (*.svg);;PDF (*.pdf)')
        if not path:
            return
        if os.path.splitext(path)[1].lower() not in FORMATS:
            path += '.png'
        self.settings.setValue('export/dir', os.path.dirname(path))
        try:
            export_view(v, path)
            self.statusBar().showMessage('saved %s' % path, 8000)
        except Exception as e:
            QtWidgets.QMessageBox.warning(self, 'Export failed', str(e))

    def export_csv(self):
        v = self.view()
        if not v:
            return
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, 'Export cursor range (or the visible range) as CSV',
            os.path.join(self.settings.value('export/dir') or os.path.expanduser('~'),
                         v.name.replace(' ', '_').replace(':', '_') + '_range.csv'), 'CSV (*.csv)')
        if not path:
            return
        header, x, cols = v.range_rows()
        with open(path, 'w') as f:
            f.write(','.join(header) + '\n')
            for i in range(len(x)):
                f.write(','.join(['%.6f' % x[i]] + ['' if not np.isfinite(c[i]) else '%.6g' % c[i]
                                                      for c in cols]) + '\n')
        self.statusBar().showMessage('saved %d rows to %s' % (len(x), path), 8000)

    def show_info(self):
        v = self.view()
        if v:
            QtWidgets.QMessageBox.information(self, v.name, v.info_text())

    def show_shortcuts(self):
        box = QtWidgets.QMessageBox(self)
        box.setWindowTitle('Mouse and keys')
        box.setText('<pre>%s</pre>' % SHORTCUTS)
        box.exec()

    # ------------------------------------------------------------ drag and drop

    def dragEnterEvent(self, ev):
        if ev.mimeData().hasUrls():
            ev.acceptProposedAction()

    def dropEvent(self, ev):
        paths = [u.toLocalFile() for u in ev.mimeData().urls() if u.isLocalFile()]
        self.open_paths([p for p in paths if os.path.splitext(p)[1].lower() in
                         ('.h5', '.hdf5', '.csv')])

    def closeEvent(self, ev):
        self.settings.setValue('window/geometry', self.saveGeometry())
        for w in self.views():
            if isinstance(w, ScopeView):
                w.close_source()
        super().closeEvent(ev)
