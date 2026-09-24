"""The launcher window of analyze_videos: which videos, where the results go,
what they are called.

It runs in its own short-lived process, which analyze_videos starts as

    python -m kinefly2.apps.analyze_launcher --rig RIG --out-json FILE

because Tk and OpenCV's windows both want the macOS main event loop, and in
separate processes they never meet. (For the same reason this module imports
nothing that pulls in OpenCV.) The choices come back as JSON in FILE; exit
status 1 means the window was cancelled.

LauncherModel holds the job list and the rules (files grouped into jobs, the
folder each job will be saved as) with no Tk in it, so it is tested directly.
"""
import argparse
import json
import os
import sys

import yaml

from ..core.naming import VIDEO_EXTENSIONS, group_videos, preview_name, safe_name

STATE_PATH = os.path.join(os.path.expanduser('~'), '.kinefly2', 'analyze.json')
SPEEDS = (0.25, 0.5, 1.0, 2.0)


def rig_cameras(rig_path):
    """Camera names in rig.yaml order (enabled or not: a camera switched off
    on the rig can still have videos), and the primary camera."""
    with open(rig_path) as f:
        rig = yaml.safe_load(f) or {}
    cams = [str(c) for c in (rig.get('cameras') or {})]
    primary = rig.get('primary')
    return cams, (primary if primary in cams else (cams[0] if cams else None))


def load_state(path=STATE_PATH):
    try:
        with open(path) as f:
            return json.load(f) or {}
    except (OSError, ValueError):
        return {}


def save_state(state, path=STATE_PATH):
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w') as f:
            json.dump(state, f, indent=1)
    except OSError:
        pass                    # Remembering folders is a convenience only.


def has_configs(folder):
    """A previous analysis to start from: <run>_<cam>_config.yaml files, or
    configs/camera_<cam>.yaml in one made before the folders were flattened."""
    if not folder or not os.path.isdir(folder):
        return False
    if any(f.endswith('_config.yaml') for f in os.listdir(folder)):
        return True
    d = os.path.join(folder, 'configs')
    return os.path.isdir(d) and any(f.startswith('camera_') and f.endswith('.yaml')
                                    for f in os.listdir(d))


class LauncherModel:
    def __init__(self, cameras, primary, state=None):
        st = state or {}
        self.cameras = list(cameras)
        self.primary = primary
        self.jobs = []                          # [{'name', 'videos': {cam: path}}]
        self.out_dir = st.get('out_dir') or ''
        self.video_dir = st.get('video_dir') or os.path.expanduser('~')
        self.seed_kind = st.get('seed_kind') if st.get('seed_kind') in ('rig', 'analysis') else 'rig'
        self.seed_path = st.get('seed_path') or ''
        self.csv = bool(st.get('csv', True))
        self.speed = min(SPEEDS, key=lambda s: abs(s - float(st.get('speed', 0.5))))

    def add_videos(self, paths):
        """Group the files into jobs and append them; returns how many."""
        paths = [str(p) for p in paths if p]
        if not paths:
            return 0
        self.video_dir = os.path.dirname(os.path.abspath(paths[0]))
        new = group_videos(paths, self.cameras, self.primary)
        self.jobs.extend(new)
        return len(new)

    def remove(self, i):
        del self.jobs[i]

    def rename(self, i, name):
        self.jobs[i]['name'] = safe_name(name, fallback=self.jobs[i]['name'])

    def swap(self, i):
        """A pair: exchange its two videos between the cameras. A single
        video: move it to the next camera."""
        videos = self.jobs[i]['videos']
        if len(videos) == 2:
            a, b = list(videos)
            self.jobs[i]['videos'] = {a: videos[b], b: videos[a]}
        elif len(videos) == 1 and self.cameras:
            (cam, path), = videos.items()
            k = self.cameras.index(cam) if cam in self.cameras else -1
            self.jobs[i]['videos'] = {self.cameras[(k + 1) % len(self.cameras)]: path}

    def previews(self):
        """The folder name each job will get -- its name, or name_001, ...
        when the save folder (or an earlier job in the list) has it."""
        taken, out = [], []
        for job in self.jobs:
            name = preview_name(self.out_dir, job['name'], taken)
            taken.append(name)
            out.append(name)
        return out

    def problems(self):
        p = []
        if not self.jobs:
            p.append('add at least one video')
        if not self.out_dir:
            p.append('choose a folder to save results to')
        for job in self.jobs:
            for path in job['videos'].values():
                if not os.path.isfile(path):
                    p.append('not found: %s' % path)
        if self.seed_kind == 'analysis' and not has_configs(self.seed_path):
            p.append('the previous analysis folder has no *_config.yaml to start from')
        return p

    def settings(self, rig_path):
        return {'rig': os.path.abspath(rig_path),
                'jobs': [{'name': j['name'], 'videos': dict(j['videos'])} for j in self.jobs],
                'out_dir': os.path.abspath(os.path.expanduser(self.out_dir)),
                'seed': {'kind': self.seed_kind,
                         'path': os.path.abspath(self.seed_path) if self.seed_kind == 'analysis' else None},
                'csv': self.csv, 'speed': self.speed}

    def state(self):
        return {'out_dir': self.out_dir, 'video_dir': self.video_dir,
                'seed_kind': self.seed_kind, 'seed_path': self.seed_path,
                'csv': self.csv, 'speed': self.speed}


class LauncherWindow:
    def __init__(self, model, rig_path):
        import tkinter as tk
        from tkinter import filedialog, ttk
        self.tk, self.ttk, self.filedialog = tk, ttk, filedialog
        self.model = model
        self.rig_path = rig_path
        self.result = None
        self.root = tk.Tk()
        self.root.title('Kinefly2 - analyze recorded videos')
        self.root.minsize(820, 480)
        self._build()
        self.refresh()
        # On macOS a window started from a terminal opens behind it.
        self.root.lift()
        self.root.attributes('-topmost', True)
        self.root.after(600, lambda: self.root.attributes('-topmost', False))
        self.root.focus_force()

    def _build(self):
        tk, ttk, root, m = self.tk, self.ttk, self.root, self.model
        pad = {'padx': 10, 'pady': 5}

        box = ttk.LabelFrame(root, text=' Videos to analyze -- one row is one analysis; '
                                        'the bottom+front videos of one recording share a row ')
        box.pack(fill='both', expand=True, **pad)
        cols = ['name'] + m.cameras + ['saves as']
        self.tree = ttk.Treeview(box, columns=cols, show='headings', selectmode='browse', height=9)
        for c in cols:
            self.tree.heading(c, text=c)
            self.tree.column(c, width=230 if c == 'name' else 190, stretch=True, anchor='w')
        sb = ttk.Scrollbar(box, orient='vertical', command=self.tree.yview)
        self.tree.configure(yscrollcommand=sb.set)
        self.tree.grid(row=0, column=0, sticky='nsew', padx=(6, 0), pady=6)
        sb.grid(row=0, column=1, sticky='ns', pady=6)
        box.rowconfigure(0, weight=1)
        box.columnconfigure(0, weight=1)
        row = ttk.Frame(box)
        row.grid(row=1, column=0, columnspan=2, sticky='ew', padx=6, pady=(0, 6))
        ttk.Button(row, text='Add videos...', command=self.add).pack(side='left')
        ttk.Button(row, text='Remove', command=self.remove).pack(side='left', padx=4)
        ttk.Button(row, text='Swap cameras', command=self.swap).pack(side='left')
        ttk.Label(row, text='    Name of selected:').pack(side='left')
        self.name_var = tk.StringVar()
        self.name_entry = ttk.Entry(row, textvariable=self.name_var, width=34)
        self.name_entry.pack(side='left', padx=4)
        self.name_entry.bind('<Return>', self.apply_name)
        self.name_entry.bind('<FocusOut>', self.apply_name)
        self.tree.bind('<<TreeviewSelect>>', self.on_select)
        self.tree.bind('<Double-1>', lambda e: self.name_entry.focus_set())

        out = ttk.LabelFrame(root, text=' Save results to (each analysis gets its own folder; '
                                        'an existing name gets _001, _002, ...) ')
        out.pack(fill='x', **pad)
        self.out_var = tk.StringVar(value=m.out_dir)
        ttk.Entry(out, textvariable=self.out_var).pack(side='left', fill='x', expand=True,
                                                       padx=6, pady=6)
        ttk.Button(out, text='Browse...', command=self.browse_out).pack(side='left', padx=6)
        self.out_var.trace_add('write', lambda *a: self.on_out())

        seed = ttk.LabelFrame(root, text=' Start tuning masks and landmarks from ')
        seed.pack(fill='x', **pad)
        self.seed_var = tk.StringVar(value=m.seed_kind)
        ttk.Radiobutton(seed, text='the rig config  (%s)' % os.path.dirname(os.path.abspath(self.rig_path)),
                        variable=self.seed_var, value='rig',
                        command=self.on_seed).grid(row=0, column=0, columnspan=3, sticky='w', padx=6)
        ttk.Radiobutton(seed, text='a previous analysis:', variable=self.seed_var, value='analysis',
                        command=self.on_seed).grid(row=1, column=0, sticky='w', padx=6)
        self.seed_path_var = tk.StringVar(value=m.seed_path)
        ttk.Entry(seed, textvariable=self.seed_path_var).grid(row=1, column=1, sticky='ew', padx=4)
        ttk.Button(seed, text='Browse...', command=self.browse_seed).grid(row=1, column=2, padx=6)
        ttk.Label(seed, text='Each later video in the list starts from the tuning of the one '
                             'before. The rig config itself is never changed.',
                  foreground='#555').grid(row=2, column=0, columnspan=3, sticky='w', padx=6, pady=(0, 6))
        seed.columnconfigure(1, weight=1)
        self.seed_path_var.trace_add('write', lambda *a: self.on_seed())

        opt = ttk.Frame(root)
        opt.pack(fill='x', **pad)
        self.csv_var = tk.BooleanVar(value=m.csv)
        ttk.Checkbutton(opt, text='Also write a CSV per camera (angles in radians)',
                        variable=self.csv_var).pack(side='left')
        ttk.Label(opt, text='      Setup playback speed:').pack(side='left')
        self.speed_var = tk.StringVar(value='%gx' % m.speed)
        ttk.Combobox(opt, textvariable=self.speed_var, values=['%gx' % s for s in SPEEDS],
                     width=6, state='readonly').pack(side='left', padx=4)

        bottom = ttk.Frame(root)
        bottom.pack(fill='x', **pad)
        self.status = ttk.Label(bottom, text='', foreground='#b00000')
        self.status.pack(side='left', fill='x', expand=True)
        ttk.Button(bottom, text='Cancel', command=self.cancel).pack(side='right')
        self.start_btn = ttk.Button(bottom, text='Start', command=self.start)
        self.start_btn.pack(side='right', padx=6)
        root.protocol('WM_DELETE_WINDOW', self.cancel)

    # ------------------------------------------------------------ list

    def selected(self):
        sel = self.tree.selection()
        return self.tree.index(sel[0]) if sel else None

    def refresh(self, select=None):
        m = self.model
        keep = self.selected() if select is None else select
        self.tree.delete(*self.tree.get_children())
        names = m.previews() if m.out_dir else [None] * len(m.jobs)
        for job, saved in zip(m.jobs, names):
            if saved is None:
                where = '(choose a save folder)'
            elif saved != job['name']:
                where = '%s   (name taken)' % saved
            else:
                where = saved
            vals = [job['name']] + [os.path.basename(job['videos'].get(c, '')) or '-'
                                    for c in m.cameras] + [where]
            self.tree.insert('', 'end', values=vals)
        items = self.tree.get_children()
        if items and keep is not None:
            item = items[min(keep, len(items) - 1)]
            self.tree.selection_set(item)
            self.tree.see(item)
        problems = m.problems()
        self.status.configure(text=problems[0] if problems else
                              '%d %s ready' % (len(m.jobs), 'analysis' if len(m.jobs) == 1
                                               else 'analyses'))
        self.start_btn.state(['disabled'] if problems else ['!disabled'])

    def on_select(self, event=None):
        i = self.selected()
        if i is not None:
            self.name_var.set(self.model.jobs[i]['name'])

    def apply_name(self, event=None):
        i = self.selected()
        if i is not None and self.name_var.get().strip() and \
                self.name_var.get().strip() != self.model.jobs[i]['name']:
            self.model.rename(i, self.name_var.get())
            self.refresh(select=i)

    def add(self):
        exts = ' '.join('*%s *%s' % (e, e.upper()) for e in VIDEO_EXTENSIONS)
        paths = self.filedialog.askopenfilenames(
            parent=self.root, title='Choose videos (select both cameras of a recording together)',
            initialdir=self.model.video_dir, filetypes=[('Videos', exts), ('All files', '*')])
        before = len(self.model.jobs)
        if self.model.add_videos(list(paths)):
            self.refresh(select=before)
            self.on_select()

    def remove(self):
        i = self.selected()
        if i is not None:
            self.model.remove(i)
            self.refresh(select=i)
            self.on_select()

    def swap(self):
        i = self.selected()
        if i is not None:
            self.model.swap(i)
            self.refresh(select=i)

    # ------------------------------------------------------------ fields

    def browse_out(self):
        d = self.filedialog.askdirectory(parent=self.root, title='Save results to',
                                         initialdir=self.model.out_dir or self.model.video_dir,
                                         mustexist=False)
        if d:
            self.out_var.set(d)

    def on_out(self):
        self.model.out_dir = self.out_var.get().strip()
        self.refresh()

    def browse_seed(self):
        d = self.filedialog.askdirectory(parent=self.root,
                                         title='A previous analysis folder (the one holding configs/)',
                                         initialdir=self.model.seed_path or self.model.out_dir
                                         or self.model.video_dir)
        if d:
            self.seed_path_var.set(d)
            self.seed_var.set('analysis')
            self.on_seed()

    def on_seed(self):
        self.model.seed_kind = self.seed_var.get()
        self.model.seed_path = self.seed_path_var.get().strip()
        self.refresh()

    def start(self):
        self.apply_name()
        m = self.model
        m.csv = bool(self.csv_var.get())
        m.speed = float(self.speed_var.get().rstrip('x'))
        problems = m.problems()
        if problems:
            self.status.configure(text=problems[0])
            self.root.bell()
            return
        self.result = m.settings(self.rig_path)
        self.root.destroy()

    def cancel(self):
        self.result = None
        self.root.destroy()

    def run(self):
        self.root.mainloop()
        return self.result


def main():
    ap = argparse.ArgumentParser(description='Launcher window for kinefly2.apps.analyze_videos')
    ap.add_argument('--rig', required=True, help='rig.yaml naming the cameras')
    ap.add_argument('--out-json', required=True, help='Write the chosen settings here')
    args = ap.parse_args()
    cams, primary = rig_cameras(args.rig)
    if not cams:
        print('%s names no cameras' % args.rig)
        return 2
    model = LauncherModel(cams, primary, load_state())
    result = LauncherWindow(model, args.rig).run()
    save_state(model.state())
    if result is None:
        return 1
    with open(args.out_json, 'w') as f:
        json.dump(result, f, indent=1)
    return 0


if __name__ == '__main__':
    sys.exit(main())
