"""
TNG 설정 창 — 왼쪽 분류 목록 + 오른쪽 내용 (묶음 단위), 확인 / 취소 / 적용.

분류: 일반 · Radio · 오디오 · 모드 · 수신/표시 · 매크로 (19부: 순서 · 묶음 모양 · '?' 없음)
Radio: WSJT-X Radio 탭과 같은 항목 · 배치 (영어), TNG 전용 항목은 아래 한 묶음.
값은 창 안에서만 바꾸다가 '적용' 또는 '확인' 때 MainWindow.apply_settings(dict) 로 한 번에 넘긴다.
송수신 중에는 오디오 · Radio 를 잠근다 (잠긴 이유 표시).
"""
import time

import sounddevice as sd
from PySide6.QtCore import Qt, QTimer
from PySide6.QtWidgets import (QDialog, QListWidget, QListWidgetItem, QStackedWidget, QWidget, QVBoxLayout,
                               QHBoxLayout, QGridLayout, QLabel, QLineEdit, QComboBox, QSlider, QFrame,
                               QPushButton, QDialogButtonBox, QFileDialog, QTableWidget, QTableWidgetItem,
                               QHeaderView, QRadioButton, QButtonGroup, QProgressBar, QSpinBox, QDoubleSpinBox,
                               QScrollArea, QCompleter, QAbstractSpinBox, QSizePolicy)

import theme as T
import rig
from strings_ko import tr

PAGES = ("일반", "Radio", "오디오", "모드", "수신/표시", "매크로")
PAGE_KEY = {"일반": "set.page.general", "Radio": "set.page.radio", "오디오": "set.page.audio", "모드": "set.page.mode",
            "수신/표시": "set.page.display", "매크로": "set.page.macro"}         # 쪽 이름 → 화면 글자 키


# ---------------------------------------------------------------- 묶음 (제목 + 1px 테두리 칸, 줄 사이 1px 선)
class Group(QWidget):
    def __init__(self, title=None):
        super().__init__()
        v = QVBoxLayout(self)
        v.setContentsMargins(0, 0, 0, 0)
        v.setSpacing(T.sp(1))
        if title:
            t = QLabel(title)
            t.setObjectName("groupTitle")
            v.addWidget(t)
        self.box = QFrame()
        self.box.setObjectName("group")
        self.rows = QVBoxLayout(self.box)
        self.rows.setContentsMargins(T.sp(3), T.sp(1), T.sp(3), T.sp(1))
        self.rows.setSpacing(0)
        v.addWidget(self.box)
        self._n = 0

    def add(self, label, w, stretch=False):
        """한 줄: 왼쪽 라벨 · 오른쪽 값 (라벨 None 이면 값만 전체 폭)"""
        if self._n:
            ln = QFrame()
            ln.setObjectName("groupLine")
            ln.setFixedHeight(1)
            self.rows.addWidget(ln)
        row = QWidget()
        h = QHBoxLayout(row)
        h.setContentsMargins(0, T.sp(1), 0, T.sp(1))
        h.setSpacing(T.sp(3))
        if label is not None:
            lb = QLabel(label)
            lb.setObjectName("rowLabel")
            h.addWidget(lb)
            if not stretch:
                h.addStretch(1)
        if isinstance(w, QWidget):
            if stretch or label is None:
                h.addWidget(w, 1)
            else:
                h.addWidget(w)
        else:
            h.addLayout(w, 1 if (stretch or label is None) else 0)
        self.rows.addWidget(row)
        self._n += 1
        return row


def radios(items, cur):
    """[(data, label)] → (가로 줄, QButtonGroup). 고른 값 = grp.checkedButton().property('data')"""
    h = QHBoxLayout()
    h.setSpacing(T.sp(3))
    grp = QButtonGroup(h)
    for data, lab in items:
        b = QRadioButton(lab)
        b.setProperty("data", data)
        grp.addButton(b)
        h.addWidget(b)
        if data == cur:
            b.setChecked(True)
    if grp.checkedButton() is None and grp.buttons():
        grp.buttons()[0].setChecked(True)
    return h, grp


def rval(grp):
    b = grp.checkedButton()
    return b.property("data") if b is not None else None


def combo(items, cur, w=None):
    cb = QComboBox()
    for data, lab in items:
        cb.addItem(lab, data)
    cb.setCurrentIndex(max(0, cb.findData(cur)))
    if w:
        cb.setMinimumWidth(w)
    return cb


def spin(lo, hi, val, step=1, suffix="", w=90):
    sp_ = QSpinBox()
    sp_.setRange(lo, hi)
    sp_.setSingleStep(step)
    sp_.setValue(int(val))
    sp_.setSuffix(suffix)
    sp_.setButtonSymbols(QAbstractSpinBox.NoButtons)          # 휠 · 입력 (잘린 위아래 버튼 없음)
    sp_.setAlignment(Qt.AlignRight)
    sp_.setFixedWidth(w)
    return sp_


class SettingsDialog(QDialog):
    def __init__(self, win):
        super().__init__(win)
        self.win = win
        self.setWindowTitle(tr("main.settings"))
        self.setObjectName("settingsDialog")
        self.resize(900, 680)
        s = win.settings
        self.lst = QListWidget()
        self.lst.setObjectName("settingsNav")
        self.lst.setFixedWidth(170)
        self.stack = QStackedWidget()
        for name in PAGES:
            self.lst.addItem(QListWidgetItem(tr(PAGE_KEY[name])))
        self.pages = {}
        self.audio_widgets, self.rig_widgets = (), []
        build = {"일반": self._p_general, "Radio": self._p_radio, "오디오": self._p_audio, "모드": self._p_mode,
                 "수신/표시": self._p_display, "매크로": self._p_macro}
        for name in PAGES:
            w = QWidget()
            w.setObjectName("plainPage")
            v = QVBoxLayout(w)
            v.setContentsMargins(T.sp(6), T.sp(4), T.sp(6), T.sp(4))
            v.setSpacing(T.sp(4))
            t = QLabel(tr(PAGE_KEY[name]))
            t.setObjectName("pageTitle")
            v.addWidget(t)
            build[name](v, s)
            if name not in ("Radio", "매크로"):
                v.addStretch(1)
            sc = QScrollArea()
            sc.setObjectName("plainScroll")
            sc.setWidget(w)
            sc.setWidgetResizable(True)
            sc.setFrameShape(QFrame.NoFrame)
            sc.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
            self.pages[name] = w
            self.stack.addWidget(sc)
        self.lst.currentRowChanged.connect(self.stack.setCurrentIndex)
        self.lst.setCurrentRow(0)
        bb = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel | QDialogButtonBox.Apply)
        bb.button(QDialogButtonBox.Ok).setText(tr("set.ok"))
        bb.button(QDialogButtonBox.Cancel).setText(tr("set.cancel"))
        bb.button(QDialogButtonBox.Apply).setText(tr("set.apply"))
        bb.accepted.connect(self._ok)
        bb.rejected.connect(self.reject)
        bb.button(QDialogButtonBox.Apply).clicked.connect(self.apply)
        for b in self.findChildren(QPushButton) + [bb.button(x) for x in (QDialogButtonBox.Ok, QDialogButtonBox.Cancel,
                                                                           QDialogButtonBox.Apply)]:
            b.setAutoDefault(False)                                # 기본 버튼 강조 (시스템 강조색) 없음
            b.setDefault(False)
        right = QVBoxLayout()
        right.setContentsMargins(0, 0, T.sp(3), T.sp(3))
        right.addWidget(self.stack, 1)
        right.addWidget(bb)
        body = QHBoxLayout(self)
        body.setContentsMargins(0, 0, 0, 0)
        body.setSpacing(0)
        body.addWidget(self.lst)
        body.addLayout(right, 1)
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._tick)
        self._timer.start(100)
        self._tick()

    # ---------------------------------------------------------------- 일반
    def _p_general(self, v, s):
        g = Group(tr("set.operator"))
        self.ed_my = QLineEdit(s["mycall"] or "")
        self.ed_my.setPlaceholderText(tr("main.e_g_6l5tng"))
        self.ed_my.setObjectName("call")
        self.ed_my.setFixedWidth(200)
        g.add(tr("set.my_call"), self.ed_my)
        v.addWidget(g)
        g = Group(tr("set.display"))
        self.cb_time = combo([("both", tr("set.utc_local")), ("utc", "UTC"), ("local", tr("set.local"))], s["time_display"], 200)
        g.add(tr("set.clock"), self.cb_time)
        import strings_ko
        self.cb_lang = combo(list(strings_ko.LANGS), s["language"] or "ko", 200)
        self.lb_lang = QLabel(tr("set.lang_restart"))
        self.lb_lang.setObjectName("dim")
        self.lb_lang.setVisible(False)
        self.cb_lang.currentIndexChanged.connect(
            lambda *_: self.lb_lang.setVisible(self.cb_lang.currentData() != strings_ko.LANG))
        lw = QWidget()
        lh = QHBoxLayout(lw)
        lh.setContentsMargins(0, 0, 0, 0)
        lh.addWidget(self.cb_lang)
        lh.addWidget(self.lb_lang)
        lh.addStretch(1)
        g.add(tr("set.language"), lw)
        v.addWidget(g)
        g = Group(tr("set.setup_group"))
        hb = QHBoxLayout()
        b1, b2 = QPushButton(tr("set.rerun_setup")), QPushButton(tr("set.rerun_diag"))
        b1.clicked.connect(lambda: (self.reject(), QTimer.singleShot(0, self.win.open_setup)))
        b2.clicked.connect(self.win.open_diag)
        hb.addWidget(b1)
        hb.addWidget(b2)
        hb.addStretch(1)
        g.add(None, hb)
        v.addWidget(g)
        g = Group(tr("set.logging"))
        row = QHBoxLayout()
        self.ed_log = QLineEdit(s.log_dir())
        b = QPushButton(tr("set.browse"))
        b.clicked.connect(self._pick_log)
        row.addWidget(self.ed_log, 1)
        row.addWidget(b)
        g.add(tr("set.log_folder"), row, stretch=True)
        v.addWidget(g)

    def _pick_log(self):
        d = QFileDialog.getExistingDirectory(self, tr("set.log_folder"), self.ed_log.text())
        if d:
            self.ed_log.setText(d)

    # ---------------------------------------------------------------- 오디오
    def _p_audio(self, v, s):
        self.lb_lock = QLabel("")
        self.lb_lock.setObjectName("warn")
        v.addWidget(self.lb_lock)
        g = Group(tr("set.devices"))
        self.cb_api, self.cb_in, self.cb_out = QComboBox(), QComboBox(), QComboBox()
        try:
            apis = [a["name"] for a in sd.query_hostapis()]
        except Exception:
            apis = []
        self.cb_api.addItems(apis)
        self.cb_api.setCurrentText(self.win.cb_api.currentText() or s["audio_api"])
        self.cb_api.currentTextChanged.connect(self._fill_dev)
        for lab, cb in ((tr("main.audio_api"), self.cb_api), (tr("main.input"), self.cb_in), (tr("main.output"), self.cb_out)):
            cb.setMinimumWidth(320)
            g.add(lab, cb)
        v.addWidget(g)
        g = Group(tr("set.level"))
        vol = QHBoxLayout()
        self.sl_vol = QSlider(Qt.Horizontal)
        self.sl_vol.setRange(0, 100)
        self.sl_vol.setValue(self.win.vol.value())
        self.sl_vol.setFixedWidth(240)
        self.lb_vol = QLabel("{}%".format(self.sl_vol.value()))
        self.lb_vol.setObjectName("num")
        self.lb_vol.setMinimumWidth(40)
        self.lb_vol.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self.sl_vol.valueChanged.connect(lambda x: self.lb_vol.setText("{}%".format(x)))
        vol.addWidget(self.sl_vol)
        vol.addWidget(self.lb_vol)
        g.add(tr("main.tx_level"), vol)
        self.pb_level = QProgressBar()
        self.pb_level.setRange(-60, 0)
        self.pb_level.setFormat("%v dBFS")
        self.pb_level.setFixedWidth(284)
        g.add(tr("meter.level"), self.pb_level)
        v.addWidget(g)
        self._fill_dev()
        self.audio_widgets = (self.cb_api, self.cb_in, self.cb_out)

    def _fill_dev(self):
        api_name = self.cb_api.currentText()
        cur_in = self.win.cb_in.currentText() or self.win.settings["audio_in"]
        cur_out = self.win.cb_out.currentText() or self.win.settings["audio_out"]
        for cb in (self.cb_in, self.cb_out):
            cb.clear()
        try:
            apis = [a["name"] for a in sd.query_hostapis()]
            devs = sd.query_devices()
        except Exception:
            return
        for d in devs:
            if apis[d["hostapi"]] != api_name:
                continue
            if d["max_input_channels"] > 0:
                self.cb_in.addItem(d["name"])
            if d["max_output_channels"] > 0:
                self.cb_out.addItem(d["name"])
        for cb, cur in ((self.cb_in, cur_in), (self.cb_out, cur_out)):
            i = cb.findText(cur)
            if i >= 0:
                cb.setCurrentIndex(i)

    # ---------------------------------------------------------------- 모드 · 표시 · 매크로
    def _p_mode(self, v, s):
        g = Group(tr("set.startup"))
        self.cb_txm = combo([("last", tr("set.last_mode")), ("TNG44", tr("set.always_tng44"))], s["tx_start_mode"], 200)
        g.add(tr("set.mode_at_start"), self.cb_txm)
        v.addWidget(g)

    def _p_display(self, v, s):
        g = Group(tr("main.rx"))
        self.sp_font = spin(self.win.qrx.SIZE_MIN, self.win.qrx.SIZE_MAX, int(self.win.qrx.size), 1, " pt")
        g.add(tr("set.rx_font_size"), self.sp_font)
        v.addWidget(g)
        g = Group(tr("main.waterfall"))
        self.cb_cmap = QComboBox()
        self.cb_cmap.addItems(T.WATERFALL_CMAPS)
        self.cb_cmap.setCurrentText(self.win.waterfall.cmap_name)
        self.cb_cmap.setMinimumWidth(200)
        g.add(tr("main.palette"), self.cb_cmap)
        v.addWidget(g)

    def _p_macro(self, v, s):
        self.tbl = QTableWidget(0, 2)
        self.tbl.setHorizontalHeaderLabels([tr("set.name"), tr("set.text")])
        self.tbl.horizontalHeader().setSectionResizeMode(1, QHeaderView.Stretch)
        self.tbl.verticalHeader().setVisible(False)
        for m in self.win.macros.macros:
            self._add_row(m["name"], m["text"])
        v.addWidget(self.tbl, 1)
        row = QHBoxLayout()
        b1, b2 = QPushButton(tr("set.add")), QPushButton(tr("set.delete"))
        b1.clicked.connect(lambda: self._add_row(tr("set.new_macro"), "{DXCALL} DE {MYCALL}"))
        b2.clicked.connect(lambda: self.tbl.removeRow(self.tbl.currentRow()) if self.tbl.currentRow() >= 0 else None)
        row.addWidget(b1)
        row.addWidget(b2)
        row.addStretch(1)
        v.addLayout(row)

    def _add_row(self, name, text):
        r = self.tbl.rowCount()
        self.tbl.insertRow(r)
        self.tbl.setItem(r, 0, QTableWidgetItem(name))
        self.tbl.setItem(r, 1, QTableWidgetItem(text))

    # ---------------------------------------------------------------- Radio (WSJT-X Radio 탭 항목 · 배치)
    def _p_radio(self, v, s):
        c = self.win.rig_cfg()
        self.lb_rig_lock = QLabel("")
        self.lb_rig_lock.setObjectName("warn")
        v.addWidget(self.lb_rig_lock)
        ports = self._serial_ports()
        # 맨 위: Rig + Poll Interval
        top = Group(None)
        self.cb_model = QComboBox()
        self.cb_model.setEditable(True)
        self.cb_model.setInsertPolicy(QComboBox.NoInsert)
        self.cb_model.addItem("None", None)
        for num, name in rig.rig_models(c["rigctld"]):
            self.cb_model.addItem("{}  ({})".format(name, num), num)
        comp = self.cb_model.completer()
        comp.setFilterMode(Qt.MatchContains)
        comp.setCompletionMode(QCompleter.PopupCompletion)
        self.cb_model.setCurrentIndex(0 if c["cat"] == "none" else max(0, self.cb_model.findData(int(c["model"]))))
        self.cb_model.setMinimumWidth(320)
        self.cb_model.setSizeAdjustPolicy(QComboBox.AdjustToMinimumContentsLengthWithIcon)
        top.add("Rig", self.cb_model)
        self.sp_poll = spin(1, 99, max(1, round(int(c["poll_ms"]) / 1000.0)), 1, " s", 70)
        top.add("Poll Interval", self.sp_poll)
        v.addWidget(top)

        cols = QHBoxLayout()
        cols.setSpacing(T.sp(4))
        left, right = QVBoxLayout(), QVBoxLayout()
        left.setSpacing(T.sp(4))
        right.setSpacing(T.sp(4))
        # 왼쪽: CAT Control
        cat = Group("CAT Control")
        self.cb_port = QComboBox()
        self.cb_port.setEditable(True)
        self.cb_port.addItems([""] + ports)
        self.cb_port.setCurrentText(c["port"])
        self.cb_port.setMinimumWidth(150)
        cat.add("Serial Port", self.cb_port)
        self.cb_baud = combo([(b, str(b)) for b in (1200, 2400, 4800, 9600, 19200, 38400, 57600, 115200)],
                             int(c["baud"]), 150)
        cat.add("Baud Rate", self.cb_baud)
        h, self.g_data = radios([(0, "Default"), (7, "Seven"), (8, "Eight")], int(c["data_bits"]))
        cat.add("Data Bits", h)
        h, self.g_stop = radios([(0, "Default"), (1, "One"), (2, "Two")], int(c["stop_bits"]))
        cat.add("Stop Bits", h)
        h, self.g_hs = radios([("Default", "Default"), ("None", "None"), ("XONXOFF", "XON/XOFF"),
                               ("Hardware", "Hardware")], c["handshake"])
        cat.add("Handshake", h)
        lines = [("Unset", ""), ("ON", "High"), ("OFF", "Low")]
        fl = QHBoxLayout()
        fl.setSpacing(T.sp(2))
        self.cb_dtr, self.cb_rts = combo(lines, c["dtr"], 70), combo(lines, c["rts"], 70)
        for lab, cb in (("DTR", self.cb_dtr), ("RTS", self.cb_rts)):
            lb = QLabel(lab)
            lb.setObjectName("dim")
            fl.addWidget(lb)
            fl.addWidget(cb)
        cat.add("Force Control Lines", fl)
        left.addWidget(cat)
        left.addStretch(1)
        # 오른쪽: PTT Method · Transmit Audio Source · Mode · Split Operation
        ptt = Group("PTT Method")
        h, self.g_ptt = radios([("VOX", "VOX"), ("DTR", "DTR"), ("CAT", "CAT"), ("RTS", "RTS")], c["ptt"])
        ptt.add(None, h)
        self.cb_pttport = QComboBox()
        self.cb_pttport.setEditable(True)
        self.cb_pttport.addItems([""] + ports)
        self.cb_pttport.setCurrentText(c["ptt_port"])
        self.cb_pttport.setMinimumWidth(150)
        ptt.add("Port", self.cb_pttport)
        right.addWidget(ptt)
        src = Group("Transmit Audio Source")
        h, self.g_src = radios([("data", "Rear/Data"), ("mic", "Front/Mic")], c["tx_source"])
        src.add(None, h)
        self.w_src = src
        right.addWidget(src)
        md = Group("Mode")
        h, self.g_mode = radios([("none", "None"), ("USB", "USB"), ("DATA", "Data/Pkt")], c["mode_set"])
        md.add(None, h)
        right.addWidget(md)
        sp_ = Group("Split Operation")
        h, self.g_split = radios([("none", "None"), ("rig", "Rig"), ("fake", "Fake It")], c["split"])
        sp_.add(None, h)
        right.addWidget(sp_)
        right.addStretch(1)
        cols.addLayout(left, 1)
        cols.addLayout(right, 1)
        v.addLayout(cols)
        # 아래: Test CAT · Test PTT · Tx delay
        trow = QHBoxLayout()
        trow.setSpacing(T.sp(2))
        self.btn_cat_test = QPushButton("Test CAT")
        self.btn_ptt_test = QPushButton("Test PTT")
        self.btn_cat_test.clicked.connect(self._cat_test)
        self.btn_ptt_test.clicked.connect(self._ptt_test)
        self.lb_rig_test = QLabel("")
        self.lb_rig_test.setObjectName("testResult")
        trow.addWidget(self.btn_cat_test)
        trow.addWidget(self.btn_ptt_test)
        trow.addWidget(self.lb_rig_test, 1)
        lb = QLabel("Tx delay")
        lb.setObjectName("rowLabel")
        self.sp_delay = QDoubleSpinBox()
        self.sp_delay.setRange(0.0, 2.0)
        self.sp_delay.setSingleStep(0.1)
        self.sp_delay.setDecimals(1)
        self.sp_delay.setSuffix(" s")
        self.sp_delay.setValue(int(c["tx_delay_ms"]) / 1000.0)
        self.sp_delay.setButtonSymbols(QAbstractSpinBox.NoButtons)
        self.sp_delay.setAlignment(Qt.AlignRight)
        self.sp_delay.setFixedWidth(70)
        trow.addWidget(lb)
        trow.addWidget(self.sp_delay)
        v.addLayout(trow)
        # TNG 전용 (WSJT-X 배치 아래 한 묶음)
        ex = Group("TNG Options")
        self.cb_cat = combo([("run", "Start rigctld with TNG"), ("connect", "Connect to running rigctld")],
                            c["cat_via"], 260)
        ex.add("rigctld", self.cb_cat)
        hp = QHBoxLayout()
        hp.setSpacing(T.sp(2))
        self.ed_host = QLineEdit(c["host"])
        self.ed_host.setFixedWidth(160)
        self.sp_tcp = spin(1, 65535, c["tcp_port"], 1, "", 80)
        hp.addWidget(self.ed_host)
        hp.addWidget(QLabel(":"))
        hp.addWidget(self.sp_tcp)
        ex.add("rigctld Address", hp)
        self.ed_rigctld = QLineEdit(c["rigctld"])
        self.ed_rigctld.setPlaceholderText(rig.find_exe("rigctld") or "rigctld.exe")
        ex.add("rigctld Path", self.ed_rigctld, stretch=True)
        self.sp_max = spin(10, 3600, c["max_tx_s"], 10, " s", 80)
        ex.add("Max TX Time", self.sp_max)
        self.sp_vox = spin(0, 2000, c["vox_lead_ms"], 10, " ms", 80)
        ex.add("VOX Lead Tone", self.sp_vox)
        v.addWidget(ex)
        bands = Group("Bands")
        bw = QWidget()
        bl = QVBoxLayout(bw)
        bl.setContentsMargins(0, T.sp(1), 0, T.sp(1))
        self.tbl_band = QTableWidget(0, 2)
        self.tbl_band.setHorizontalHeaderLabels(["Band", "Dial (MHz)"])
        self.tbl_band.horizontalHeader().setSectionResizeMode(1, QHeaderView.Stretch)
        self.tbl_band.verticalHeader().setVisible(False)
        self.tbl_band.setMinimumHeight(180)
        for n, hz in (s["bands"] or rig.DEFAULT_BANDS):
            self._band_row(n, hz)
        bl.addWidget(self.tbl_band)
        br = QHBoxLayout()
        b1, b2, b3 = QPushButton("Add"), QPushButton("Delete"), QPushButton("Defaults")
        b1.clicked.connect(lambda: self._band_row("New", 14078000))
        b2.clicked.connect(lambda: self.tbl_band.removeRow(self.tbl_band.currentRow()) if self.tbl_band.currentRow() >= 0 else None)
        b3.clicked.connect(self._band_defaults)
        for b in (b1, b2, b3):
            br.addWidget(b)
        br.addStretch(1)
        bl.addLayout(br)
        bands.add(None, bw)
        v.addWidget(bands)
        v.addStretch(1)
        self.rig_widgets = [w for w in self.pages.get("Radio", v.parentWidget()).findChildren(QWidget)
                            if isinstance(w, (QComboBox, QSpinBox, QDoubleSpinBox, QLineEdit, QPushButton, QRadioButton))]
        self.g_ptt.buttonToggled.connect(lambda *_: self._radio_enable())
        self.cb_model.currentIndexChanged.connect(lambda *_: self._radio_enable())
        self._radio_enable()

    def _radio_enable(self):
        """WSJT-X 처럼: Rig None 이면 CAT 칸 끔, Port 는 DTR/RTS 일 때만, Transmit Audio Source 는 CAT PTT 일 때만"""
        rig_on = self.cb_model.currentData() is not None
        ptt = rval(self.g_ptt)
        for w in (self.cb_port, self.cb_baud, self.cb_dtr, self.cb_rts, self.sp_poll):
            w.setEnabled(rig_on)
        for g in (self.g_data, self.g_stop, self.g_hs, self.g_mode, self.g_split):
            for b in g.buttons():
                b.setEnabled(rig_on)
        for b in self.g_ptt.buttons():
            if b.property("data") == "CAT":
                b.setEnabled(rig_on)
        self.cb_pttport.setEnabled(ptt in ("DTR", "RTS"))
        self.w_src.setEnabled(rig_on and ptt == "CAT")
        self.btn_cat_test.setEnabled(rig_on)

    @staticmethod
    def _serial_ports():
        try:
            from serial.tools import list_ports
            return sorted(p.device for p in list_ports.comports())
        except Exception:
            return []

    def _band_row(self, name, hz):
        r = self.tbl_band.rowCount()
        self.tbl_band.insertRow(r)
        self.tbl_band.setItem(r, 0, QTableWidgetItem(str(name)))
        self.tbl_band.setItem(r, 1, QTableWidgetItem("{:.6f}".format(int(hz) / 1e6).rstrip("0").rstrip(".")))

    def _band_defaults(self):
        self.tbl_band.setRowCount(0)
        for n, hz in rig.DEFAULT_BANDS:
            self._band_row(n, hz)

    def rig_values(self):
        c = self.win.rig_cfg()
        model = self.cb_model.currentData()
        ptt = rval(self.g_ptt)
        via = self.cb_cat.currentData()
        c.update({"cat": "none" if model is None else via, "cat_via": via, "model": int(model or c["model"] or 1),
                  "port": self.cb_port.currentText().strip(), "baud": int(self.cb_baud.currentData()),
                  "data_bits": int(rval(self.g_data)), "stop_bits": int(rval(self.g_stop)),
                  "handshake": rval(self.g_hs), "dtr": self.cb_dtr.currentData(), "rts": self.cb_rts.currentData(),
                  "host": self.ed_host.text().strip() or "127.0.0.1", "tcp_port": int(self.sp_tcp.value()),
                  "poll_ms": int(self.sp_poll.value()) * 1000, "mode_set": rval(self.g_mode),
                  "split": rval(self.g_split), "ptt": ptt if (model is not None or ptt != "CAT") else "VOX",
                  "ptt_port": self.cb_pttport.currentText().strip(), "tx_source": rval(self.g_src),
                  "tx_delay_ms": int(round(self.sp_delay.value() * 1000)),
                  "vox_lead_ms": int(self.sp_vox.value()), "max_tx_s": int(self.sp_max.value()),
                  "rigctld": self.ed_rigctld.text().strip()})
        return c

    def band_values(self):
        out = []
        for r in range(self.tbl_band.rowCount()):
            n, f = self.tbl_band.item(r, 0), self.tbl_band.item(r, 1)
            hz = rig.parse_hz(f.text()) if f else None
            if n and n.text().strip() and hz:
                out.append([n.text().strip(), int(hz)])
        return out

    def _set_result(self, text, state):
        """결과 한 줄: 차분한 글자색만 (ok · err · 진행)"""
        self.lb_rig_test.setText(text)
        self.lb_rig_test.setProperty("state", state)
        self.lb_rig_test.style().unpolish(self.lb_rig_test)
        self.lb_rig_test.style().polish(self.lb_rig_test)

    def _rig_apply_now(self):
        """테스트 전: 이 창의 Radio 값을 앱에 적용 (바뀌었으면 CAT 다시 시작)"""
        if self.win._transmitting:
            self._set_result("Transmitting — test not available", "err")
            return False
        self.win.apply_settings({"rig": self.rig_values()})
        return True

    def _cat_test(self):
        if not self._rig_apply_now():
            return
        r = self.win.rig
        if r is None or not r.cat_enabled():
            self._set_result("No rig selected", "err")
            return
        self._set_result("Connecting…", "busy")
        r.read_now()
        t0 = time.time()

        def poll():
            st = r.state
            if st.get("connected") and st.get("freq"):
                self._set_result("CAT OK  {}  {}".format(rig.fmt_hz(st["freq"]), st.get("mode") or ""), "ok")
                return
            if time.time() - t0 > 8.0 or r is not self.win.rig:
                self._set_result("CAT failed: " + (st.get("error") or "no response"), "err")
                return
            QTimer.singleShot(150, poll)
        QTimer.singleShot(150, poll)

    def _ptt_test(self):
        if not self._rig_apply_now():
            return
        r = self.win.rig
        self._set_result("PTT on…", "busy")
        t0 = time.time()

        def go():
            if r.cfg["ptt"] == "CAT" and not r.state.get("connected"):
                if time.time() - t0 < 8.0 and r is self.win.rig:
                    QTimer.singleShot(150, go)
                    return
                self._set_result("PTT failed: CAT not connected", "err")
                return
            self.win.ptt_test(3.0, lambda ok, msg: self._set_result(("PTT OK  " if ok else "PTT failed: ") + msg,
                                                                     "ok" if ok else "err"))
        go()

    # ---------------------------------------------------------------- 동작
    def _tick(self):
        busy = self.win.audio_locked()
        for w in self.audio_widgets:
            w.setEnabled(not busy)
        self.lb_lock.setText((tr("set.locked") + busy) if busy else "")
        self.lb_lock.setVisible(bool(busy))
        tx = self.win._transmitting or bool(self.win.ptt_busy())
        if tx != getattr(self, "_rig_locked", None):
            self._rig_locked = tx
            for w in self.rig_widgets:
                w.setEnabled(not tx)
            if not tx:
                self._radio_enable()
            self.lb_rig_lock.setText("Locked while transmitting" if tx else "")
            self.lb_rig_lock.setVisible(tx)
        lv = self.win.input_level_db()
        self.pb_level.setValue(int(max(-60, min(0, lv))) if lv is not None else -60)

    def values(self):
        macros = []
        for r in range(self.tbl.rowCount()):
            n, t = self.tbl.item(r, 0), self.tbl.item(r, 1)
            if n and t and t.text().strip():
                macros.append({"name": n.text().strip() or "매크로", "text": t.text()})
        d = {"mycall": self.ed_my.text().strip().upper(),
             "time_display": self.cb_time.currentData(),
             "language": self.cb_lang.currentData(),
             "log_dir": self.ed_log.text().strip(),
             "volume": self.sl_vol.value(),
             "tx_start_mode": self.cb_txm.currentData(),
             "qso_font": self.sp_font.value(),
             "cmap": self.cb_cmap.currentText(),
             "macros": macros,
             "bands": self.band_values()}
        if not (self.win._transmitting or self.win.ptt_busy()):
            d["rig"] = self.rig_values()
        if not self.win.audio_locked():
            d.update({"audio_api": self.cb_api.currentText(), "audio_in": self.cb_in.currentText(),
                      "audio_out": self.cb_out.currentText()})
        return d

    def apply(self):
        self.win.apply_settings(self.values())

    def _ok(self):
        self.apply()
        self.accept()
