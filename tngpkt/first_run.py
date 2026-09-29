"""
초기 설정 (24부): 설정 파일이 없을 때 첫 실행에만. 언어 → 호출부호 → 오디오 → 무전기 → 진단 → 완료.

  · 앱 창 모듈 (widgets · neuromod_app) 을 읽기 전에 띄운다 → 고른 언어가 import 때 만들어지는 문구에도 바로 들어감 (재시작 없음)
  · 끝까지 가야 설정 파일을 만든다. 중간에 닫으면 파일이 없으니 다음 실행 때 다시 뜬다
  · 설정 '일반 · 초기 설정 다시 하기' 로 다시 열 수 있음 (그때 언어 변경은 이 창에만 바로, 앱은 재시작 후)
진단 화면 (DiagPanel) 은 설정 '진단 다시 하기' 창과 같이 쓴다. 진단 자체는 diag.py (따로 프로세스)
"""
import json
import os
import sys
import time

import numpy as np
import sounddevice as sd
from PySide6.QtCore import Qt, QTimer, QProcess
from PySide6.QtWidgets import (QApplication, QDialog, QListWidget, QListWidgetItem, QStackedWidget, QWidget,
                               QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QComboBox, QLineEdit, QProgressBar,
                               QGridLayout, QCompleter, QSizePolicy)

from tngpkt import app_paths
from tngpkt import rig
from tngpkt import strings_ko
from tngpkt import theme as T
from tngpkt.strings_ko import tr
from tngpkt.settings_dialog import Group, combo, radios, rval

STEPS = ("lang", "call", "audio", "radio", "diag", "done")
SKIPPABLE = ("radio", "diag")
HERE = os.path.dirname(os.path.abspath(__file__))


def default_macros(lang):
    """매크로 기본값 (고른 언어로 이름). 내용은 두 언어 같음 (교신 문구는 영어)"""
    names = {"ko": ("CQ", "응답", "리포트", "QSL", "73"), "en": ("CQ", "Reply", "Report", "QSL", "73")}[lang if lang == "en" else "ko"]
    texts = ("CQ CQ CQ DE {MYCALL} {MYCALL} K", "{DXCALL} DE {MYCALL} TNX FER CALL UR SIG OK",
             "{DXCALL} DE {MYCALL} UR SNR {SNR} DB", "{DXCALL} DE {MYCALL} QSL TNX", "{DXCALL} DE {MYCALL} 73 SK")
    return [{"name": n, "text": t} for n, t in zip(names, texts)]


def needs_setup():
    return not os.path.exists(app_paths.settings_path())


# ====================================================================== 진단 화면 (초기 설정 · 설정 창 공용)
class DiagPanel(QWidget):
    """진단 시작 버튼 · 진행 막대 · 결과 줄. done(res) 로 결과를 알림"""
    GRADE_COL = {"ok": "ok", "warn": "fec", "bad": "err"}

    def __init__(self, get_devices, result=None, done=None):
        super().__init__()
        self.get_devices, self.done_cb, self.res = get_devices, done, result
        self.proc = None
        v = QVBoxLayout(self)
        v.setContentsMargins(0, 0, 0, 0)
        v.setSpacing(T.sp(3))
        h = QHBoxLayout()
        h.setSpacing(T.sp(3))
        self.btn = QPushButton(tr("diag.run"))
        self.btn.setAutoDefault(False)
        self.btn.clicked.connect(self.run)
        self.lb_step = QLabel("")                         # 진단 중: 지금 점검 중인 항목
        self.lb_step.setObjectName("dim")
        self.lb_time = QLabel("")                         # 결과 있을 때만: 점검 시각
        self.lb_time.setObjectName("dim")
        h.addWidget(self.btn)
        h.addWidget(self.lb_step)
        h.addStretch(1)
        h.addWidget(self.lb_time)
        v.addLayout(h)
        self.pb = QProgressBar()                          # 진행 막대: 진단 중에만, 얇게 (입력칸처럼 안 보이게)
        self.pb.setRange(0, 100)
        self.pb.setTextVisible(False)
        self.pb.setFixedHeight(4)
        self.pb.setStyleSheet("QProgressBar{{border:none;background:{};min-height:4px;max-height:4px;padding:0;}} QProgressBar::chunk{{background:{};}}".format(
            T.c("seg_off"), T.c("text3")))
        self.pb.hide()
        v.addWidget(self.pb)
        self.g = Group(None)
        self.grid = QGridLayout()
        self.grid.setHorizontalSpacing(T.sp(4))
        self.grid.setVerticalSpacing(T.sp(2))
        self.g.add(None, self.grid)
        v.addWidget(self.g)
        self._rows = []
        self.show_result(result)

    def show_result(self, res):
        from tngpkt import diag
        for w in self._rows:
            w.setParent(None)
        self._rows = []
        self.g.setVisible(bool(res))
        self.lb_time.setVisible(bool(res))
        if not res:
            return
        self.lb_time.setText(tr("diag.time").format(res.get("time", "")))
        for i, (name, g, why) in enumerate(diag.diag_view(res)):
            a, b, c = QLabel(name), QLabel(tr("diag." + g)), QLabel(why)
            a.setObjectName("rowLabel")
            b.setStyleSheet("color:{}".format(T.c(self.GRADE_COL[g])))
            c.setObjectName("dim")
            for j, w in enumerate((a, b, c)):
                self.grid.addWidget(w, i, j)
                self._rows.append(w)
        self.grid.setColumnStretch(2, 1)

    def run(self):
        if self.proc is not None:
            return
        api, din, dout = self.get_devices()
        self.btn.setEnabled(False)
        self.pb.setValue(0)                               # 버튼 글자는 그대로 (옆 줄이 점검 항목)
        self.pb.show()
        self.g.hide()
        self.lb_time.hide()
        self.lb_step.setText(tr("diag.step.start"))
        self._out = ""
        p = QProcess(self)
        p.setProcessChannelMode(QProcess.MergedChannels)
        p.readyReadStandardOutput.connect(self._read)
        p.finished.connect(self._fin)
        self.proc = p
        self._t0 = time.time()
        env = p.processEnvironment()
        worker = ["--diag-worker"] if getattr(sys, "frozen", False) else ["-m", "tngpkt.diag"]
        if not getattr(sys, "frozen", False):
            p.setWorkingDirectory(os.path.dirname(HERE))
        p.start(sys.executable, worker + ["--api", api or "MME", "--in", din or "", "--out", dout or ""])
        QTimer.singleShot(120000, self._timeout)

    def _read(self):
        self._out += bytes(self.proc.readAllStandardOutput()).decode("utf-8", "replace")
        for ln in self._out.splitlines():
            if ln.startswith("P "):
                try:
                    _, pc, what = (ln.split(None, 2) + [""])[:3]
                    self.pb.setValue(int(pc))
                    self.lb_step.setText(self.step_text(what))
                except ValueError:
                    pass

    @staticmethod
    def step_text(what):
        """diag.py 진행 단계 이름 → 한 줄 (지금 점검 중인 항목)"""
        w = what.split()
        if not w:
            return ""
        if w[0] in ("TNG44", "TNG5", "TNG1"):
            return tr("diag.step.mode1" if "1core" in w else "diag.step.mode").format(w[0])
        k = {"start": "start", "audio": "audio", "load": "load", "1core": "load", "done": "done"}.get(w[0])
        return tr("diag.step." + k) if k else ""

    def _timeout(self):
        if self.proc is not None and time.time() - self._t0 > 110:
            self.proc.kill()

    def _fin(self, *a):
        if self.proc is not None:
            self._out += bytes(self.proc.readAllStandardOutput()).decode("utf-8", "replace")
        self.proc = None
        self.btn.setEnabled(True)
        self.btn.setText(tr("diag.run"))
        self.pb.hide()
        self.lb_step.setText("")
        res = None
        for ln in self._out.splitlines():
            if ln.startswith("R "):
                try:
                    res = json.loads(ln[2:])
                except ValueError:
                    pass
        if res is None:
            last = [ln for ln in self._out.splitlines() if ln.strip() and not ln.startswith("P ")]
            self.lb_step.setText(tr("diag.failed").format(last[-1][:60] if last else "-"))
            return
        self.res = res
        self.show_result(res)
        if self.done_cb:
            self.done_cb(res)

    def stop(self):
        if self.proc is not None:
            self.proc.kill()
            self.proc.waitForFinished(2000)
            self.proc = None


class DiagDialog(QDialog):
    """설정 '진단 다시 하기': 지난 결과 보이고 다시 돌림, 결과는 설정에 저장"""

    def __init__(self, win):
        super().__init__(win)
        self.win = win
        self.setWindowTitle(tr("diag.title"))
        self.setObjectName("settingsDialog")
        self.resize(620, 330)
        v = QVBoxLayout(self)
        v.setContentsMargins(T.sp(6), T.sp(4), T.sp(6), T.sp(4))
        t = QLabel(tr("diag.title"))
        t.setObjectName("pageTitle")
        v.addWidget(t)
        s = win.settings
        self.panel = DiagPanel(lambda: (win.cb_api.currentText(), win.cb_in.currentText(), win.cb_out.currentText()),
                               s["diag"], lambda r: s.__setitem__("diag", r))
        v.addWidget(self.panel)
        v.addStretch(1)

    def done(self, r):
        self.panel.stop()
        super().done(r)


# ====================================================================== 초기 설정
class SetupWizard(QDialog):
    def __init__(self, parent=None, settings=None, win=None):
        """settings=None: 첫 실행 (끝에서 설정 파일 만듦). settings 주면 다시 하기 (그 설정에 저장, win 에 반영)"""
        super().__init__(parent)
        self.settings, self.win = settings, win
        s = settings.data if settings is not None else {}
        from tngpkt import app_settings
        r = dict(rig.DEFAULTS)
        r.update(s.get("rig") or {})
        self.vals = {"language": s.get("language") or strings_ko.LANG, "mycall": s.get("mycall") or "",
                     "audio_api": s.get("audio_api") or app_settings.DEFAULTS["audio_api"],
                     "audio_in": s.get("audio_in") or "", "audio_out": s.get("audio_out") or "", "rig": r,
                     "diag": s.get("diag")}
        self.step = 0
        self._stream, self._level, self._rig = None, None, None
        self.setObjectName("settingsDialog")
        self.resize(820, 560)
        if parent is None:
            self.setStyleSheet(T.load_qss())
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._tick)
        self._timer.start(100)
        self._build()

    # ---------------------------------------------------------------- 틀 (언어 바꾸면 통째로 다시)
    def _build(self):
        if self.layout() is not None:
            old = QWidget(self)
            old.hide()
            old.setLayout(self.layout())
            old.deleteLater()                                 # 언어 변경 신호 처리가 끝난 뒤 옛 배치 버리기
        self.setWindowTitle(tr("setup.title"))
        self.lst = QListWidget()
        self.lst.setObjectName("settingsNav")
        self.lst.setFixedWidth(170)
        self.lst.setFocusPolicy(Qt.NoFocus)
        self.lst.setSelectionMode(QListWidget.SingleSelection)
        for k in STEPS:
            it = QListWidgetItem(tr("setup.step." + k))
            self.lst.addItem(it)
        self.lst.currentRowChanged.connect(lambda r: r != self.step and self.lst.setCurrentRow(self.step))   # 눌러서 건너뛰지 않음
        self.stack = QStackedWidget()
        for k in STEPS:
            w = QWidget()
            w.setObjectName("plainPage")
            v = QVBoxLayout(w)
            v.setContentsMargins(T.sp(6), T.sp(4), T.sp(6), T.sp(4))
            v.setSpacing(T.sp(4))
            t = QLabel(tr("setup.step." + k))
            t.setObjectName("pageTitle")
            v.addWidget(t)
            hint = QLabel(tr("setup.hint." + k))
            hint.setObjectName("dim")
            hint.setWordWrap(True)
            v.addWidget(hint)
            getattr(self, "_p_" + k)(v)
            v.addStretch(1)
            self.stack.addWidget(w)
        self.b_prev, self.b_skip, self.b_next = QPushButton(tr("setup.prev")), QPushButton(tr("setup.skip")), QPushButton("")
        for b in (self.b_prev, self.b_skip, self.b_next):
            b.setAutoDefault(False)
            b.setDefault(False)
            b.setMinimumWidth(90)
        self.b_prev.clicked.connect(lambda: self._go(-1))
        self.b_skip.clicked.connect(lambda: self._go(1, skip=True))
        self.b_next.clicked.connect(lambda: self._go(1))
        bb = QHBoxLayout()
        bb.addStretch(1)
        bb.addWidget(self.b_prev)
        bb.addWidget(self.b_skip)
        bb.addWidget(self.b_next)
        right = QVBoxLayout()
        right.setContentsMargins(0, 0, T.sp(3), T.sp(3))
        right.addWidget(self.stack, 1)
        right.addLayout(bb)
        body = QHBoxLayout(self)
        body.setContentsMargins(0, 0, 0, 0)
        body.setSpacing(0)
        body.addWidget(self.lst)
        body.addLayout(right, 1)
        self._show()

    def _show(self):
        k = STEPS[self.step]
        self.stack.setCurrentIndex(self.step)
        self.lst.setCurrentRow(self.step)
        self.b_prev.setEnabled(self.step > 0)
        self.b_skip.setVisible(k in SKIPPABLE)
        self.b_next.setText(tr("setup.finish") if k == "done" else tr("setup.next"))
        if k == "audio":
            self._open_level()
        else:
            self._close_level()

    def _go(self, d, skip=False):
        k = STEPS[self.step]
        self._collect(k, skip)
        if k == "done" and d > 0:
            self._finish()
            return
        self.step = max(0, min(len(STEPS) - 1, self.step + d))
        self._show()

    # ---------------------------------------------------------------- 단계
    def _p_lang(self, v):
        g = Group(None)
        self.cb_lang = combo(list(strings_ko.LANGS), self.vals["language"], 200)
        self.cb_lang.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        self.cb_lang.currentIndexChanged.connect(self._lang_changed)
        hl = QHBoxLayout()
        hl.addWidget(self.cb_lang)
        hl.addStretch(1)
        g.add(None, hl)                                   # 칸 이름 없음 (단계 제목과 같음), 왼쪽 정렬
        v.addWidget(g)

    def _lang_changed(self, *a):
        lang = self.cb_lang.currentData()
        if lang == self.vals["language"]:
            return
        self.vals["language"] = lang
        strings_ko.set_lang(lang)                              # 이 창 · (첫 실행이면) 앱 전체 바로
        self._close_level()
        self._build()

    def _p_call(self, v):
        g = Group(None)
        self.ed_call = QLineEdit(self.vals["mycall"])
        self.ed_call.setObjectName("call")
        self.ed_call.setPlaceholderText(tr("main.e_g_6l5tng"))
        self.ed_call.setFixedWidth(200)
        hc = QHBoxLayout()
        hc.addWidget(self.ed_call)
        hc.addStretch(1)
        g.add(None, hc)                                   # 칸 이름 없음 (단계 제목과 같음), 왼쪽 정렬
        v.addWidget(g)

    def _p_audio(self, v):
        g = Group(None)
        self.cb_api, self.cb_in, self.cb_out = QComboBox(), QComboBox(), QComboBox()
        try:
            apis = [a["name"] for a in sd.query_hostapis()]
        except Exception:
            apis = []
        self.cb_api.addItems(apis)
        self.cb_api.setCurrentText(self.vals["audio_api"])
        self.cb_api.currentTextChanged.connect(self._fill_dev)
        self.b_tone = QPushButton(tr("setup.test_tone"))
        self.b_tone.setAutoDefault(False)
        self.b_tone.clicked.connect(self._tone)
        self.pb_level = QProgressBar()
        self.pb_level.setRange(-60, 0)
        self.pb_level.setFormat("%v dBFS")
        for cb in (self.cb_api, self.cb_in, self.cb_out):
            cb.setMinimumWidth(320)
        g.add(tr("main.audio_api"), self.cb_api)
        g.add(tr("main.input"), self.cb_in)
        self.pb_level.setFixedWidth(320)
        g.add(tr("meter.level"), self.pb_level)
        ho = QHBoxLayout()
        ho.setSpacing(T.sp(2))
        ho.addStretch(1)
        ho.addWidget(self.cb_out)
        ho.addWidget(self.b_tone)
        g.add(tr("main.output"), ho)
        v.addWidget(g)
        self._fill_dev()
        self.cb_in.currentIndexChanged.connect(lambda *_: self._open_level() if STEPS[self.step] == "audio" else None)

    def _fill_dev(self):
        api = self.cb_api.currentText()
        for cb in (self.cb_in, self.cb_out):
            cb.blockSignals(True)
            cb.clear()
        try:
            apis = [a["name"] for a in sd.query_hostapis()]
            devs = sd.query_devices()
            dflt = sd.default.device
        except Exception:
            apis, devs, dflt = [], [], (-1, -1)
        for i, d in enumerate(devs):
            if apis[d["hostapi"]] != api:
                continue
            if d["max_input_channels"] > 0:
                self.cb_in.addItem(d["name"], i)
            if d["max_output_channels"] > 0:
                self.cb_out.addItem(d["name"], i)
        for cb, key, di in ((self.cb_in, "audio_in", dflt[0]), (self.cb_out, "audio_out", dflt[1])):
            j = cb.findText(self.vals[key]) if self.vals[key] else -1
            if j < 0 and di is not None and di >= 0:
                j = cb.findText(devs[di]["name"]) if di < len(devs) else -1
            cb.setCurrentIndex(max(0, j))
            cb.blockSignals(False)
        if STEPS[self.step] == "audio":
            self._open_level()

    def _open_level(self):
        self._close_level()
        idx = self.cb_in.currentData() if hasattr(self, "cb_in") else None
        if idx is None:
            return
        self._level = None

        def cb(indata, frames, t, status):
            pk = float(np.max(np.abs(indata))) if len(indata) else 0.0
            self._level = max(pk, (self._level or 0.0) * 0.8)
        try:
            fs = float(sd.query_devices(idx)["default_samplerate"])
            self._stream = sd.InputStream(device=idx, channels=1, samplerate=fs, blocksize=2048, callback=cb)
            self._stream.start()
        except Exception:
            self._stream = None

    def _close_level(self):
        s, self._stream = self._stream, None
        if s is not None:
            try:
                s.stop()
                s.close()
            except Exception:
                pass

    def _tone(self):
        idx = self.cb_out.currentData()
        try:
            fs = float(sd.query_devices(idx)["default_samplerate"])
            t = np.arange(int(fs * 1.0)) / fs
            a = 0.3 * np.sin(2 * np.pi * 1500 * t) * np.minimum(1, np.minimum(t, t[-1] - t) / 0.02)
            sd.play(a.astype(np.float32), fs, device=idx)
        except Exception:
            pass

    def _p_radio(self, v):
        c = self.vals["rig"]
        g = Group(None)
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
        g.add("Rig", self.cb_model)
        from tngpkt.settings_dialog import SettingsDialog
        ports = SettingsDialog._serial_ports()
        self.cb_port = QComboBox()
        self.cb_port.setEditable(True)
        self.cb_port.addItems([""] + ports)
        self.cb_port.setCurrentText(c["port"])
        self.cb_port.setMinimumWidth(150)
        g.add("Serial Port", self.cb_port)
        self.cb_baud = combo([(b, str(b)) for b in (1200, 2400, 4800, 9600, 19200, 38400, 57600, 115200)], int(c["baud"]), 150)
        g.add("Baud Rate", self.cb_baud)
        h, self.g_ptt = radios([("VOX", "VOX"), ("CAT", "CAT"), ("DTR", "DTR"), ("RTS", "RTS")], c["ptt"])
        g.add("PTT Method", h)
        self.cb_pttport = QComboBox()
        self.cb_pttport.setEditable(True)
        self.cb_pttport.addItems([""] + ports)
        self.cb_pttport.setCurrentText(c["ptt_port"])
        self.cb_pttport.setMinimumWidth(150)
        g.add("PTT Port", self.cb_pttport)
        v.addWidget(g)
        h = QHBoxLayout()
        self.b_cat, self.b_ptt = QPushButton("Test CAT"), QPushButton("Test PTT")
        for b in (self.b_cat, self.b_ptt):
            b.setAutoDefault(False)
            h.addWidget(b)
        self.lb_rig = QLabel("")
        self.lb_rig.setObjectName("testResult")
        h.addWidget(self.lb_rig, 1)
        v.addLayout(h)
        self.b_cat.clicked.connect(self._cat_test)
        self.b_ptt.clicked.connect(self._ptt_test)

    def _rig_cfg(self):
        c = dict(self.vals["rig"])
        model = self.cb_model.currentData()
        ptt = rval(self.g_ptt)
        c.update({"cat": "none" if model is None else (c.get("cat_via") or "run"), "model": int(model or c["model"] or 1),
                  "port": self.cb_port.currentText().strip(), "baud": int(self.cb_baud.currentData()),
                  "ptt": ptt if (model is not None or ptt != "CAT") else "VOX", "ptt_port": self.cb_pttport.currentText().strip()})
        return c

    def _rig_obj(self):
        """다시 하기 (앱 떠 있음): 앱의 무전기 연결로 (rigctld 두 개가 같은 포트를 잡지 않게). 첫 실행: 이 창 전용"""
        cfg = self._rig_cfg()
        if self.win is not None:
            self.win.apply_settings({"rig": cfg})
            return self.win.rig
        if self._rig is None or self._rig.cfg != cfg:
            if self._rig is not None:
                self._rig.close()
            self._rig = rig.Rig(cfg)
            self._rig.start()
        return self._rig

    def _set_rig(self, text, state):
        self.lb_rig.setText(text)
        self.lb_rig.setProperty("state", state)
        self.lb_rig.style().unpolish(self.lb_rig)
        self.lb_rig.style().polish(self.lb_rig)

    def _cat_test(self):
        r = self._rig_obj()
        if r is None or not r.cat_enabled():
            self._set_rig("No rig selected", "err")
            return
        self._set_rig("Connecting…", "busy")
        r.read_now()
        t0 = time.time()

        def poll():
            st = r.state
            if st.get("connected") and st.get("freq"):
                self._set_rig("CAT OK  {}  {}".format(rig.fmt_hz(st["freq"]), st.get("mode") or ""), "ok")
            elif time.time() - t0 > 8.0:
                self._set_rig("CAT failed: " + (st.get("error") or "no response"), "err")
            else:
                QTimer.singleShot(150, poll)
        QTimer.singleShot(150, poll)

    def _ptt_test(self):
        r = self._rig_obj()
        if r is None:
            return
        if r.cfg["ptt"] == "VOX":
            self._set_rig("VOX: TNG PKT does not key the rig (audio keys VOX)", "ok")
            return
        self._set_rig("PTT on…", "busy")
        t0 = time.time()

        def go():
            if r.cfg["ptt"] == "CAT" and not r.state.get("connected"):
                if time.time() - t0 < 8.0:
                    QTimer.singleShot(150, go)
                    return
                self._set_rig("PTT failed: CAT not connected", "err")
                return
            err = r.ptt(True)
            if err:
                self._set_rig("PTT failed: " + err, "err")
                return

            def off():
                r.ptt(False)
                self._set_rig("PTT OK  {}".format(r.cfg["ptt"]), "ok")
            QTimer.singleShot(2000, off)
        go()

    def _p_diag(self, v):
        self.diag = DiagPanel(lambda: (self.vals["audio_api"], self.vals["audio_in"], self.vals["audio_out"]),
                              self.vals["diag"], lambda r: self.vals.__setitem__("diag", r))
        v.addWidget(self.diag)

    def _p_done(self, v):
        pass

    # ---------------------------------------------------------------- 값
    def _collect(self, k, skip=False):
        if k == "call":
            self.vals["mycall"] = self.ed_call.text().strip().upper()
        elif k == "audio":
            self.vals.update({"audio_api": self.cb_api.currentText(), "audio_in": self.cb_in.currentText(),
                              "audio_out": self.cb_out.currentText()})
        elif k == "radio" and not skip:
            self.vals["rig"] = self._rig_cfg()

    def result_settings(self):
        d = {"language": self.vals["language"], "mycall": self.vals["mycall"], "audio_api": self.vals["audio_api"],
             "audio_in": self.vals["audio_in"], "audio_out": self.vals["audio_out"], "rig": self.vals["rig"],
             "setup_done": True}
        if self.vals.get("diag"):
            d["diag"] = self.vals["diag"]
        return d

    def _finish(self):
        d = self.result_settings()
        if self.settings is None:                              # 첫 실행: 여기서 처음 설정 파일을 만든다
            from tngpkt import app_settings
            d["macros"] = default_macros(self.vals["language"])
            s = app_settings.Settings()
            s.update(d)
        elif self.win is not None:
            self.win.apply_settings(d)
        else:
            self.settings.update(d)
        self._cleanup()
        self.accept()

    def _cleanup(self):
        self._close_level()
        if hasattr(self, "diag"):
            self.diag.stop()
        if self._rig is not None:
            self._rig.close()
            self._rig = None

    def reject(self):
        self._cleanup()
        super().reject()

    def _tick(self):
        if STEPS[self.step] == "audio" and hasattr(self, "pb_level"):
            lv = self._level
            db = 20 * np.log10(lv) if lv and lv > 1e-6 else -60
            try:
                self.pb_level.setValue(int(max(-60, min(0, db))))
            except RuntimeError:
                pass


def maybe_run():
    """앱 시작 때 (창 모듈 읽기 전): 설정 파일 없으면 초기 설정. 끝내면 True, 닫으면 앱 종료"""
    if not needs_setup():
        return True
    app = QApplication.instance() or QApplication(sys.argv[:1])
    w = SetupWizard()
    ok = w.exec() == QDialog.Accepted
    if not ok:
        sys.exit(0)
    return True
