"""
프로토콜 대역 확대 워터폴 (고해상도). 계산은 analysis.ZoomAnalyzer 스레드, 여기서는 그리기만 한다.

  위: 스펙트럼 (최근 줄), 아래: 워터폴 (가로 = 주파수, 세로 = 시간, 위 = 최신) — 전체 대역 워터폴과 같은 방향
  가로축 연동 + 왼쪽 축 폭 고정(theme.AXIS_W) → 두 그래프가 픽셀 단위로 맞는다
  겹침 표시: ±250Hz 대역 경계, 코스타스 7개 톤 주파수, 검출된 주파수 오차 위치
  세로 dB 범위는 고정 (스펙트럼 · 워터폴 공통), '자동'은 누를 때 한 번만 맞춘다
"""

import time

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QComboBox, QSizePolicy

from tngpkt import theme as T
from PySide6.QtWidgets import QCheckBox
from tngpkt.widgets import _plot, _lab, DBRange, waterfall_auto_range, grid_rect, SegOverlay
from tngpkt.strings_ko import tr

CENTER, HALF_BW = 1500.0, 250.0
SPANS = (150, 300, 500)                 # 확대 범위 ±Hz (ZoomAnalyzer 의 약 2000Hz 데시메이션 → ±800Hz 까지 가능)


class ZoomPanel(QWidget):
    presetChanged = Signal(str)
    speedChanged = Signal(str)
    spanChanged = Signal(int)
    rangeChanged = Signal(float, float)
    ROWS = 240                          # 워터폴 줄 수 (시간 방향)

    def __init__(self, presets, preset, span=300, rng=None, cmap="inferno", tones=(), speed="보통"):
        super().__init__()
        v = QVBoxLayout(self)
        v.setContentsMargins(T.sp(1), T.sp(1), T.sp(1), T.sp(1))
        v.setSpacing(T.sp(1))
        row = QHBoxLayout()
        row.setContentsMargins(T.sp(1), 0, 0, 0)
        row.setSpacing(T.sp(2))
        row.addWidget(_lab(tr("zoom.span"), "dim"))
        self.cb_span = QComboBox()
        for s_ in SPANS:
            self.cb_span.addItem("±{} Hz".format(s_), s_)
        self.cb_span.setCurrentIndex(SPANS.index(span) if span in SPANS else 1)
        self.cb_span.currentIndexChanged.connect(self._span_changed)
        row.addWidget(self.cb_span)
        row.addWidget(_lab(tr("zoom.resolution"), "dim"))
        self.cb_res = QComboBox()
        for n_ in presets:
            self.cb_res.addItem(tr("zoom.res." + n_), n_)         # 값 = 프리셋 이름 (설정 저장), 글자 = 언어 따라
        self.cb_res.setCurrentIndex(max(0, self.cb_res.findData(preset)))
        self.cb_res.currentIndexChanged.connect(lambda *_: self._preset_changed(self.cb_res.currentData()))
        row.addWidget(self.cb_res)
        row.addWidget(_lab(tr("main.speed"), "dim"))
        self.cb_speed = QComboBox()
        for k_, n_ in (("느림", "wf.speed.slow"), ("보통", "wf.speed.normal"), ("빠름", "wf.speed.fast")):
            self.cb_speed.addItem(tr(n_), k_)
        self.cb_speed.setCurrentIndex(max(0, self.cb_speed.findData(speed)))
        self.cb_speed.currentIndexChanged.connect(lambda *_: self._speed_changed(self.cb_speed.currentData()))
        row.addWidget(self.cb_speed)
        self.lb_res = _lab("", "num")
        row.addWidget(self.lb_res)
        row.addStretch(1)
        v.addLayout(row)
        row2 = QHBoxLayout()                          # dBFS 범위는 둘째 줄 (한 줄이면 좁은 창에서 칸이 잘렸다)
        row2.setContentsMargins(T.sp(1), 0, 0, 0)
        self.chk_segs = QCheckBox(tr("main.segments"))
        self.chk_segs.setChecked(True)
        row2.addWidget(self.chk_segs)
        row2.addStretch(1)
        self.range = DBRange(*(rng or (-120, -20)))
        self.range.changed.connect(self._set_range)
        self.range.autoRequested.connect(self._auto)
        row2.addWidget(self.range)
        v.addLayout(row2)

        self.spec = _plot(None, None, "dBFS")
        self.spec.getAxis("left").setWidth(T.AXIS_W)
        self.spec.setMinimumHeight(110)
        self.curve = self.spec.plot(pen=T.pen("rx"))
        self.wf = _plot(None, tr("w.frequency_hz"), tr("w.elapsed_s"))
        self.wf.getAxis("left").setWidth(T.AXIS_W)
        self.wf.getPlotItem().layout.setContentsMargins(0, T.sp(2), 0, 0)
        self.wf.getViewBox().invertY(True)
        self.img = pg.ImageItem()
        self.wf.addItem(self.img)
        self.set_cmap(cmap)
        self.spec.setXLink(self.wf)
        # 겹침 표시 (두 그래프 모두)
        self.df_lines = []
        self._band = []
        for p in (self.spec, self.wf):
            for f, a_, st_ in ((CENTER, 110, None), (CENTER - HALF_BW, 150, Qt.DashLine), (CENTER + HALF_BW, 150, Qt.DashLine)):
                ln_ = pg.InfiniteLine(f, angle=90, pen=T.pen("wf_band", 1, a_, st_))
                p.addItem(ln_)
                self._band.append((ln_, f - CENTER))
            ln = pg.InfiniteLine(CENTER, angle=90, pen=T.pen("wf_a", 1.6))
            ln.setVisible(False)
            p.addItem(ln)
            self.df_lines.append(ln)
        self._tone_lines = []
        self._tone_key = None
        self.set_tones([(tones, "dot")] if tones else [])
        self.df_text = pg.TextItem("", color=T.c("wf_a"), anchor=(0, 0))
        self.df_text.setFont(T.mono_font(T.PT_SMALL))
        self.spec.addItem(self.df_text)
        v.addWidget(self.spec, 2)
        v.addWidget(self.wf, 5)

        self.span = self.cb_span.currentData()
        self.buf = None
        self._nvalid = 0
        self.freqs = None
        self.dt = 0.032
        self._last = None
        self.abs_top, self.fs_in = None, 48000.0
        self._ov_items = []
        self.overlay = SegOverlay(self.wf)
        self.chk_segs.toggled.connect(self._segs_toggled)
        self._set_range(*self.range.value())
        self._apply_span()

    # ---------------------------------------------------------------- 조작
    def set_cmap(self, name):
        self.img.setLookupTable(T.cmap_lut(name))

    def _span_changed(self, _):
        self.span = self.cb_span.currentData()
        self.buf = None
        self._apply_span()
        self.spanChanged.emit(int(self.span))

    def set_center(self, f):
        """중심 주파수 바꾸기 (16부): 대역선 · 가로축. 확대 분석기는 수신 흐름 (1500 Hz 로 되돌린 소리) 을 보므로 그리기에서 옮김"""
        global CENTER
        CENTER = float(f)
        for ln, off in self._band:
            ln.setValue(CENTER + off)
        self.buf = None
        self._apply_span()

    def _apply_span(self):
        self.wf.setXRange(CENTER - self.span, CENTER + self.span, padding=0)

    def _preset_changed(self, name):
        self.buf = None
        self.presetChanged.emit(name)

    def _speed_changed(self, name):
        self.buf = None
        self.speedChanged.emit(name)

    def _set_range(self, lo, hi):
        self.lo, self.hi = lo, hi
        self.spec.setYRange(lo, hi, padding=0)
        if self.buf is not None:
            self.img.setLevels((lo, hi))
        self.rangeChanged.emit(lo, hi)

    def _auto(self):
        """'자동' (누를 때 한 번): 최근 약 2초의 워터폴 줄로 잡음 바닥을 재서 하한, 하한 + 60dB 를 상한으로"""
        if self.buf is None or self._nvalid == 0:
            return
        r = waterfall_auto_range(self.buf[:min(self._nvalid, 64)])
        if r is not None:
            self.range.set_value(*r)

    def _segs_toggled(self, on):
        self.overlay.visible = bool(on)
        self.draw_overlay()

    def set_overlay(self, items):
        self._ov_items = list(items or [])
        self.draw_overlay()

    def draw_overlay(self):
        if self.abs_top is None or self.buf is None:
            self.overlay.hide()
            return
        top, fs = self.abs_top, self.fs_in
        hist = self.ROWS * self.dt
        items = [dict(it) for it in self._ov_items if (top - it["abs1"]) / fs < hist and it["abs0"] <= top]
        span = self.span
        for it in items:                                  # 확대 범위에 맞춰 가로 폭을 줄인다 (구간 상자는 대역 안쪽)
            it["f0"], it["f1"] = max(it["f0"], CENTER - span), min(it["f1"], CENTER + span)
        h = max(self.wf.getViewBox().height(), 1.0)
        self.overlay.render(items, lambda a: max(-self.dt / 2, (top - a) / fs), h / max(hist, 1e-6))

    def set_info(self, info):
        self.dt = info["dt_s"]
        self.fs_in = float(info.get("fs_in") or self.fs_in)
        self.lb_res.setText("{:.2f} Hz · {:.0f} ms".format(info["df_hz"], info["dt_s"] * 1e3))

    def set_tones(self, groups, key=None):
        """코스타스 톤 안내선: [(주파수 목록, 'dot' | 'dash')]. key 가 같으면 다시 만들지 않는다"""
        if key is not None and key == self._tone_key:
            return
        self._tone_key = key
        for p, ln in self._tone_lines:
            p.removeItem(ln)
        self._tone_lines = []
        for p in (self.spec, self.wf):
            for freqs, style in groups:
                pen = T.pen("text2", 1, 120, Qt.DotLine) if style == "dot" else T.pen("wf_b", 1, 110, Qt.DashLine)
                for f in freqs:
                    ln = pg.InfiniteLine(f, angle=90, pen=pen)
                    p.addItem(ln)
                    self._tone_lines.append((p, ln))

    def set_df(self, df):
        """코스타스 A 에서 검출된 주파수 오차 → 초록 선"""
        f = CENTER + float(df)
        for ln in self.df_lines:
            ln.setValue(f)
            ln.setVisible(True)
        self.df_text.setText("A {:+.1f} Hz".format(df))
        self.df_text.setPos(f + 3, self.hi - 2)

    # ---------------------------------------------------------------- 그리기 (화면 갱신 타이머에서, 최대 20fps)
    def push(self, rows):
        if not rows:
            return
        freqs = rows[-1][1] + (CENTER - 1500.0)             # 분석기는 1500 Hz 기준 (수신 흐름) → 선택 중심으로
        sel = np.abs(freqs - CENTER) <= self.span + 1e-9
        fsel = freqs[sel]
        if self.buf is None or self.freqs is None or len(self.freqs) != len(fsel) or \
                abs(self.freqs[0] - fsel[0]) > 1e-6:
            self.freqs = fsel
            self.buf = np.full((self.ROWS, len(fsel)), self.lo, dtype=np.float32)
            self._nvalid = 0
        new = np.stack([r[0][sel] for r in rows[-self.ROWS:]])[::-1]     # 최신이 위
        k = len(new)
        self._nvalid = min(self.ROWS, self._nvalid + k)
        self.buf = np.concatenate([new, self.buf[:-k]], axis=0) if k < self.ROWS else new[:self.ROWS]
        self.img.setImage(self.buf, autoLevels=False, levels=(self.lo, self.hi))
        self.img.setRect(grid_rect(self.freqs, np.arange(self.ROWS) * self.dt))
        self.wf.setYRange(-self.dt / 2, (self.ROWS - 0.5) * self.dt, padding=0)
        if len(rows[-1]) > 2:
            self.abs_top = rows[-1][2]
        self.draw_overlay()
        cur = np.mean(np.stack([r[0][sel] for r in rows[-3:]]), axis=0)
        self._last = cur
        self.curve.setData(self.freqs, cur)
        if self.df_lines[0].isVisible():
            self.df_text.setPos(self.df_lines[0].value() + 3, self.hi - 2)
