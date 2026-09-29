"""
NeuroMod 화면 부품 — 그리기만 한다. 계산은 analysis.py 스레드가 한다.
색·폰트는 theme.py 에서만 가져온다.
"""

import time
from collections import deque
from datetime import datetime, timezone

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import Qt, QRectF, Signal, QPointF, QTimer, QSize
from PySide6.QtGui import QPainter, QColor, QPen, QFont, QFontMetricsF, QBrush
from PySide6.QtWidgets import (QWidget, QHBoxLayout, QVBoxLayout, QGridLayout, QLabel, QToolButton,
                               QTableWidget, QTableWidgetItem, QHeaderView, QAbstractItemView,
                               QPushButton, QInputDialog, QMenu, QButtonGroup, QToolTip, QSizePolicy,
                               QStackedWidget, QGraphicsRectItem)

import theme as T
from scipy import signal as sg
from strings_ko import tr, trf

FMAX, CENTER, HALF_BW = 3000.0, 1500.0, 250.0


def fmt_clock(sec):
    """남은 · 경과 시간 표기: 00:12, 1시간 이상 1:02:05. 값이 없으면 —"""
    if sec is None or not np.isfinite(sec):
        return "—"
    s = int(round(max(float(sec), 0.0)))
    h, rem = divmod(s, 3600)
    m, s = divmod(rem, 60)
    return "{}:{:02d}:{:02d}".format(h, m, s) if h else "{:02d}:{:02d}".format(m, s)


class FitAxis(pg.AxisItem):
    """
    눈금 개수를 축 길이와 글자 크기에 맞춰 고르는 축. 숫자끼리 겹치지 않게 한 단계(큰 눈금)만 쓴다.
    세로축: 글자 높이의 2배마다 하나, 가로축: '-0000.0' 폭의 1.6배마다 하나.
    """

    def tickSpacing(self, minVal, maxVal, size):
        if self._tickSpacing is not None:
            return self._tickSpacing
        dif = abs(maxVal - minVal)
        if dif == 0 or size <= 0:
            return []
        fm = QFontMetricsF(self.style.get("tickFont") or T.mono_font(T.PT_SMALL))
        gap = fm.height() * 2.0 if self.orientation in ("left", "right") else fm.horizontalAdvance("-0000.0") * 1.6
        n = max(1, int(size / gap))
        raw = dif / n
        p10 = 10 ** np.floor(np.log10(raw))
        step = next(m * p10 for m in (1, 2, 2.5, 5, 10) if m * p10 >= raw * 0.999)
        return [(step, 0)]


def fit_axes(*orients):
    return {o: FitAxis(o) for o in orients}


def _plot(title=None, xl=None, yl=None):
    p = pg.PlotWidget(axisItems=fit_axes("left", "bottom"))
    T.style_plot(p, title, xl, yl)
    p.getViewBox().setMouseEnabled(x=False, y=False)
    return p


def _band_lines(plot, col="text3", a_center=150, a_edge=120):
    plot._band_lines = [pg.InfiniteLine(CENTER, angle=90, pen=T.pen(col, 1, a_center))]
    for f in (CENTER - HALF_BW, CENTER + HALF_BW):
        plot._band_lines.append(pg.InfiniteLine(f, angle=90, pen=T.pen(col, 1, a_edge, Qt.DashLine)))
    for ln in plot._band_lines:
        plot.addItem(ln)


def set_center(f, plots=()):
    """송수신 중심 주파수 바꾸기 (16부): 이 모듈의 CENTER (워터폴 표지 · 구간) 와 대역선"""
    global CENTER
    CENTER = float(f)
    for p in plots:
        for ln, v in zip(getattr(p, "_band_lines", []), (CENTER, CENTER - HALF_BW, CENTER + HALF_BW)):
            ln.setValue(v)


def hline():
    """1px 가로 구분선"""
    w = QWidget()
    w.setObjectName("hline")
    w.setAttribute(Qt.WA_StyledBackground, True)
    return w


def vline():
    w = QWidget()
    w.setObjectName("vline")
    w.setAttribute(Qt.WA_StyledBackground, True)
    return w


def segment(buttons):
    """평평한 세그먼트 버튼 묶음 (테두리 하나, 칸 사이 1px 선). buttons = QToolButton 목록"""
    box = QWidget()
    box.setObjectName("segment")
    box.setAttribute(Qt.WA_StyledBackground, True)
    h = QHBoxLayout(box)
    h.setContentsMargins(0, 0, 0, 0)
    h.setSpacing(0)
    for i, b in enumerate(buttons):
        b.setProperty("first", i == 0)
        b.setToolButtonStyle(Qt.ToolButtonTextOnly)
        h.addWidget(b)
    box.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
    return box


def mono_when_filled(edit):
    """입력칸: 글자가 있을 때만 고정폭 (안내 문구는 일반 글꼴). QSS 의 [empty] 속성으로 바꾼다."""
    def upd(txt):
        e = "true" if not txt else "false"
        if edit.property("empty") != e:
            edit.setProperty("empty", e)
            edit.style().unpolish(edit)
            edit.style().polish(edit)
    edit.textChanged.connect(upd)
    upd(edit.text())
    return edit


def short_device(name, n=18):
    """장치 이름 줄이기: '마이크(5- USB Audio CODEC )' → 'USB Audio CODEC'"""
    import re
    s = (name or "").strip()
    m = re.match(r"^([^()]+)\((.*)$", s)
    if m:
        pre, inner = m.group(1).strip(), m.group(2).rstrip(")").strip()
        # '마이크(…)' '스피커(…)' 처럼 앞이 한글 종류 이름이면 괄호 안, 아니면 앞 이름을 쓴다
        s = inner if (inner and not pre.isascii()) else pre
    s = re.sub(r"^\d+-\s*", "", s).strip()
    return s if len(s) <= n else s[:n - 1].rstrip() + "…"


class ElideLabel(QLabel):
    """넓이가 모자라면 끝을 … 로 줄이는 한 줄 라벨 (전체 문장은 툴팁)"""

    def __init__(self, text=""):
        super().__init__()
        self._full = ""
        self.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
        self.setMinimumWidth(40)
        self.setText(text)

    def setText(self, t):
        self._full = t or ""
        self.setToolTip(self._full)
        self._elide()

    def text(self):
        return self._full

    def sizeHint(self):
        return QSize(self.fontMetrics().horizontalAdvance(self._full) + 8, super().sizeHint().height())

    def minimumSizeHint(self):
        return QSize(self.minimumWidth(), super().minimumSizeHint().height())

    def resizeEvent(self, ev):
        super().resizeEvent(ev)
        self._elide()

    def _elide(self):
        fm = self.fontMetrics()
        QLabel.setText(self, fm.elidedText(self._full, Qt.ElideRight, max(10, self.width() - 8)))


def _lab(text="", name=None, font=None):
    l = QLabel(text)
    if name:
        l.setObjectName(name)
    if font is not None:
        l.setFont(font)
    return l


# ====================================================================== '?' 도움말 네모 (설정 창 · VFO)
class HelpButton(QToolButton):
    """'?' 네모: 올림 = 툴팁, 누름 = 고정 표시 (다시 누름 · 바깥 클릭 = 닫힘). 패널 제목줄 '?' 와 같은 동작 (18부)"""

    def __init__(self, text, parent=None):
        super().__init__(parent)
        self.setObjectName("helpBox")
        self.setText("?")
        self.setToolTip(text)
        self.help_text = text
        self._pop, self._pop_closed = None, 0.0
        self.clicked.connect(self.toggle)

    def toggle(self):
        from PySide6.QtWidgets import QFrame
        if self._pop is not None and self._pop.isVisible():
            self._pop.close()
            return
        if time.time() - self._pop_closed < 0.25:
            return
        pop = QFrame(self, Qt.Popup)
        pop.setObjectName("helpPop")
        pop.setAttribute(Qt.WA_StyledBackground, True)
        lv = QVBoxLayout(pop)
        lv.setContentsMargins(T.sp(3), T.sp(2), T.sp(3), T.sp(2))
        lb = QLabel(self.help_text)
        lb.setObjectName("helpText")
        lv.addWidget(lb)
        pop.adjustSize()
        g = self.mapToGlobal(self.rect().bottomRight())
        x = g.x() - pop.width()
        left = self.window().mapToGlobal(self.window().rect().topLeft()).x()
        if x < left:                                         # 창 왼쪽 밖이면 버튼 왼쪽에 맞춰 오른쪽으로 펼침
            x = self.mapToGlobal(self.rect().bottomLeft()).x()
        pop.move(x, g.y() + 2)

        def closed(ev, _orig=pop.closeEvent):
            self._pop_closed = time.time()
            _orig(ev)
        pop.closeEvent = closed
        self._pop = pop
        pop.show()


# ====================================================================== 도킹 패널 제목
class DockTitle(QWidget):
    """
    패널 제목줄: 이름 + 종류 표시(LIVE = 연속 실시간 / 패킷 = 복조될 때마다) + 마지막 갱신 경과 + 분리/닫기.
    마우스를 제목줄에서 끌면 패널이 움직인다 (QDockWidget 기본 동작).
    """

    def __init__(self, dock, title, kind=None, help_text=None):
        super().__init__()
        self.setObjectName("dockTitle")
        self.dock = dock
        self.setAttribute(Qt.WA_StyledBackground, True)
        self.setFixedHeight(T.TITLE_H)
        h = QHBoxLayout(self)
        h.setContentsMargins(T.sp(2), 0, T.sp(1), 0)
        h.setSpacing(T.sp(2))
        tl = _lab(title, "dockTitleText")
        # 19부: 이름 옆 장식 (실시간 초록 점 · '패킷' 배지) 없앰 — 갱신 방식은 이름 툴팁으로만
        h.addWidget(tl)
        self.badge, self.kind = None, kind
        self.age = ElideLabel("")                   # 폭이 모자라면 … + 툴팁 (글자 중간이 잘리지 않게)
        self.age.setObjectName("age")
        self.age.setMinimumWidth(24)
        self.age.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred)
        h.addWidget(self.age, 0)
        h.addStretch(1)
        self.btns = []
        self.help = None
        if help_text:                                # 패널 설명: 올림 = 툴팁, 누름 = 고정 표시 (다시 누름 · 바깥 클릭 = 닫힘)
            self.help = QToolButton()
            self.help.setObjectName("titleBtn")
            self.help.setText("?")
            self.help.setToolTip(help_text)
            self.help.clicked.connect(lambda: self._toggle_help(help_text))
            self.help.hide()
            h.addWidget(self.help)
            self.btns.append(self.help)
            self._pop, self._pop_closed = None, 0.0
        for sym, tip, fn in (("⧉", tr("w.detach_attach"), lambda: dock.setFloating(not dock.isFloating())),
                             ("✕", tr("w.close"), dock.close)):
            b = QToolButton()
            b.setObjectName("titleBtn")
            b.setText(sym)
            b.setToolTip(tip)
            b.clicked.connect(fn)
            b.hide()                                # 마우스를 올렸을 때만 보인다 (숨긴 동안은 자리도 차지하지 않는다)
            h.addWidget(b)
            self.btns.append(b)
        self._t = None

    # QDockWidget 은 제목줄 높이를 sizeHint 로 잡는다 (고정 높이만으로는 내용이 제목 아래쪽을 덮는다).
    # 레이아웃이 1초에 수백 번 묻기 때문에 계산 없이 고정값을 돌려준다 (넓이는 패널 내용이 정한다)
    _HINT = QSize(160, T.TITLE_H)
    _MIN = QSize(80, T.TITLE_H)

    def sizeHint(self):
        return self._HINT

    def minimumSizeHint(self):
        return self._MIN

    def truncated(self):
        """제목 · 배지 · 경과 글자 중 폭이 모자라 잘리거나 … 로 줄어든 것이 있으면 True"""
        labs = [l for l in self.findChildren(QLabel) if l.isVisible() and l is not self.age]
        if any(l.width() + 1 < l.sizeHint().width() for l in labs):
            return True
        return self.age.isVisible() and QLabel.text(self.age) != self.age.text()

    def _toggle_help(self, text):
        from PySide6.QtWidgets import QFrame
        if self._pop is not None and self._pop.isVisible():
            self._pop.close()
            return
        if time.time() - self._pop_closed < 0.25:          # 방금 바깥(= 이 버튼) 클릭으로 닫힌 경우: 다시 열지 않음
            return
        pop = QFrame(self, Qt.Popup)
        pop.setObjectName("helpPop")
        pop.setAttribute(Qt.WA_StyledBackground, True)
        lv = QVBoxLayout(pop)
        lv.setContentsMargins(T.sp(3), T.sp(2), T.sp(3), T.sp(2))
        lb = QLabel(text)
        lb.setObjectName("helpText")
        lv.addWidget(lb)
        pop.adjustSize()
        g = self.help.mapToGlobal(self.help.rect().bottomRight())
        pop.move(g.x() - pop.width(), g.y() + 2)
        pop.destroyed.connect(lambda *_: None)

        def closed(ev, _orig=pop.closeEvent):
            self._pop_closed = time.time()
            _orig(ev)
        pop.closeEvent = closed
        self._pop = pop
        pop.show()

    def enterEvent(self, ev):
        for b in self.btns:
            b.show()
        super().enterEvent(ev)

    def leaveEvent(self, ev):
        for b in self.btns:
            b.hide()
        super().leaveEvent(ev)

    def touch(self):
        self._t = time.time()
        self.refresh()

    def refresh(self):
        if self._t is None:
            self.age.setText(tr("main.waiting"))
            return
        d = time.time() - self._t
        self.age.setText(tr("w.updated_ago").format(fmt_clock(d)) if d >= 1.5 else tr("w.just_updated"))


# ====================================================================== 미터
class LEDMeter(QWidget):
    """세그먼트 LED 막대 + 피크 홀드. zones = [(이 값까지, 색 이름)]"""

    def __init__(self, title, lo, hi, unit, zones, n_seg=36, fmt="{:+.1f}", sub_line=False):
        super().__init__()
        self.title, self.lo, self.hi, self.unit, self.zones, self.n, self.fmt = title, lo, hi, unit, zones, n_seg, fmt
        self.value = None
        self.peak = None
        self.peak_t = 0.0
        self.sub = ""
        self.sub_line = sub_line                           # 회색 부연 줄 (23부: 기본 없음)
        self.setMinimumHeight(48 if sub_line else 32)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

    def set(self, v, sub=""):
        self.value = None if (v is None or not np.isfinite(v)) else float(v)
        self.sub = sub
        now = time.time()
        if self.value is not None:
            if self.peak is None or self.value >= self.peak or now - self.peak_t > 1.5:
                if self.peak is None or self.value >= self.peak:
                    self.peak, self.peak_t = self.value, now
                else:
                    self.peak = max(self.value, self.peak - (self.hi - self.lo) * 0.02)
        self.update()

    def _color(self, v):
        for upto, name in self.zones:
            if v <= upto:
                return name
        return self.zones[-1][1]

    def paintEvent(self, ev):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing, False)
        w, h = self.width(), self.height()
        p.setFont(T.ui_font(T.PT_BODY))
        p.setPen(T.qcolor("text2"))
        p.drawText(0, 0, w, 20, Qt.AlignLeft | Qt.AlignVCenter, self.title)
        p.setFont(T.mono_font(T.PT_VALUE, True))
        txt = "-" if self.value is None else (self.fmt.format(self.value) + " " + self.unit)
        p.setPen(T.qcolor("text") if self.value is not None else T.qcolor("text3"))
        p.drawText(0, 0, w, 20, Qt.AlignRight | Qt.AlignVCenter, txt)
        y0, sh = 21, 7
        gap = 1
        sw = (w - gap * (self.n - 1)) / self.n
        for i in range(self.n):
            v = self.lo + (i + 0.5) / self.n * (self.hi - self.lo)
            name = self._color(v)
            on = self.value is not None and v <= self.value
            pk = self.peak is not None and abs(v - self.peak) <= (self.hi - self.lo) / self.n / 2
            col = T.qcolor(name) if (on or pk) else T.qcolor("seg_off")
            p.fillRect(QRectF(i * (sw + gap), y0, sw, sh), col)
        if self.sub_line:
            p.setFont(T.ui_font(T.PT_SMALL))
            p.setPen(T.qcolor("text3"))
            p.drawText(0, y0 + sh + 2, w, 18, Qt.AlignLeft | Qt.AlignVCenter, self.sub)
        p.end()


class CenterMeter(QWidget):
    """가운데가 0 인 막대 (주파수 오차). 범위를 넘으면 끝에서 빨강."""

    def __init__(self, title, span, unit):
        super().__init__()
        self.title, self.span, self.unit = title, span, unit
        self.value, self.sub = None, ""
        self.setMinimumHeight(48)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

    def set(self, v, sub=""):
        self.value = None if (v is None or not np.isfinite(v)) else float(v)
        self.sub = sub
        self.update()

    def paintEvent(self, ev):
        p = QPainter(self)
        w = self.width()
        p.setFont(T.ui_font(T.PT_BODY))
        p.setPen(T.qcolor("text2"))
        p.drawText(0, 0, w, 20, Qt.AlignLeft | Qt.AlignVCenter, self.title)
        p.setFont(T.mono_font(T.PT_VALUE, True))
        p.setPen(T.qcolor("text") if self.value is not None else T.qcolor("text3"))
        p.drawText(0, 0, w, 20, Qt.AlignRight | Qt.AlignVCenter,
                   "-" if self.value is None else "{:+.1f} {}".format(self.value, self.unit))
        y0, sh = 21, 7
        p.fillRect(QRectF(0, y0, w, sh), T.qcolor("seg_off"))
        p.fillRect(QRectF(w / 2 - 0.5, y0 - 2, 1, sh + 4), T.qcolor("text3"))
        if self.value is not None:
            f = np.clip(self.value / self.span, -1, 1)
            x = w / 2 + f * w / 2
            name = "err" if abs(self.value) > self.span else "rx"
            a, b = sorted((w / 2, x))
            p.fillRect(QRectF(a, y0 + 2, max(b - a, 2), sh - 4), T.qcolor(name, 140))
            p.fillRect(QRectF(x - 1.5, y0 - 3, 3, sh + 6), T.qcolor(name))
        p.setFont(T.ui_font(T.PT_SMALL))
        p.setPen(T.qcolor("text3"))
        p.drawText(0, y0 + sh + 2, w, 18, Qt.AlignLeft | Qt.AlignVCenter, self.sub)
        p.drawText(0, y0 + sh + 2, w, 18, Qt.AlignRight | Qt.AlignVCenter,
                   "±{:.0f} {}".format(self.span, self.unit))
        p.end()


class MarginGauge(QWidget):
    """복원 여유 = 추정 SNR − 복조 실패 한계(-6.5dB). 3dB 미만 빨강, 3~6 노랑, 6 이상 초록."""
    LO, HI = -6.0, 30.0

    def __init__(self):
        super().__init__()
        self.margin = None
        self.sub = ""                                      # 23부: 부연 줄 없음
        self.setMinimumHeight(38)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

    @staticmethod
    def color(m):
        return "err" if m < 3 else ("fec" if m < 6 else "ok")

    def set(self, m, sub=""):
        self.margin = None if (m is None or not np.isfinite(m)) else float(m)
        self.sub = sub
        self.update()

    def paintEvent(self, ev):
        p = QPainter(self)
        w = self.width()
        p.setFont(T.ui_font(T.PT_BODY))
        p.setPen(T.qcolor("text2"))
        p.drawText(0, 0, w, 24, Qt.AlignLeft | Qt.AlignVCenter, tr("meter.margin"))
        p.setFont(T.mono_font(T.PT_BIG, True))
        name = "text3" if self.margin is None else self.color(self.margin)
        p.setPen(T.qcolor(name))
        p.drawText(0, 0, w, 24, Qt.AlignRight | Qt.AlignVCenter,
                   "-" if self.margin is None else "{:+.1f} dB".format(self.margin))
        y0, sh = 27, 8
        span = self.HI - self.LO
        for a, b, nm in ((self.LO, 3, "err"), (3, 6, "fec"), (6, self.HI, "ok")):
            x0, x1 = (a - self.LO) / span * w, (b - self.LO) / span * w
            p.fillRect(QRectF(x0, y0, x1 - x0 - 1, sh), T.qcolor(nm + "_dim"))
        if self.margin is not None:
            x = (np.clip(self.margin, self.LO, self.HI) - self.LO) / span * w
            p.fillRect(QRectF(0, y0, x, sh), T.qcolor(name, 170))
            p.fillRect(QRectF(x - 1.5, y0 - 3, 3, sh + 6), T.qcolor("text"))
        p.end()


class TXIndicator(QWidget):
    """
    송신 표시등 (송신 패널 맨 위 한 줄, 높이 고정이라 켜지고 꺼질 때 배치가 흔들리지 않는다).
    평소: 왼쪽에 작고 어두운 'TX' 칸 + '대기'. 송신 중: 줄 전체를 붉게 채운 큰 TX + 남은 시간 + 진행 막대.
    """
    H = 36

    def __init__(self):
        super().__init__()
        self.on, self.frac, self.left = False, 0.0, 0.0
        self.setFixedHeight(self.H)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

    def set(self, on, frac=0.0, left=0.0):
        self.on, self.frac, self.left = on, frac, left
        self.update()

    def paintEvent(self, ev):
        p = QPainter(self)
        w, h = self.width(), self.height()
        if not self.on:
            box = QRectF(0.5, (h - 20) / 2, 32, 20)
            p.setPen(QPen(T.qcolor("line2"), 1))
            p.setBrush(Qt.NoBrush)
            p.drawRect(box)
            p.setFont(T.mono_font(T.PT_SMALL, True))
            p.setPen(T.qcolor("text3"))
            p.drawText(box, Qt.AlignCenter, "TX")
            p.setFont(T.ui_font(T.PT_SMALL))
            p.drawText(QRectF(40, 0, w - 40, h), Qt.AlignLeft | Qt.AlignVCenter,
                       tr("w.done") if self.frac >= 1.0 else tr("w.idle"))
            p.end()
            return
        p.fillRect(QRectF(0, 0, w, h - 4), T.qcolor("err"))
        p.setPen(T.qcolor("accent_text"))
        p.setFont(T.mono_font(16, True))
        p.drawText(QRectF(T.sp(3), 0, 80, h - 4), Qt.AlignLeft | Qt.AlignVCenter, "TX")
        p.setFont(T.ui_font(T.PT_BODY, True))
        p.drawText(QRectF(64, 0, w - 64 - T.sp(3), h - 4), Qt.AlignLeft | Qt.AlignVCenter, tr("main.tx2"))
        p.setFont(T.mono_font(T.PT_BODY, True))
        p.drawText(QRectF(0, 0, w - T.sp(3), h - 4), Qt.AlignRight | Qt.AlignVCenter,
                   tr("w.remaining").format(fmt_clock(self.left)))
        p.fillRect(QRectF(0, h - 4, w, 4), T.qcolor("err_dim"))
        p.fillRect(QRectF(0, h - 4, w * min(max(self.frac, 0.0), 1.0), 4), T.qcolor("err"))
        p.end()


# ====================================================================== 요약 칸 (카드 상자 대신 구분선)
class StatGrid(QWidget):
    """이름 ····· 값 을 한 줄씩 (계측기 표시창처럼). 열 사이는 1px 세로선. 값은 고정폭, 오른쪽 정렬."""

    def __init__(self, keys, cols=2):
        super().__init__()
        g = QGridLayout(self)
        g.setContentsMargins(T.sp(2), T.sp(1), T.sp(2), T.sp(1))
        g.setHorizontalSpacing(T.sp(2))
        g.setVerticalSpacing(0)
        self.v = {}
        rows = (len(keys) + cols - 1) // cols
        for i, (key, label) in enumerate(keys):
            cc, r = divmod(i, rows)
            k = _lab(label, "statKey")
            k.setMinimumHeight(22)
            val = _lab("-", "statVal")
            val.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
            g.addWidget(k, r, cc * 3)
            g.addWidget(val, r, cc * 3 + 1)
            self.v[key] = val
        for cc in range(cols - 1):
            g.addWidget(vline(), 0, cc * 3 + 2, rows, 1)
        for cc in range(cols):
            g.setColumnStretch(cc * 3, 1)

    def set(self, key, text, color=None):
        lab = self.v[key]
        lab.setText(text)
        lab.setStyleSheet("" if color is None else "color:{};".format(T.c(color)))


# ====================================================================== 스펙트럼 · 워터폴
class DBRange(QWidget):
    """세로 dB 범위: 최저 · 최고 칸 + '자동' (누를 때 한 번만 맞추고 다시 고정). changed(lo, hi)"""
    changed = Signal(float, float)
    autoRequested = Signal()

    def __init__(self, lo, hi):
        super().__init__()
        from PySide6.QtWidgets import QSpinBox
        h = QHBoxLayout(self)
        h.setContentsMargins(0, 0, 0, 0)
        h.setSpacing(T.sp(1))
        h.addWidget(_lab("dBFS", "dim"))
        self.lo, self.hi = QSpinBox(), QSpinBox()
        for sb, v, tip in ((self.lo, lo, tr("w.min")), (self.hi, hi, tr("w.max"))):
            sb.setRange(-200, 20)
            sb.setSingleStep(5)
            sb.setValue(int(v))
            # 위아래 화살표 버튼 없음 (스타일이 없어 '·' 모양으로 보이고 숫자 칸을 가렸다). 휠 · ↑↓ 키로 5 dB 단위
            sb.setButtonSymbols(QSpinBox.NoButtons)
            sb.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
            sb.setFont(T.mono_font(T.PT_BODY))
            # 폭 = 부호 포함 4자리 "-200" + 안쪽 여백 좌우 8px (style.qss) + 테두리 2px + 여유 24px (배율 · 글꼴 차이)
            from PySide6.QtGui import QFontMetrics
            sb.setFixedWidth(QFontMetrics(T.mono_font(T.PT_BODY)).horizontalAdvance("-200") + 2 * T.sp(2) + 2 + 24)
            sb.setObjectName("dbfs")
            sb.valueChanged.connect(self._emit)
            h.addWidget(sb)
        b = QPushButton(tr("w.auto"))
        b.clicked.connect(self.autoRequested.emit)
        h.addWidget(b)

    def value(self):
        return float(self.lo.value()), float(self.hi.value())

    def set_value(self, lo, hi):
        for sb in (self.lo, self.hi):
            sb.blockSignals(True)
        self.lo.setValue(int(round(lo)))
        self.hi.setValue(int(round(hi)))
        for sb in (self.lo, self.hi):
            sb.blockSignals(False)
        self._emit()

    def _emit(self, *_):
        lo, hi = self.value()
        if hi <= lo + 5:
            hi = lo + 5
        self.changed.emit(lo, hi)


def waterfall_auto_range(db, span=60.0):
    """
    워터폴 '자동' (누를 때 한 번): 하한 = 잡음 중앙값 + 3dB (잡음 대부분이 검정),
    상한 = 하한 + 고정 폭 60dB (나중에 들어오는 신호도 포화 없이). 1dB 단위
    """
    db = np.asarray(db, dtype=np.float64)
    db = db[np.isfinite(db)]
    if db.size == 0:
        return None
    lo = float(np.round(np.median(db) + 3.0))
    return lo, lo + span


def auto_range(db):
    """'자동': 잡음 바닥(하위 10%) 아래 10dB ~ 최대값 위 5dB, 5dB 단위로"""
    db = np.asarray(db, dtype=np.float64)
    db = db[np.isfinite(db)]
    if db.size == 0:
        return T.SPEC_RANGE
    lo = np.floor((np.percentile(db, 10) - 10) / 5) * 5
    hi = np.ceil((np.max(db) + 5) / 5) * 5
    return float(lo), float(max(hi, lo + 20))


class SpectrumPanel(QWidget):
    """
    FFT 스펙트럼. 세로축은 고정 범위(기본 theme.SPEC_RANGE dBFS) — 사용자가 바꾸거나 '자동'으로 한 번 맞춘다.
    가로축은 워터폴과 연동(link)하고 왼쪽 축 폭을 theme.AXIS_W 로 고정해 픽셀 단위로 맞춘다.
    """
    rangeChanged = Signal(float, float)

    def __init__(self, rng=None):
        super().__init__()
        v = QVBoxLayout(self)
        v.setContentsMargins(T.sp(1), T.sp(1), T.sp(1), 0)
        v.setSpacing(0)
        self.plot = _plot(None, None, "dBFS")
        self.plot.getAxis("left").setWidth(T.AXIS_W)
        self.plot.setXRange(0, FMAX, padding=0)
        _band_lines(self.plot)
        self.peak = self.plot.plot(pen=T.pen("text3", 1, 255, Qt.DotLine))
        self.rx = self.plot.plot(pen=T.pen("rx"))
        self.tx = self.plot.plot(pen=T.pen("tx"))
        v.addWidget(self.plot, 1)
        row = QHBoxLayout()
        row.setContentsMargins(T.sp(1), 0, T.sp(1), 0)
        row.addWidget(_lab(tr("w.span_style_color_rx").format(
                               T.c("rx"), T.c("tx"), T.c("text3")), "note"))
        row.addStretch(1)
        row.setSpacing(T.sp(3))
        from PySide6.QtWidgets import QCheckBox
        self.chk_peak = QCheckBox(tr("w.peak_hold"))
        self.chk_peak.toggled.connect(self._reset)
        row.addWidget(self.chk_peak)
        self.range = DBRange(*(rng or T.SPEC_RANGE))
        self.range.changed.connect(self._set_range)
        self.range.autoRequested.connect(self._auto)
        row.addWidget(self.range)
        v.addLayout(row)
        self._peak, self._last = None, None
        self._set_range(*self.range.value())

    def _set_range(self, lo, hi):
        self.plot.setYRange(lo, hi, padding=0)
        self.rangeChanged.emit(lo, hi)

    def _auto(self):
        if self._last is not None:
            self.range.set_value(*auto_range(self._last))

    def link_x(self, other):
        """가로축을 다른 그래프(워터폴)와 묶는다"""
        self.plot.setXLink(other)

    def _reset(self, _=None):
        self._peak = None
        self.peak.setData([], [])

    def update_curves(self, f, rx_db=None, tx_db=None):
        if rx_db is not None:
            self._last = rx_db
            self.rx.setData(f, rx_db)
            if self.chk_peak.isChecked():
                self._peak = rx_db.copy() if self._peak is None else np.maximum(self._peak - 0.12, rx_db)
                self.peak.setData(f, self._peak)
        else:
            self.rx.setData([], [])
        self.tx.setData(f, tx_db) if tx_db is not None else self.tx.setData([], [])


class WaterfallWidget(pg.PlotWidget):
    freqClicked = Signal(float)                   # 16부: 클릭한 주파수 (중심 선택)
    """
    가로 = 주파수 0~3000Hz, 세로 = 시간. 실시간 모드: 위 = 최신 / 파일 모드: 위 = 파일 시작.
    코스타스 A/B 검출 위치를 사각형으로 표시한다 (워터폴과 함께 흘러간다).
    """
    COSTAS_S = 0.224

    def __init__(self, cols=480, rows=260, cmap="inferno"):
        super().__init__()
        T.style_plot(self, None, tr("w.frequency_hz"), tr("w.time_s"), grid=False)
        self.getPlotItem().layout.setContentsMargins(0, T.sp(2), 0, 0)   # 맨 위 눈금 숫자 '0' 이 잘리지 않게 (그림 · 범위는 그대로)
        self.getAxis("left").setWidth(T.AXIS_W)          # 스펙트럼과 같은 폭 → 1500Hz 선이 같은 픽셀
        self.cols, self.rows = cols, rows
        self.fgrid = np.linspace(0, FMAX, cols)
        self.live = False
        self.img = pg.ImageItem()
        self.addItem(self.img)
        self.set_cmap(cmap)
        vb = self.getViewBox()
        vb.invertY(True)
        vb.setMouseEnabled(x=False, y=False)
        self.setXRange(0, FMAX, padding=0)
        _band_lines(self, "wf_band", 110, 90)     # 워터폴 안 표시는 바꾸지 않는다
        self._markers = []
        self.recent = []
        self.file_spectrum = None
        self.abs_now = None
        self.overlay = SegOverlay(self)
        self._ov_items = []
        self._file = None
        self.scene().sigMouseClicked.connect(self._on_click)
        self._show(np.full((rows, cols), -120.0), rows * 0.032)

    def _on_click(self, ev):
        if ev.button() != Qt.LeftButton or not self.live:
            return
        vb = self.getViewBox()
        if not vb.sceneBoundingRect().contains(ev.scenePos()):
            return
        self.freqClicked.emit(float(vb.mapSceneToView(ev.scenePos()).x()))

    def set_cmap(self, name):
        self.cmap_name = name
        self.img.setLookupTable(T.cmap_lut(name))

    def _show(self, data, duration):
        self.img.setImage(data, autoLevels=False)
        hi = np.percentile(data, 99.5)
        self.img.setLevels((hi - 70, hi))
        self.img.setRect(QRectF(0, 0, FMAX, max(duration, 1e-3)))
        self.setYRange(0, max(duration, 1e-3), padding=0)

    def spectrum_rows(self, x, fs):
        nper = max(256, int(fs * 0.128))
        hop = max(1, nper // 4)
        if len(x) < nper:
            x = np.concatenate([x, np.zeros(nper - len(x))])
        f, _, S = sg.spectrogram(x, fs=fs, window="hann", nperseg=nper, noverlap=nper - hop, mode="psd")
        S = 10 * np.log10(np.maximum(S, 1e-16))
        return np.stack([np.interp(self.fgrid, f, S[:, j]) for j in range(S.shape[1])]), hop / fs

    @staticmethod
    def psd_to_dbfs(nper, fs):
        """
        이 워터폴이 쓰는 PSD [dB/Hz] (|X|²/(fs·Σw²), 한 창 hann nper) → 칸당 dBFS (풀스케일 사인 = 0dBFS) 로
        바꿀 때 더할 값: 10·log10(4·fs·Σw² / (Σw)²)
        """
        w = np.hanning(int(nper))
        return float(10 * np.log10(4.0 * fs * np.sum(w ** 2) / np.sum(w) ** 2))

    def dbfs_offset(self):
        if self.live:
            return self.psd_to_dbfs(self._nper, self.live_fs)
        return getattr(self, "_file_off", 0.0)

    def show_audio(self, x, fs):
        self.live = False
        # 파일 모드는 scipy spectrogram(한쪽 밀도, 직류 밖 칸을 2배) 이라 3dB 를 뺀다
        self._file_off = self.psd_to_dbfs(max(256, int(fs * 0.128)), fs) - 10 * np.log10(2.0)
        self.clear_markers()
        self.setLabel("left", tr("w.time_s"))
        rows, dt = self.spectrum_rows(np.asarray(x, dtype=np.float64), fs)
        self._show(rows, rows.shape[0] * dt)
        self._file = (float(fs), rows.shape[0] * dt, (len(x) / fs) / max(rows.shape[0] * dt, 1e-9))
        self._ov_items = []
        self.draw_overlay()
        self.file_spectrum = 10 * np.log10(np.mean(10 ** (rows / 10.0), axis=0) + 1e-20)

    SPEEDS = {"느림": 2.0, "보통": 1.0, "빠름": 0.5}      # 줄 사이 시간 배율 (보통 = 32 ms)
    speed = "보통"

    def set_speed(self, name):
        """흐름 속도: 줄 사이 시간 간격만 바꾼다 (FFT 길이 · 색 · 레벨은 그대로)"""
        self.speed = name if name in self.SPEEDS else "보통"
        if self.live:
            self.live_start(self.live_fs)

    def live_start(self, fs):
        self.live = True
        self.live_fs = float(fs)
        self._nper = max(256, int(fs * 0.128))
        self._hop = max(1, int(self._nper // 4 * self.SPEEDS[self.speed]))
        self._win = np.hanning(self._nper)
        self._fbins = np.fft.rfftfreq(self._nper, 1.0 / fs)
        self._buf = np.zeros(0)
        self._data = np.full((self.rows, self.cols), -150.0)
        self._lo = None
        self.abs_now = None
        self.recent = []
        self.clear_markers()
        self._ov_items = []
        self._file = None
        self.overlay.hide()
        self.setLabel("left", tr("w.elapsed_s"))
        self.img.setImage(self._data, autoLevels=False)
        self.img.setRect(QRectF(0, 0, FMAX, self.rows * self._hop / fs))
        self.setYRange(0, self.rows * self._hop / fs, padding=0)

    def push_samples(self, x, abs_end=None):
        if not self.live or len(x) == 0:
            return
        if abs_end is not None:
            self.abs_now = abs_end
        self._buf = np.concatenate([self._buf, x])
        new = []
        while len(self._buf) >= self._nper:
            seg = self._buf[:self._nper] * self._win
            pw = np.abs(np.fft.rfft(seg)) ** 2 / (self.live_fs * np.sum(self._win ** 2))
            new.append(np.interp(self.fgrid, self._fbins, 10 * np.log10(pw + 1e-16)))
            self._buf = self._buf[self._hop:]
        if not new:
            return
        self.recent = (self.recent + new)[-31:]
        rows = np.stack(new[::-1])
        n = min(len(rows), self.rows)
        self._data = np.concatenate([rows[:n], self._data[:-n]], axis=0)
        floor = float(np.median(rows))
        self._lo = floor if self._lo is None else 0.95 * self._lo + 0.05 * floor
        self.img.setImage(self._data, autoLevels=False)
        self.img.setLevels((self._lo - 5, self._lo + 65))
        self._update_markers()
        self.draw_overlay()

    def set_overlay(self, items):
        """패킷 구간 표시 항목 (SegOverlay 형식). 실시간: 입력 절대 번호, 파일: 파일 샘플 번호"""
        self._ov_items = list(items or [])
        self.draw_overlay()

    def draw_overlay(self):
        h = max(self.getViewBox().height(), 1.0)
        if self.live:
            if self.abs_now is None:
                return
            fs, hist = self.live_fs, self.rows * self._hop / self.live_fs
            now = self.abs_now
            items = [it for it in self._ov_items if (now - it["abs1"]) / fs < hist and it["abs0"] <= now]
            y_of = lambda a: max(0.0, (now - a) / fs)
            self.overlay.render(items, y_of, h / max(hist, 1e-6))
        elif self._file is not None:
            fs, dur, _ = self._file
            self.overlay.render(self._ov_items, lambda a: a / fs, h / max(dur, 1e-6))
        else:
            self.overlay.hide()

    def clear_markers(self):
        for m in self._markers:
            self.removeItem(m["box"])
            self.removeItem(m["txt"])
        self._markers = []

    def add_marker(self, m, fs):
        if not self.live:
            return
        col = "wf_tent" if m.get("tentative") else ("wf_a" if m["kind"] == "A" else "wf_b")    # 확정 전 = 자홍 점선 (세 모드 공통)
        old = next((x for x in self._markers if m.get("id") and x.get("id") == m["id"]), None)
        if old is not None:                                   # 같은 후보: 삭제 또는 임시 → 확정 (실선)
            if m.get("remove"):
                self.removeItem(old["box"])
                self.removeItem(old["txt"])
                self._markers.remove(old)
            else:
                old["box"].setPen(T.pen(col, 1.6) if not m.get("tentative") else T.pen(col, 1.2, style=Qt.DashLine))
                old["txt"].setColor(T.c(col))
                old["txt"].setText("{} {} {:+.0f}Hz{}".format(m.get("mode", "TNG44"), m["kind"], m["df"],
                                                             " ?" if m.get("tentative") else ""))
            return
        if m.get("remove"):                                   # id 없는 표시 (TNG44): 같은 위치 · 모드 · 종류
            for x in [x for x in self._markers if x.get("id") is None and x["abs"] == m["abs"] and
                      x.get("mode", "TNG44") == m.get("mode", "TNG44") and x.get("kind") == m["kind"]]:
                self.removeItem(x["box"])
                self.removeItem(x["txt"])
                self._markers.remove(x)
            return
        box = pg.PlotDataItem(pen=T.pen(col, 1.6) if not m.get("tentative") else T.pen(col, 1.2, style=Qt.DashLine))
        txt = pg.TextItem(trf(m.get("label") or "{} {} {:+.0f}Hz{}".format(m.get("mode", "TNG44"), m["kind"], m["df"],
                                                       " ?" if m.get("tentative") else "")), color=T.c(col), anchor=(0, 1))
        txt.setFont(T.mono_font(8))
        self.addItem(box)
        self.addItem(txt)
        self._markers.append({"abs": m["abs"], "fs": float(fs), "f": CENTER + m["df"], "box": box, "txt": txt,
                              "dur": m.get("dur", self.COSTAS_S), "half": m.get("half") or (125 if m.get("mode") == "TNG5" else 205),
                              "id": m.get("id"), "mode": m.get("mode", "TNG44"), "kind": m["kind"]})
        self._update_markers()

    def _update_markers(self):
        if self.abs_now is None or not self.live:
            return
        hist = self.rows * self._hop / self.live_fs
        keep = []
        for m in self._markers:
            age0 = (self.abs_now - m["abs"]) / m["fs"]
            age1 = age0 - m.get("dur", self.COSTAS_S)
            if age1 > hist:
                self.removeItem(m["box"])
                self.removeItem(m["txt"])
                continue
            f0, f1 = m["f"] - m.get("half", 205), m["f"] + m.get("half", 205)
            m["box"].setData([f0, f1, f1, f0, f0], [age1, age1, age0, age0, age1])
            m["txt"].setPos(f1 + 8, age1)
            keep.append(m)
        self._markers = keep


# ====================================================================== 워터폴 구간 표시
class SegOverlay:
    """
    워터폴 위 패킷 구간 상자 + 라벨 (톤 · 코스타스 · 가드 · 32바이트 구간 · 가드 · 코스타스).
    항목: {'abs0','abs1' (입력 절대 샘플 번호), 'kind', 'label', 'state', 'tip', 'f0','f1'}
      kind: tone / costas_a / costas_b / costas_m (TNG5 중간 동기) / costas_mx (중간 동기 확인 실패) / guard / seg / block
      state (seg · block): pending 대기 / ok CRC 통과 / fail 실패
      tent: True = 확정 전 (메시지 첫 CRC 전) → 상태와 관계없이 자홍 점선 (워터폴 표지와 같은 규칙, 세 모드 공통)
    라벨은 상자 높이가 LABEL_PX 보다 작으면 숨기고 마우스를 올리면 툴팁으로 보인다.
    """
    LABEL_PX = 13
    STYLE = {  # (테두리 색, 채움 색, 채움 알파, 점선)
        "tone": ("tx", "tx", 40, False), "costas_a": ("wf_a", None, 0, False), "costas_b": ("wf_b", None, 0, False),
        "costas_m": ("wf_a", "wf_a", 35, False), "costas_mx": ("err", None, 0, True),   # TNG5 v2 중간 동기 (확인 실패 = 점선)
        "guard": ("text3", None, 0, True), "pending": ("text3", "text3", 18, True),
        "ok": ("ok", "ok", 70, False), "fail": ("err", "err", 90, False),
        "tent": ("wf_tent", "wf_tent", 22, True),
    }

    def __init__(self, plot):
        self.plot = plot
        self.items = []
        self._pool = []
        self.visible = True
        plot.scene().sigMouseMoved.connect(self._hover)

    def _style(self, it):
        if it.get("tent"):
            return self.STYLE["tent"]
        return self.STYLE.get(it.get("state") if it["kind"] in ("seg", "block") else it["kind"],
                              self.STYLE["pending"])

    def render(self, items, y_of, px_per_s):
        """y_of(abs) → 그림 y 값 (초). px_per_s: 세로 1초당 화소 (라벨 생략 판단)"""
        self.items = items if self.visible else []
        while len(self._pool) < len(self.items):
            r = QGraphicsRectItem()
            r.setZValue(5)
            t = pg.TextItem("", anchor=(0, 0.5), fill=T.brush("bg0", 200))      # 어떤 컬러맵 위에서도 읽히게
            t.setFont(T.mono_font(8))
            t.setZValue(6)
            self.plot.addItem(r)
            self.plot.addItem(t)
            self._pool.append((r, t))
        for i, (r, t) in enumerate(self._pool):
            if i >= len(self.items):
                r.setVisible(False)
                t.setVisible(False)
                continue
            it = self.items[i]
            ya, yb = y_of(it["abs0"]), y_of(it["abs1"])
            y0, y1 = min(ya, yb), max(ya, yb)
            it["_rect"] = (it["f0"], y0, it["f1"] - it["f0"], y1 - y0)
            line, fill, alpha, dash = self._style(it)
            pen = pg.mkPen(T.c(line), width=1.3, style=Qt.DashLine if dash else Qt.SolidLine)
            pen.setCosmetic(True)
            r.setPen(pen)
            r.setBrush(T.brush(fill, alpha) if fill else QBrush(Qt.NoBrush))
            r.setRect(QRectF(*it["_rect"]))
            r.setVisible(True)
            show_lab = bool(it.get("label")) and (y1 - y0) * px_per_s >= self.LABEL_PX
            t.setVisible(show_lab)
            if show_lab:
                t.setText(trf(it["label"]))
                t.setColor(T.c(line))
                t.setPos(it["f1"] + 6, (y0 + y1) / 2)

    def hide(self):
        self.render([], lambda a: 0.0, 1.0)

    def _hover(self, pos):
        if not self.items:
            return
        vb = self.plot.getViewBox()
        if not vb.sceneBoundingRect().contains(pos):
            return
        p = vb.mapSceneToView(pos)
        for it in self.items:
            x, y, w, h = it.get("_rect", (0, 0, 0, 0))
            if x <= p.x() <= x + w and y <= p.y() <= y + h:
                vp = self.plot.mapFromScene(pos)
                QToolTip.showText(self.plot.mapToGlobal(vp.toPoint() if hasattr(vp, "toPoint") else vp),
                                  trf(it.get("tip") or it.get("label") or ""), self.plot)
                return


# ====================================================================== 패킷 타임라인
class TimelinePanel(pg.PlotWidget):
    """
    두 줄. 위 = 프레임 줄 (톤 → A → 가드 → F1..Fn → 가드 → B, 프레임 색 = 신호 품질)
          아래 = 구간 줄 (7단계: 32바이트 구간별 CRC16 통과 / 실패 / 대기, 옛 형식: 패킷 전체 CRC 1개)
    프레임 클릭 → frameClicked (분석 패널 프레임 선택) + 그 프레임에 비트가 실린 구간 강조
    구간 클릭 → 그 구간 비트가 실린 프레임들 강조 (인터리버로 약 ±3.2초에 흩어짐)
    송신 해부용: set_cursor(ms) 현재 위치 세로선, set_tx_states(프레임 상태, 구간 상태)
    """
    frameClicked = Signal(int)
    COL = {"ok": "ok", "fixed": "fec", "fail": "err", "guard": "line2", "tone": "tx_dim",
           "costas": "rx_dim", "missing": "bg2", "unk": "bg4", "pending": "bg3", "sending": "err",
           "sent": "text3", "wait": "bg3"}
    FY, SY, H = 0.52, 0.04, 0.42          # 프레임 줄 · 구간 줄 아래 끝, 높이

    def __init__(self, empty_text=tr("main.waiting")):
        super().__init__(axisItems=fit_axes("left", "bottom"))
        T.style_plot(self, None, tr("w.time_ms"), None, grid=False)
        self.getAxis("left").setTicks([[(self.FY + self.H / 2, tr("w.frame")), (self.SY + self.H / 2, tr("w.seg"))]])
        self.getAxis("left").setWidth(44)
        self.getViewBox().setMouseEnabled(x=False, y=False)
        self.setYRange(-0.02, 1.0, padding=0)
        self._items, self._segs, self._srow, self._fr = [], [], [], {}
        self._seg_frames = {}
        self._frame_bars = None
        self._seg_bars = None
        self.hl = None
        self.sel = None
        self.setMouseTracking(True)
        self.scene().sigMouseMoved.connect(self._hover)
        self.scene().sigMouseClicked.connect(self._click)
        self.empty = pg.TextItem(empty_text, color=T.c("text3"), anchor=(0.5, 0.5))
        self.empty.setFont(T.ui_font(T.PT_SMALL))
        self.addItem(self.empty)
        self.empty.setPos(1500, 0.5)
        self.setXRange(0, 3000, padding=0)
        self.cursor = pg.InfiniteLine(0, angle=90, pen=T.pen("text", 1.6))
        self.cursor.setVisible(False)
        self.addItem(self.cursor)
        self._labs, self._lab_items = [], {}
        self.getViewBox().sigResized.connect(self._fit_labels)          # 폭이 바뀌면 들어가는 번호 다시 판정

    def show_packet(self, tl, frame_err=None, frame_llr=None):
        """tl: 두 줄 dict {'frames','segments','seg_frames'} 또는 옛 프레임 목록"""
        for it in self._items:
            self.removeItem(it)
        self._items = []
        if isinstance(tl, dict):
            fr, sr = list(tl.get("frames") or []), list(tl.get("segments") or [])
            self._seg_frames = dict(tl.get("seg_frames") or {})
        else:
            fr, sr, self._seg_frames = list(tl or []), [], {}
        self._segs, self._srow = fr, sr
        self.empty.setVisible(not fr)
        self._fr = {"err": frame_err, "llr": frame_llr}
        self.sel = None
        self._frame_bars = self._seg_bars = None
        self._tx_key = None
        self._labs, self._lab_items = [], {}
        if not fr:
            return
        end = fr[-1][1] + fr[-1][2]
        self.setXRange(-10, end + 10, padding=0)
        self._frame_bars = self._bars(fr, self.FY)
        self._seg_bars = self._bars(sr, self.SY) if sr else None
        self.hl = pg.PlotDataItem(pen=T.pen("text", 1.6), connect="finite")
        self.addItem(self.hl)
        self._items.append(self.hl)
        self._fit_labels()

    def _bars(self, rows, y0):
        br = [T.brush(self.COL.get(s[3], "bg3")) for s in rows]
        bars = pg.BarGraphItem(x0=[s[1] for s in rows], width=[s[2] * 0.98 for s in rows],
                               y0=y0, height=self.H, brushes=br, pen=pg.mkPen(T.c("bg1")))
        self.addItem(bars)
        self._items.append(bars)
        for name, t, d, kind, lab in rows:                                  # 라벨은 칸에 들어갈 때만 (_fit_labels)
            if not lab:
                continue
            font = T.mono_font(T.PT_SMALL, True) if lab.isascii() else T.ui_font(T.PT_SMALL)
            wpx = QFontMetricsF(font).horizontalAdvance(lab) + 4               # 실제 글꼴로 잰 글자 폭
            self._labs.append((t + d / 2, y0 + self.H / 2, d, wpx, lab, font,
                               T.c("bg0") if kind in ("fixed", "ok", "sent") else T.c("text2")))
        return bars

    def _fit_labels(self, *_):
        """
        칸 폭(현재 화면 폭 기준 화소) ≥ 글자 폭이면 표시. 예전에는 패킷을 받을 때 한 번, 추정 폭(7 화소/글자 + 4)과
        그 순간의 위젯 폭으로만 판정해서 두 자리 번호(10~)가 칸이 넓어도 빠졌다.
        """
        if not self._labs:
            return
        vb = self.getViewBox()
        x0, x1 = vb.viewRange()[0]
        pxu = vb.width() / max(x1 - x0, 1e-9)                                  # 1 ms 당 화소
        for i, (x, y, d, wpx, lab, font, col) in enumerate(self._labs):
            ti = self._lab_items.get(i)
            fit = d * pxu >= wpx
            if fit and ti is None:
                ti = pg.TextItem(lab, color=col, anchor=(0.5, 0.5))
                ti.setFont(font)
                ti.setPos(x, y)
                self.addItem(ti)
                self._items.append(ti)
                self._lab_items[i] = ti
            elif ti is not None:
                ti.setVisible(fit)

    def set_cursor(self, ms):
        self.cursor.setVisible(ms is not None)
        if ms is not None:
            self.cursor.setValue(ms)

    def set_tx_states(self, frame_states, seg_states):
        """송신 진행 색: 데이터 프레임 · 구간마다 'sent' / 'sending' / 'wait'"""
        key = (id(self._frame_bars), id(self._seg_bars), tuple(frame_states or ()), tuple(seg_states or ()))
        if key == getattr(self, "_tx_key", None):             # 상태가 같으면 색도 같다
            return
        self._tx_key = key
        for bars, rows, states, pre in ((self._frame_bars, self._segs, frame_states, "F"),
                                        (self._seg_bars, self._srow, seg_states, "S")):
            if bars is None or states is None:
                continue
            br, j = [], 0
            for s in rows:
                if s[0].startswith(pre) and s[0][1:].isdigit():
                    k = states[j] if j < len(states) else "wait"
                    j += 1
                    br.append(T.brush(self.COL.get(k, "bg3")))
                else:
                    br.append(T.brush(self.COL.get(s[3], "bg3")))
            bars.setOpts(brushes=br)

    def _row_at(self, pos):
        if not self.sceneBoundingRect().contains(pos):
            return None, None
        p = self.getViewBox().mapSceneToView(pos)
        for rows, y0, key in ((self._segs, self.FY, "f"), (self._srow, self.SY, "s")):
            if y0 <= p.y() <= y0 + self.H:
                for i, s in enumerate(rows):
                    if s[1] <= p.x() <= s[1] + s[2]:
                        return key, i
        return None, None

    def _frame_idx(self, i):
        name = self._segs[i][0]
        return int(name[1:]) - 1 if name.startswith("F") and name[1:].isdigit() else None

    def _seg_idx(self, i):
        name = self._srow[i][0]
        return int(name[1:]) - 1 if name.startswith("S") and name[1:].isdigit() else None

    def _hover(self, pos):
        key, i = self._row_at(pos)
        if key is None:
            QToolTip.hideText()
            return
        if key == "s":
            name, t, d, kind, lab = self._srow[i]
            k = self._seg_idx(i)
            st = {"ok": tr("w.crc16_ok"), "fail": tr("w.crc16_fail"), "pending": tr("w.pending"), "sending": tr("main.tx2"),
                  "sent": tr("w.sent"), "wait": tr("qso.tx_idle")}.get(kind, kind)
            if k is None:
                txt = "{}  {}".format(lab or name, st)
            else:
                fr = self._seg_frames.get(k, [])
                txt = tr("w.seg_bytes_bits_in").format(
                    k + 1, 32 * k, 32 * k + 31, st, len(fr), (min(fr) + 1) if fr else "-",
                    (max(fr) + 1) if fr else "-")
        else:
            name, t, d, kind, lab = self._segs[i]
            fi = self._frame_idx(i)
            if fi is not None:
                e = self._fr.get("err")
                l = self._fr.get("llr")
                st = {"ok": tr("w.ok"), "fixed": tr("w.fec_fixed"), "fail": tr("w.fail"),
                      "unk": tr("w.unknown_fail_segment_bits")}.get(kind, kind)
                segs = [k + 1 for k, fr in self._seg_frames.items() if fi in fr]
                ev = None if e is None else e[fi]
                txt = tr("w.frame_fixed_bits_mean").format(
                    fi + 1, st, "-" if ev is None or (isinstance(ev, float) and not np.isfinite(ev)) else int(ev),
                    "-" if l is None else "{:.1f}".format(float(l[fi])),
                    tr("w.segments").format(", ".join(map(str, segs))) if segs else "")
            else:
                txt = {"tone": tr("w.lead_tone_300_ms"), "costas": tr("w.costas_array_224_ms"),
                       "guard": tr("w.guard_frame_tx_ramp"), "missing": tr("w.costas_b_not_found")}.get(kind, name)
        vp = self.mapFromScene(pos)                       # PySide6 는 QPoint 를 돌려준다 (버전에 따라 QPointF)
        QToolTip.showText(self.mapToGlobal(vp.toPoint() if hasattr(vp, "toPoint") else vp), trf(txt), self)

    def _outline(self, rows, idxs, y0):
        xs, ys = [], []
        for i in idxs:
            s = rows[i]
            xs += [s[1], s[1] + s[2], s[1] + s[2], s[1], s[1], np.nan]
            ys += [y0, y0, y0 + self.H, y0 + self.H, y0, np.nan]
        return xs, ys

    def select_segment(self, k):
        """구간 k 와 그 비트가 실린 프레임들 강조 (코드에서 부를 때)"""
        i = next((j for j in range(len(self._srow)) if self._seg_idx(j) == k), None)
        if i is None or self.hl is None:
            return
        fr = set(self._seg_frames.get(k, []))
        idx = [j for j in range(len(self._segs)) if self._frame_idx(j) in fr]
        x1, y1 = self._outline(self._segs, idx, self.FY)
        x2, y2 = self._outline(self._srow, [i], self.SY)
        self.hl.setData(x1 + x2, y1 + y2, connect="finite")

    def _click(self, ev):
        key, i = self._row_at(ev.scenePos())
        if key is None or self.hl is None:
            return
        if key == "s":
            k = self._seg_idx(i)
            if k is not None:
                self.select_segment(k)
            return
        fi = self._frame_idx(i)
        if fi is None:
            return
        self.sel = fi
        sk = [j for j in range(len(self._srow)) if fi in self._seg_frames.get(self._seg_idx(j), [])]
        x1, y1 = self._outline(self._segs, [i], self.FY)
        x2, y2 = self._outline(self._srow, sk, self.SY)
        self.hl.setData(x1 + x2, y1 + y2, connect="finite")
        self.frameClicked.emit(fi)


# ====================================================================== 추이 그래프
class TrendPanel(pg.GraphicsLayoutWidget):
    """최근 N 개 패킷: SNR, 주파수 오차, FEC 고친 비트(막대), hard 판정 BER, 성공/실패(점). 가로축 = UTC"""

    ROW_H = 50          # 한 줄 최소 높이: 9pt 세로 축 제목('FEC 비트')이 잘리지 않는 높이
    AXIS_W = 74

    def __init__(self, n=60):
        super().__init__()
        self.setBackground(T.c("plot"))
        self.n = n
        self.rows = deque(maxlen=n)
        self.p = {}
        spec = [("snr", "SNR dB"), ("df", "Δf Hz"), ("fec", tr("w.fec_bits")),
                ("ber", "BER %"), ("ok", tr("w.result"))]
        first = None
        for i, (k, lab) in enumerate(spec):
            ax = pg.DateAxisItem(orientation="bottom", utcOffset=0)
            pl = self.addPlot(row=i, col=0, axisItems={"bottom": ax, "left": FitAxis("left")})
            T.style_plot(pl, None, None, lab)
            pl.getAxis("left").setWidth(self.AXIS_W)      # 세로 축 제목 + 숫자가 잘리지 않는 폭 (모든 줄 같게)
            pl.setMinimumHeight(self.ROW_H + (10 if k == "ok" else 0))   # '성공' '실패' 두 글자 줄이 겹치지 않게
            pl.setMouseEnabled(x=False, y=False)
            if i < len(spec) - 1:
                pl.getAxis("bottom").setStyle(showValues=False)
            if first is None:
                first = pl
            else:
                pl.setXLink(first)
            self.p[k] = pl
        self.p["snr"].addItem(pg.InfiniteLine(-6.5, angle=0, pen=T.pen("err", 1, 160, Qt.DashLine)))
        self.c_snr = self.p["snr"].plot(pen=T.pen("rx"), symbol="o", symbolSize=4, symbolBrush=T.brush("rx"),
                                        symbolPen=None)
        self.c_df = self.p["df"].plot(pen=T.pen("rx"), symbol="o", symbolSize=4, symbolBrush=T.brush("rx"),
                                      symbolPen=None)
        self.b_fec = None
        self.c_ber = self.p["ber"].plot(pen=T.pen("fec"), symbol="o", symbolSize=4,
                                        symbolBrush=T.brush("fec"), symbolPen=None)
        self.s_ok = pg.ScatterPlotItem(size=6, pen=None)
        self.p["ok"].addItem(self.s_ok)
        self.p["ok"].setYRange(-0.3, 1.3, padding=0)
        self.p["ok"].getAxis("left").setTicks([[(0, tr("w.fail")), (0.5, tr("w.partial")), (1, tr("w.ok2"))]])

    def add(self, row):
        self.rows.append(row)
        self._redraw()

    def _redraw(self):
        if not self.rows:
            return
        t = np.array([r["t"] for r in self.rows], float)
        g = lambda k: np.array([np.nan if r.get(k) is None else r[k] for r in self.rows], float)
        self.c_snr.setData(t, g("snr"), connect="finite")
        self.c_df.setData(t, g("df"), connect="finite")
        self.c_ber.setData(t, 100 * g("ber"), connect="finite")
        if self.b_fec is not None:
            self.p["fec"].removeItem(self.b_fec)
        fec = np.nan_to_num(g("fec"))
        dt = np.diff(t)
        w = float(np.clip(np.min(dt) * 0.5, 0.3, 3.0)) if len(t) > 1 else 1.0
        self.b_fec = pg.BarGraphItem(x=t, height=fec, width=w, brush=T.brush("fec", 200), pen=None)
        self.p["fec"].addItem(self.b_fec)
        ok = g("ok")
        self.s_ok.setData(t, ok, brush=[T.brush("ok") if v > 0.75 else (T.brush("fec") if v > 0.25 else T.brush("err"))
                                        for v in ok])
        pad = max(5.0, (t[-1] - t[0]) * 0.05)
        self.p["snr"].setXRange(t[0] - pad, t[-1] + pad, padding=0)


# ====================================================================== 메시지 표
class MessageTable(QTableWidget):
    """
    UTC, SNR, 주파수 오차, 복원 여유, FEC 수정 비트, 메시지. 행을 고르면 packetSelected(id).
    방향은 따로 열을 두지 않고 메시지 글자색으로 (송신 = 황토, 수신 = 기본, 실패 = 벽돌).
    """
    packetSelected = Signal(object)
    COLS = ["UTC", "SNR", "Δf Hz", tr("w.margin"), "FEC", tr("w.message")]
    C_FEC = 4

    def __init__(self):
        super().__init__(0, len(self.COLS))
        self.setHorizontalHeaderLabels(self.COLS)
        self.verticalHeader().setVisible(False)
        self.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.setSelectionMode(QAbstractItemView.SingleSelection)
        self.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.setAlternatingRowColors(False)
        self.setShowGrid(False)
        self.setWordWrap(False)
        self.setFocusPolicy(Qt.NoFocus)
        h = self.horizontalHeader()
        h.setHighlightSections(False)
        h.setDefaultAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        h.setFixedHeight(24)
        for i, w in enumerate((80, 64, 64, 60, 48)):
            self.setColumnWidth(i, w)
            h.setSectionResizeMode(i, QHeaderView.Fixed)
            self.horizontalHeaderItem(i).setTextAlignment(Qt.AlignRight | Qt.AlignVCenter if i else
                                                          Qt.AlignLeft | Qt.AlignVCenter)
        h.setSectionResizeMode(len(self.COLS) - 1, QHeaderView.Stretch)
        self.verticalHeader().setDefaultSectionSize(24)
        self.itemSelectionChanged.connect(self._sel)
        self._mono = T.mono_font(T.PT_BODY)

    def _item(self, v, pid, num=False):
        it = QTableWidgetItem(v)
        it.setData(Qt.UserRole, pid)
        if num:
            it.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
            it.setFont(self._mono)
        return it

    def add(self, row):
        r = self.rowCount()
        self.insertRow(r)
        d = row.get("dir", "RX")
        pid = row.get("id")
        vals = [row.get("utc_hms", ""),
                "" if row.get("snr") is None else "{:+.1f}".format(row["snr"]),
                "" if row.get("df") is None else "{:+.1f}".format(row["df"]),
                "" if row.get("margin") is None else "{:+.1f}".format(row["margin"]),
                "" if row.get("fec") is None else str(row["fec"])]
        for cidx, v in enumerate(vals):
            it = self._item(v, pid, num=True)
            if cidx == 0:
                it.setTextAlignment(Qt.AlignLeft | Qt.AlignVCenter)
                it.setForeground(T.qcolor("text2"))
            if cidx == 3 and row.get("margin") is not None:
                it.setForeground(T.qcolor(MarginGauge.color(row["margin"])))
            self.setItem(r, cidx, it)
        it = self._item(row.get("text", ""), pid)
        it.setForeground(T.qcolor("tx" if d == "TX" else ("text" if row.get("ok") else "err")))
        it.setToolTip(row.get("text", ""))
        self.setItem(r, len(self.COLS) - 1, it)
        self.scrollToBottom()

    def set_fec(self, pid, v):
        """패킷 분석이 끝나면 그 행의 FEC 칸을 채운다"""
        for row in range(self.rowCount() - 1, -1, -1):
            it = self.item(row, 0)
            if it is not None and it.data(Qt.UserRole) == pid:
                c = self._item(str(v), pid, num=True)
                c.setForeground(T.qcolor("fec" if v else "text3"))
                self.setItem(row, self.C_FEC, c)
                break

    def _sel(self):
        items = self.selectedItems()
        if items:
            self.packetSelected.emit(items[0].data(Qt.UserRole))


# ====================================================================== 매크로
class MacroBar(QWidget):
    """매크로 버튼. 누르면 macroText, 오른쪽 클릭으로 편집. 저장은 부르는 쪽(settings)."""
    macroText = Signal(str)
    changed = Signal(list)

    def __init__(self, macros):
        super().__init__()
        self.macros = [dict(m) for m in macros]
        self.lay = QHBoxLayout(self)
        self.lay.setContentsMargins(0, 0, 0, 0)
        self.lay.setSpacing(T.sp(1))
        self._build()

    def _build(self):
        while self.lay.count():
            w = self.lay.takeAt(0).widget()
            if w:
                w.deleteLater()
        for i, m in enumerate(self.macros):
            b = QPushButton(m["name"])
            b.setObjectName("macro")
            b.setToolTip(m["text"])
            b.clicked.connect(lambda _=False, t=m["text"]: self.macroText.emit(t))
            b.setContextMenuPolicy(Qt.CustomContextMenu)
            b.customContextMenuRequested.connect(lambda _p, i=i: self._menu(i))
            self.lay.addWidget(b)
        add = QPushButton("+")
        add.setObjectName("macro")
        add.setToolTip(tr("w.add_macro"))
        add.clicked.connect(self._add)
        self.lay.addWidget(add)
        self.lay.addStretch(1)

    def _menu(self, i):
        menu = QMenu(self)
        menu.addAction(tr("w.edit"), lambda: self._edit(i))
        menu.addAction(tr("set.delete"), lambda: self._delete(i))
        menu.exec(self.cursor().pos())

    def _edit(self, i):
        m = self.macros[i]
        name, ok = QInputDialog.getText(self, tr("w.macro_name"), tr("set.name"), text=m["name"])
        if not ok:
            return
        text, ok = QInputDialog.getText(self, tr("w.macro_text"), tr("w.mycall_dxcall_snr_allowed"), text=m["text"])
        if ok:
            self.macros[i] = {"name": name or m["name"], "text": text}
            self._build()
            self.changed.emit(self.macros)

    def _delete(self, i):
        del self.macros[i]
        self._build()
        self.changed.emit(self.macros)

    def _add(self):
        self.macros.append({"name": tr("set.new_macro"), "text": "{DXCALL} DE {MYCALL}"})
        self._build()
        self._edit(len(self.macros) - 1)


# ====================================================================== 분석 패널들
class ViewToggle(QWidget):
    """'등화 / 원본' 같은 보기 전환 (세그먼트 버튼, 여러 패널이 같은 상태를 공유)"""
    changed = Signal(str)

    def __init__(self, options):
        super().__init__()
        h = QHBoxLayout(self)
        h.setContentsMargins(T.sp(1), T.sp(1), T.sp(1), T.sp(1))
        h.setSpacing(0)
        self.g = QButtonGroup(self)
        self.g.setExclusive(True)
        self.btn = {}
        bl = []
        for key, lab, tip in options:
            b = QToolButton()
            b.setText(lab)
            b.setCheckable(True)
            self.g.addButton(b)
            bl.append(b)
            self.btn[key] = b
            b.clicked.connect(lambda _=False, k=key: self.changed.emit(k))
        options and self.btn[options[0][0]].setChecked(True)
        h.addWidget(segment(bl))
        h.addStretch(1)

    def set(self, key):
        self.btn[key].setChecked(True)


class ConstellationPanel(QWidget):
    """
    IQ 성상도.
      · 신호가 들어오는 동안: 코스타스 A 로 주파수·위상을 맞춘 심볼 중앙 샘플을 실시간으로 흘린다 (흐린 시안)
      · 복조되면: 데이터 보조 등화 결과(초록) + 이상 송신점(앰버 흐림) / '원본' 이면 등화 전 (시안)
      · 프레임을 고르면 그 프레임 점만 진하게
    """

    def __init__(self):
        super().__init__()
        v = QVBoxLayout(self)
        v.setContentsMargins(T.sp(1), T.sp(1), T.sp(1), T.sp(1))
        v.setSpacing(T.sp(1))
        self.plot = _plot(None, "I", "Q")
        self.plot.setAspectLocked(True)
        self.plot.setXRange(-2.4, 2.4)
        self.plot.setYRange(-2.4, 2.4)
        self.s_live = pg.ScatterPlotItem(size=3, pen=None, brush=T.brush("rx", 70))
        self.s_ideal = pg.ScatterPlotItem(size=3, pen=None, brush=T.brush("tx", 70))
        self.s_main = pg.ScatterPlotItem(size=3.5, pen=None)
        self.s_sel = pg.ScatterPlotItem(size=7, pen=T.pen("text", 1), brush=None)
        self.s_tx = pg.ScatterPlotItem(size=3, pen=None, brush=T.brush("tx", 130))
        for s in (self.s_live, self.s_ideal, self.s_tx, self.s_main, self.s_sel):
            self.plot.addItem(s)
        v.addWidget(self.plot, 1)
        self.info = _lab("", "note")
        self.info.setWordWrap(True)
        v.addWidget(self.info)
        self.live = deque(maxlen=900)
        self.pkt = None
        self.view = "eq"
        self.frame = None

    def push_live(self, pts):
        self.live.extend(pts)
        a = np.array(self.live)
        self.s_live.setData(a.real, a.imag)

    def clear_live(self):
        self.live.clear()
        self.s_live.setData([], [])

    def set_tx(self, pts):
        self.s_tx.setData(pts.real, pts.imag) if pts is not None else self.s_tx.setData([], [])

    def set_packet(self, out):
        self.pkt = out
        self.frame = None
        self._draw()

    def set_view(self, view):
        self.view = view
        self._draw()

    def set_frame(self, fi):
        self.frame = fi
        self._draw()

    def _draw(self):
        o = self.pkt
        if o is None:
            self.info.setText(tr("main.waiting"))
            return
        eq = o.get("const_eq") if self.view == "eq" else None
        p = eq if eq is not None else o.get("const_raw")
        if p is None:
            self.s_main.setData([], [])
            self.info.setText(tr("w.no_constellation_data").format(o.get("reason") or "-"))
            return
        col = "ok" if eq is not None else "rx"
        fr = o.get("const_frame")
        self.s_main.setData(p.real, p.imag, brush=T.brush(col, 150 if self.frame is None else 45))
        ide = o.get("const_ideal") if eq is not None else None
        self.s_ideal.setData(ide.real, ide.imag) if ide is not None else self.s_ideal.setData([], [])
        if self.frame is not None and fr is not None:
            m = fr[:len(p)] == self.frame
            self.s_sel.setData(p.real[m], p.imag[m], brush=T.brush(col))
        else:
            self.s_sel.setData([], [])
        parts = []
        if eq is not None:
            parts.append(tr("w.evm_1f_db_before").format(o["evm_eq"], o["evm_raw"]))
            parts.append(tr("w.residual_drift_2f_hz").format(o["drift_hz"]))
        elif self.view == "eq" and o.get("const_eq") is None:
            parts.append(tr("w.no_eq_decode_failed"))
        if not o.get("phase_ref", True):
            parts.append(tr("w.no_phase_reference_old"))
        if self.frame is not None:
            parts.append(tr("w.frame_highlighted").format(self.frame + 1))
        self.info.setText(" · ".join(parts))


class EyePanel(QWidget):
    """아이 다이어그램 I, Q (심볼 2개). 등화/원본 보기, 프레임 강조, 송신 겹치기(앰버)."""

    def __init__(self):
        super().__init__()
        v = QVBoxLayout(self)
        v.setContentsMargins(T.sp(1), T.sp(1), T.sp(1), T.sp(1))
        v.setSpacing(T.sp(1))
        self.pi = _plot(None, None, None)
        self.pq = _plot(None, tr("w.symbol"), None)
        self.c = {}
        for p, k in ((self.pi, "i"), (self.pq, "q")):
            # 세로 축 제목 'I' 를 90도 돌리면 가로줄처럼 보여서, 그래프 왼쪽 위에 똑바로 쓴다
            lab = pg.TextItem(k.upper(), color=T.c("text2"), anchor=(0, 0))
            lab.setFont(T.mono_font(T.PT_BODY, True))
            lab.setParentItem(p.getPlotItem().getViewBox())
            lab.setPos(T.sp(1), 0)
            p.setYRange(-2.6, 2.6)
            self.c[k + "tx"] = p.plot(pen=T.pen("tx", 1, 45), connect="finite")
            self.c[k + "rx"] = p.plot(pen=T.pen("rx", 1, 60), connect="finite")
            self.c[k + "sel"] = p.plot(pen=T.pen("text", 1.2, 200), connect="finite")
            v.addWidget(p, 1)
        self.pkt, self.view, self.frame = None, "eq", None

    @staticmethod
    def _tr(x, Y):
        if Y is None or len(Y) == 0:
            return [], []
        xs = np.tile(np.append(x, np.nan), len(Y))
        ys = np.concatenate([Y, np.full((len(Y), 1), np.nan)], axis=1).ravel()
        return xs, ys

    def set_tx(self, e):
        if e is None:
            for k in ("itx", "qtx"):
                self.c[k].setData([], [])
            return
        x, I, Q, _ = e
        self.c["itx"].setData(*self._tr(x, I))
        self.c["qtx"].setData(*self._tr(x, Q))

    def set_packet(self, o):
        self.pkt, self.frame = o, None
        self._draw()

    def set_view(self, v):
        self.view = v
        self._draw()

    def set_frame(self, fi):
        self.frame = fi
        self._draw()

    def _draw(self):
        o = self.pkt
        if o is None:
            return
        e = o.get("eye") if (self.view == "eq" and o.get("eye") is not None) else o.get("eye_raw")
        if e is None:
            return
        x, I, Q, fr = e
        col = "ok" if (self.view == "eq" and o.get("eye") is not None) else "rx"
        a = 60 if self.frame is None else 25
        for k, Y in (("i", I), ("q", Q)):
            self.c[k + "rx"].setPen(T.pen(col, 1, a))
            self.c[k + "rx"].setData(*self._tr(x, Y))
            if self.frame is not None:
                self.c[k + "sel"].setData(*self._tr(x, Y[fr == self.frame]))
            else:
                self.c[k + "sel"].setData([], [])


def peak_subcell(z, i, j):
    """2차원 격자 최대점 (i, j) 의 칸 안 위치를 포물선 맞춤으로 (-0.5 ~ +0.5 칸) 구한다"""
    def one(a, b, c):
        d = a - 2 * b + c
        return 0.0 if d >= 0 else float(np.clip(0.5 * (a - c) / d, -0.5, 0.5))
    di = one(z[i - 1, j], z[i, j], z[i + 1, j]) if 0 < i < z.shape[0] - 1 else 0.0
    dj = one(z[i, j - 1], z[i, j], z[i, j + 1]) if 0 < j < z.shape[1] - 1 else 0.0
    return di, dj


def grid_rect(x, y):
    """
    격자 값의 좌표 배열 x(열) · y(행) → ImageItem 의 사각형.
    픽셀 k 의 가운데가 x[k] 에 오도록 양끝을 반 칸씩 넓힌다
    (x[0]~x[-1] 로 두면 N 칸이 N-1 간격에 늘어나 픽셀 가운데가 최대 반 칸 어긋난다).
    """
    dx = (x[-1] - x[0]) / max(len(x) - 1, 1)
    dy = (y[-1] - y[0]) / max(len(y) - 1, 1)
    return QRectF(x[0] - dx / 2, y[0] - dy / 2, dx * len(x), dy * len(y))


class SyncMapPanel(QWidget):
    """
    코스타스 A 동기 맵. 값 = 검출기와 같은 정규화 상관 ρ² (같은 전처리 · 같은 정규화, analysis.LiveAnalyzer).

    데이터용 열지도라 워터폴처럼 '차분한 색' 규칙의 예외 (인지적으로 균일하고 대비가 강한 컬러맵, 고를 수 있음).
      · 값 10·log10(ρ²) [dB], 표시 범위는 **절대값으로 고정** LEVEL_DB = -10 ~ -1 dB (ρ² 0.10 ~ 0.79):
        잡음(10분 최대 0.116)은 거의 검정, 문턱 0.18(-7.4dB) 근처부터 밝아진다. 신호가 없으면 거의 어두운 화면
      · 문턱 이상 영역 = 초록 윤곽선, ○ = 원래 ρ² 최대점 (칸 안 위치까지)
      · '부드럽게': 봉우리가 한 칸이라 점 하나로 보이는 것을 표시용으로만 넓힌다 (숫자 · ○ · 윤곽선은 원래 값)
      · 실시간 / 검출 스냅샷: A 가 검출되면 그 위치 둘레 맵(검출기와 같은 2ms 격자)을 고정해 보여 준다.
        다음 검출 때 바뀌고, '해제'로 지운다. 스냅샷의 + = 검출기가 보고한 위치 (○ 와 겹쳐야 한다)
    """
    cmapChanged = Signal(str)
    smoothChanged = Signal(bool)
    LEVEL_DB = (-10.0, -1.0)

    def __init__(self, threshold, cmap="magma", smooth=True):
        super().__init__()
        from PySide6.QtWidgets import QCheckBox, QComboBox
        v = QVBoxLayout(self)
        v.setContentsMargins(T.sp(1), T.sp(1), T.sp(1), T.sp(1))
        v.setSpacing(T.sp(1))
        # 1줄: 부드럽게 + 수치 (그래프를 가리지 않게 밖에)
        r1 = QHBoxLayout()
        r1.setContentsMargins(T.sp(1), 0, 0, 0)
        r1.setSpacing(T.sp(2))
        self.chk_smooth = QCheckBox(tr("w.smooth"))
        self.chk_smooth.setChecked(bool(smooth))
        self.chk_smooth.toggled.connect(self._smooth_toggled)
        r1.addWidget(self.chk_smooth)
        self.num = ElideLabel(tr("w.waiting_for_input"))
        self.num.setObjectName("dim")                 # 한글 섞인 문장이라 일반 글꼴
        self.num.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
        r1.addWidget(self.num, 1)
        v.addLayout(r1)
        self._r1 = r1
        # 2줄: 실시간 / 검출 스냅샷 · 해제 · 컬러맵
        r2 = QHBoxLayout()
        r2.setContentsMargins(T.sp(1), 0, 0, 0)
        r2.setSpacing(T.sp(2))
        self.b_live, self.b_snap = QToolButton(), QToolButton()
        self.g_mode = QButtonGroup(self)
        for bt, lab, tip in ((self.b_live, tr("w.live"), tr("w.last_1_5_s")),
                             (self.b_snap, tr("w.snapshot"), tr("w.detection_snapshot_map_at"))):
            bt.setText(lab)
            bt.setCheckable(True)
            self.g_mode.addButton(bt)
        self.b_live.setChecked(True)
        self.b_snap.setEnabled(False)
        self.b_live.clicked.connect(lambda: self.set_mode("live"))
        self.b_snap.clicked.connect(lambda: self.set_mode("snap"))
        r2.addWidget(segment([self.b_live, self.b_snap]))
        self.b_clear = QPushButton(tr("w.release"))
        self.b_clear.setEnabled(False)
        self.b_clear.clicked.connect(self.clear_snapshot)
        r2.addWidget(self.b_clear)
        r2.addStretch(1)
        v.addLayout(r2)
        # 컬러맵은 1줄 끝에 (2줄이 넓어져 패널 최소 폭이 410px 까지 커지면 '전체' 배치가 창 폭 1614px 을 요구했다)
        self.cb_cmap = QComboBox()
        self.cb_cmap.addItems(T.WATERFALL_CMAPS)
        self.cb_cmap.setCurrentText(cmap if cmap in T.WATERFALL_CMAPS else "magma")
        self.cb_cmap.currentTextChanged.connect(self._cmap_changed)
        self._r1.addWidget(self.cb_cmap)

        self.plot = _plot(None, tr("w.time_s_0_now"), tr("w.freq_error_hz"))
        self.plot.setMinimumHeight(150)
        self.img = pg.ImageItem()
        self.img.setLookupTable(T.cmap_lut(self.cb_cmap.currentText()))
        self.plot.addItem(self.img)
        self.iso = pg.IsocurveItem(level=threshold, pen=T.pen("ok", 1.4))      # 문턱 윤곽선 = 동기 색
        self.iso.setParentItem(self.img)            # 이미지 픽셀 좌표 그대로 (칸 가운데 기준)
        self.pk = pg.ScatterPlotItem(size=14, pen=T.pen("text", 1.5), brush=None)
        self.plot.addItem(self.pk)
        self.detm = pg.ScatterPlotItem(size=10, symbol="+", pen=T.pen("text", 1.2), brush=None)
        self.plot.addItem(self.detm)
        for ax in ("bottom", "left"):
            self.plot.getAxis(ax).enableAutoSIPrefix(False)
        self.plot.setXRange(-1.5, 0.0, padding=0)
        self.plot.setYRange(-100, 100, padding=0)
        v.addWidget(self.plot, 1)
        self.thr = threshold
        self.mode = "live"
        self._live, self._snap = None, None
        self.last = None                             # 마지막으로 그린 값 (검사 · 로그용)

    # ---------------------------------------------------------------- 조작
    def _cmap_changed(self, name):
        self.img.setLookupTable(T.cmap_lut(name))
        self.cmapChanged.emit(name)

    def _smooth_toggled(self, on):
        self.smoothChanged.emit(bool(on))
        self._redraw()

    def set_mode(self, mode):
        if mode == "snap" and self._snap is None:
            mode = "live"
        self.mode = mode
        (self.b_snap if mode == "snap" else self.b_live).setChecked(True)
        self.plot.getPlotItem().setLabel("bottom", T.axis_label(
            tr("w.time_s_0_detected") if mode == "snap" else tr("w.time_s_0_now")))
        self._redraw()

    def clear_snapshot(self):
        self._snap = None
        self.b_snap.setEnabled(False)
        self.b_clear.setEnabled(False)
        self.set_mode("live")

    def set_live(self, s):
        self._live = s
        if self.mode == "live":
            self._draw(s, None)

    def set_snapshot(self, sn):
        """검출 스냅샷 (analysis.LiveAnalyzer._snapshot 결과) — 받으면 스냅샷 보기로 바뀐다"""
        if sn is None or "rho" not in sn:
            return
        self._snap = sn
        self.b_snap.setEnabled(True)
        self.b_clear.setEnabled(True)
        self.set_mode("snap")

    def _redraw(self):
        if self.mode == "snap" and self._snap is not None:
            self._draw(self._snap, self._snap)
        elif self._live is not None:
            self._draw(self._live, None)

    # ---------------------------------------------------------------- 그리기
    def _draw(self, s, snap):
        from scipy import ndimage
        rho, f, t = s["rho"], s["f"], s["t"]         # rho: (주파수 행, 시간 열)
        if rho.size == 0 or rho.ndim != 2 or rho.shape[0] < 3 or rho.shape[1] < 3:
            return
        rho = np.asarray(rho, dtype=np.float64)
        shown = rho
        if self.chk_smooth.isChecked():
            shown = ndimage.gaussian_filter(ndimage.maximum_filter(rho, size=(3, 7)), sigma=(0.7, 1.5))
        db = 10 * np.log10(np.maximum(shown, 1e-5))
        lo, hi = self.LEVEL_DB
        self.img.setImage(db, autoLevels=False, levels=(lo, hi))
        self.img.setRect(grid_rect(t, f))
        dt = (t[-1] - t[0]) / (len(t) - 1)
        dfh = (f[-1] - f[0]) / (len(f) - 1)
        self.plot.setXRange(t[0] - dt / 2, t[-1] + dt / 2, padding=0)
        self.plot.setYRange(f[0] - dfh / 2, f[-1] + dfh / 2, padding=0)

        i, j = np.unravel_index(np.argmax(rho), rho.shape)   # 원래 값 최대점 (행 = 주파수, 열 = 시간)
        mx = float(rho[i, j])
        di, dj = peak_subcell(rho, i, j)
        tp, fp = t[j] + dj * dt, f[i] + di * dfh
        self.last = {"i": int(i), "j": int(j), "t": float(tp), "f": float(fp), "rho": mx, "mode": self.mode,
                     "t_grid": float(t[j]), "f_grid": float(f[i]), "rect": grid_rect(t, f), "levels": (lo, hi),
                     "median": float(np.median(rho))}
        if mx >= self.thr:
            self.pk.setData([tp], [fp])
            self._contour(rho)
        else:
            self.pk.setData([], [])
            self.iso.setData(None)
        if snap is not None:
            d = snap["det"]
            self.detm.setData([0.0], [d["df"]])
            ck = snap.get("check", {})
            self.num.setText(tr("w.snapshot_det_2f_map").format(
                time.strftime("%H:%M:%S", time.localtime(snap.get("utc", time.time()))),
                d["rho"], mx, tp, fp) + snap.get("note", ""))
            self.num.setToolTip((snap.get("note_tip", "") + "\n") * bool(snap.get("note_tip")) +tr("w.detector_3f_fine_grid").format(
                                    d["rho"], ck.get("refine_rho", float("nan")), dfh, mx,
                                    ck.get("map_at_det", float("nan"))))
        else:
            self.detm.setData([], [])
            if mx >= self.thr:
                self.num.setText(tr("w.2f_above_threshold_3f").format(mx, tp, fp))
            else:
                self.num.setText(tr("w.max_2f_threshold_2f").format(mx, self.thr))

    def _contour(self, rho):
        """문턱 윤곽선: 문턱 넘는 칸을 감싸는 작은 창에서만 계산 (전체 격자는 Python 반복이 무겁다)"""
        m = rho >= self.thr * 0.7
        rows, cols = np.nonzero(m)
        if len(rows) == 0 or len(rows) > 4000:
            self.iso.setData(None)
            return
        r0, r1 = max(rows.min() - 2, 0), min(rows.max() + 3, rho.shape[0])
        c0, c1 = max(cols.min() - 2, 0), min(cols.max() + 3, rho.shape[1])
        self.iso.setLevel(self.thr)
        self.iso.setData(rho[r0:r1, c0:c1])
        self.iso.setPos(c0, r0)                     # 이미지 픽셀 좌표: x = 열(시간), y = 행(주파수)


class LLRPanel(QWidget):
    def __init__(self):
        super().__init__()
        v = QVBoxLayout(self)
        v.setContentsMargins(T.sp(1), T.sp(1), T.sp(1), T.sp(1))
        self.plot = _plot(None, "LLR", tr("w.bits"))
        self.plot.setMinimumHeight(140)
        v.addWidget(self.plot, 1)
        self.info = _lab("", "note")
        self.info.setWordWrap(True)
        v.addWidget(self.info)
        self._items = []

    def set_packet(self, h):
        for it in self._items:
            self.plot.removeItem(it)
        self._items = []
        if h is None:
            return
        e = h["edges"]
        ctr, w = 0.5 * (e[:-1] + e[1:]), e[1] - e[0]
        if "all" in h:
            items = [("all", "text2", w * 0.9, 0.0)]
            self.info.setText(tr("w.all"))
        else:
            items = [("zero", "cat1", w * 0.45, -w * 0.22), ("one", "cat0", w * 0.45, w * 0.22)]
            self.info.setText(tr("w.hard_errors_2").format(
                h["wrong"], h["n"], h["wrong"] / max(h["n"], 1)))
        for key, col, width, sh in items:
            b = pg.BarGraphItem(x=ctr + sh, height=h[key], width=width, brush=T.brush(col, 220), pen=None)
            self.plot.addItem(b)
            self._items.append(b)
        top = max(1, max(int(np.max(h[k])) for k, *_ in items))
        self.plot.setYRange(0, top * 1.1, padding=0)
        self.plot.setXRange(e[0], e[-1], padding=0)


class FeaturePanel(QWidget):
    """AI 파형에 맞는 대안 시각화: 수신 NN 특징(PCA) / 한 프레임의 위상 궤적"""

    def __init__(self):
        super().__init__()
        v = QVBoxLayout(self)
        v.setContentsMargins(T.sp(1), T.sp(1), T.sp(1), T.sp(1))
        v.setSpacing(T.sp(1))
        self.toggle = ViewToggle([("feat", tr("w.nn_features"), tr("w.2d_pca_of_rx")),
                                  ("traj", tr("w.phase_trace"), tr("w.eq_i_q_trace"))])
        self.toggle.changed.connect(self._mode)
        v.addWidget(self.toggle)
        self.stack = QStackedWidget()
        self.pf = _plot(None, "PC1", "PC2")
        self.pf.setAspectLocked(True)
        self.sf = [pg.ScatterPlotItem(size=4, pen=None, brush=T.brush("cat{}".format(k), 170)) for k in range(4)]
        self.sfu = pg.ScatterPlotItem(size=4, pen=None, brush=T.brush("text2", 150))
        self.sfs = pg.ScatterPlotItem(size=7, pen=T.pen("text", 1), brush=None)
        for s in self.sf + [self.sfu, self.sfs]:
            self.pf.addItem(s)
        self.pt = _plot(None, "I", "Q")
        self.pt.setAspectLocked(True)
        self.ct = self.pt.plot(pen=T.pen("ok", 1.2, 200))
        self.st = pg.ScatterPlotItem(size=6, pen=None, brush=T.brush("text"))
        self.pt.addItem(self.st)
        self.stack.addWidget(self.pf)
        self.stack.addWidget(self.pt)
        v.addWidget(self.stack, 1)
        self.info = _lab("", "note")
        self.info.setWordWrap(True)
        v.addWidget(self.info)
        self.pkt, self.frame = None, None

    def _mode(self, k):
        self.stack.setCurrentIndex(0 if k == "feat" else 1)
        self._draw()

    def set_packet(self, o):
        self.pkt, self.frame = o, None
        self._draw()

    def set_frame(self, fi):
        self.frame = fi
        self._draw()

    def _draw(self):
        o = self.pkt
        if o is None:
            return
        if self.stack.currentIndex() == 0:
            f = o.get("feat")
            if f is None:
                self.info.setText(tr("w.no_feature_data"))
                return
            xy, cls = f["xy"], f["cls"]
            if cls is not None:
                for k in range(4):
                    m = cls == k
                    self.sf[k].setData(xy[m, 0], xy[m, 1])
                self.sfu.setData([], [])
                self.info.setText(tr("w.cluster_accuracy_0").format(f["sep"]))
            else:
                for s in self.sf:
                    s.setData([], [])
                self.sfu.setData(xy[:, 0], xy[:, 1])
                self.info.setText(tr("w.decode_failed"))
            fr = o.get("feat_frame")
            if self.frame is not None and fr is not None:
                m = fr == self.frame
                self.sfs.setData(xy[m, 0], xy[m, 1])
            else:
                self.sfs.setData([], [])
        else:
            traj = o.get("traj")
            if not traj:
                self.ct.setData([], [])
                self.st.setData([], [])
                self.info.setText(tr("w.decode_failed"))
                return
            fi = 0 if self.frame is None else min(self.frame, len(traj) - 1)
            y, c = traj[fi]
            self.ct.setData(y.real, y.imag)
            self.st.setData(c.real, c.imag)
            self.info.setText(tr("w.frame_eq_i_q").format(fi + 1))
