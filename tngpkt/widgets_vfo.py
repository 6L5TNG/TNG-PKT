"""
VFO 패널 (18부) — 다이얼 주파수 · 밴드 · 무전기 모드 · 오디오 중심 · 실제 RF · 송수신 상태 · PTT · CAT 상태.

주파수 표시 000.000.000 (MHz.kHz.Hz): 자릿수에 마우스를 올리고 휠 = 그 자리 올리기/내리기, 클릭 = 직접 입력,
키보드 (포커스 뒤) ↑↓ = 올린 자리, ←→ = 자리 옮기기, 숫자 = 직접 입력 시작.
"""
import time

from PySide6.QtCore import Qt, Signal, QRectF
from PySide6.QtGui import QPainter, QFontMetricsF
from PySide6.QtWidgets import (QWidget, QGridLayout, QHBoxLayout, QLabel, QComboBox, QPushButton, QLineEdit,
                               QSizePolicy, QAbstractSpinBox)

from tngpkt import theme as T
from tngpkt import rig
from tngpkt.strings_ko import tr

NDIG = 9                     # 999.999.999 Hz 까지 (단파 · VHF · UHF)


class FreqDisplay(QWidget):
    freqChanged = Signal(int)            # 사용자가 바꾼 값 (Hz)

    def __init__(self, pt=22.0, editable=True):
        super().__init__()
        self.hz = 0
        self.editable = editable
        self.font_ = T.mono_font(pt, bold=True)
        self._hover = None               # 올린 자리 (0 = 맨 왼쪽 100 MHz 자리)
        self._rects = []
        self._ed = None
        self.hold_until = 0.0            # 사용자 변경 직후 CAT 읽은 값으로 되돌리지 않음
        self.valid = True                # False = CAT 없음 → 빈 표시 (--.---.---)
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.StrongFocus if editable else Qt.NoFocus)
        self.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)
        fm = QFontMetricsF(self.font_)
        self._cw = fm.horizontalAdvance("0")
        self._dw = fm.horizontalAdvance(".")
        self.setFixedHeight(int(fm.height() + T.sp(2)))
        self.setMinimumWidth(int(self._cw * NDIG + self._dw * 2 + T.sp(4)))

    def sizeHint(self):
        return self.minimumSize()

    # ---------------------------------------------------------------- 값
    def set_freq(self, hz, from_user=False):
        hz = int(max(0, min(10 ** NDIG - 1, int(hz or 0))))
        if not from_user and (time.time() < self.hold_until or self._ed is not None):
            return
        if hz == self.hz:
            return
        self.hz = hz
        self.update()
        if from_user:
            self.hold_until = time.time() + 1.5
            self.freqChanged.emit(hz)

    def _text(self):
        return rig.fmt_hz(self.hz).rjust(NDIG + 2)

    # ---------------------------------------------------------------- 그리기
    def paintEvent(self, ev):
        p = QPainter(self)
        p.setFont(self.font_)
        fm = QFontMetricsF(self.font_)
        s = "{:0{}d}".format(self.hz, NDIG)
        lead = next((i for i, ch in enumerate(s) if ch != "0"), NDIG - 1)
        lead = min(lead, NDIG - 7)                         # MHz 1의 자리부터는 항상 밝게
        if not self.valid:
            s, lead = " " + "-" * (NDIG - 1), NDIG
        x = (self.width() - (self._cw * NDIG + self._dw * 2)) / 2.0
        y = (self.height() + fm.ascent() - fm.descent()) / 2.0
        self._rects = []
        for i, ch in enumerate(s):
            r = QRectF(x, 0, self._cw, self.height())
            self._rects.append(r)
            if i == self._hover and self.editable and self.valid:
                p.fillRect(r.adjusted(0, 2, 0, -2), T.qcolor("bg3"))
                p.fillRect(QRectF(r.left(), r.bottom() - 3, r.width(), 2), T.qcolor("text2"))
            p.setPen(T.qcolor("text3" if i < lead else "text"))
            p.drawText(QRectF(x, 0, self._cw, self.height()), Qt.AlignHCenter | Qt.AlignVCenter, ch)
            x += self._cw
            if i in (2, 5):
                p.setPen(T.qcolor("text3"))
                p.drawText(QRectF(x, 0, self._dw, self.height()), Qt.AlignHCenter | Qt.AlignVCenter, ".")
                x += self._dw
        p.end()

    def _digit_at(self, pos):
        for i, r in enumerate(self._rects):
            if r.left() <= pos.x() < r.right():
                return i
        return None

    # ---------------------------------------------------------------- 마우스 · 키보드
    def mouseMoveEvent(self, ev):
        d = self._digit_at(ev.position())
        if d != self._hover:
            self._hover = d
            self.update()

    def leaveEvent(self, ev):
        if not self.hasFocus():
            self._hover = None
            self.update()
        super().leaveEvent(ev)

    def wheelEvent(self, ev):
        if not self.editable:
            return
        d = self._digit_at(ev.position())
        if d is None:
            return
        self._hover = d
        steps = 1 if ev.angleDelta().y() > 0 else -1 if ev.angleDelta().y() < 0 else 0
        if steps:
            self.set_freq(self.hz + steps * 10 ** (NDIG - 1 - d), from_user=True)
        ev.accept()

    def mousePressEvent(self, ev):
        if self.editable and ev.button() == Qt.LeftButton:
            self.setFocus()
            self.start_edit()

    def keyPressEvent(self, ev):
        if not self.editable:
            return super().keyPressEvent(ev)
        k = ev.key()
        if self._hover is None:
            self._hover = NDIG - 4                           # kHz 1의 자리
        if k in (Qt.Key_Up, Qt.Key_Down):
            self.set_freq(self.hz + (1 if k == Qt.Key_Up else -1) * 10 ** (NDIG - 1 - self._hover), from_user=True)
        elif k in (Qt.Key_Left, Qt.Key_Right):
            self._hover = max(0, min(NDIG - 1, self._hover + (-1 if k == Qt.Key_Left else 1)))
            self.update()
        elif k in (Qt.Key_Return, Qt.Key_Enter):
            self.start_edit()
        elif ev.text() and (ev.text().isdigit() or ev.text() == "."):
            self.start_edit(ev.text())
        else:
            return super().keyPressEvent(ev)

    def focusOutEvent(self, ev):
        self._hover = None
        self.update()
        super().focusOutEvent(ev)

    def start_edit(self, first=None):
        """직접 입력: 14.074 (MHz) · 14.074.000 · 14074000 (Hz) · 7074k. Enter = 적용, Esc = 취소"""
        if self._ed is not None:
            return
        ed = QLineEdit(self)
        ed.setObjectName("freqEdit")
        ed.setFont(self.font_)
        ed.setAlignment(Qt.AlignCenter)
        ed.setGeometry(self.rect())
        ed.setText(first if first is not None else "{:.6f}".format(self.hz / 1e6).rstrip("0").rstrip(".") or "0")
        if first is None:
            ed.selectAll()
        self._ed = ed

        def done(apply):
            if self._ed is None:
                return
            e, self._ed = self._ed, None
            if apply:
                v = rig.parse_hz(e.text())
                if v is not None:
                    self.set_freq(v, from_user=True)
            e.deleteLater()
            self.update()
        ed.returnPressed.connect(lambda: done(True))
        ed.editingFinished.connect(lambda: done(True))

        def key(ev, _orig=ed.keyPressEvent):
            if ev.key() == Qt.Key_Escape:
                done(False)
                return
            _orig(ev)
        ed.keyPressEvent = key
        ed.show()
        ed.setFocus()
        if first is not None:
            ed.setCursorPosition(len(first))


class VfoPanel(QWidget):
    """다이얼 (CAT 연결 때만) · 밴드 · 중심 (sp_center 는 앱이 만들어 넘김) · RF · 상태 · PTT · CAT.
    19부: 무전기 모드 선택 없음 (설정 Radio 의 Mode 로만), CAT 없으면 주파수 · RF 는 빈 표시 · 밴드 끔"""
    dialChanged = Signal(int)
    pttPressed = Signal(bool)

    def __init__(self, sp_center, bands, dial_hz=None, rig_mode=None):
        super().__init__()
        self.bands = [list(b) for b in (bands or rig.DEFAULT_BANDS)]
        # 20부: 두 줄 — [다이얼 · 밴드 · (연결되지 않음)] / [중심 · RF · PTT]
        g = QGridLayout(self)
        g.setContentsMargins(T.sp(3), T.sp(1), T.sp(3), T.sp(2))
        g.setHorizontalSpacing(T.sp(3))
        g.setVerticalSpacing(T.sp(2))
        self.freq = FreqDisplay(22.0)
        self.freq.freqChanged.connect(self._on_dial)
        self.cb_band = QComboBox()
        self._fill_bands()
        self.cb_band.activated.connect(self._on_band)
        self.cb_band.setFixedWidth(96)
        self.lb_cat = QLabel("")
        self.lb_cat.setObjectName("vfoCat")
        r1 = QHBoxLayout()
        r1.setSpacing(T.sp(3))
        r1.addWidget(self.freq)
        r1.addWidget(self.cb_band)
        r1.addStretch(1)
        r1.addWidget(self.lb_cat)
        g.addLayout(r1, 0, 0)

        def lab(t):
            l_ = QLabel(t)
            l_.setObjectName("dim")
            return l_
        self.sp_center = sp_center
        sp_center.setButtonSymbols(QAbstractSpinBox.NoButtons)       # 휠 · 입력만 (잘린 위아래 버튼 없음)
        sp_center.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        sp_center.setFixedWidth(96)
        self.lb_rf = QLabel("")
        self.lb_rf.setObjectName("vfoRf")
        self.lb_rf.setMinimumWidth(96)
        self.btn_ptt = QPushButton(tr("vfo.ptt"))
        self.btn_ptt.setObjectName("pttButton")
        self.btn_ptt.setProperty("on", False)
        self.btn_ptt.pressed.connect(lambda: self.pttPressed.emit(True))
        self.btn_ptt.released.connect(lambda: self.pttPressed.emit(False))
        r2 = QHBoxLayout()
        r2.setSpacing(T.sp(2))
        r2.addWidget(lab(tr("vfo.center")))
        r2.addWidget(sp_center)
        r2.addSpacing(T.sp(3))
        r2.addWidget(lab(tr("vfo.rf")))
        r2.addWidget(self.lb_rf)
        r2.addStretch(1)
        r2.addWidget(self.btn_ptt)
        g.addLayout(r2, 1, 0)
        g.setRowStretch(2, 1)
        self.connected = None
        self.set_connected(False)

    # ---------------------------------------------------------------- CAT 연결 여부
    def set_connected(self, on):
        on = bool(on)
        if on == self.connected:
            return
        self.connected = on
        self.freq.valid = on
        self.freq.editable = on
        self.freq.setFocusPolicy(Qt.StrongFocus if on else Qt.NoFocus)
        self.freq.update()
        self.cb_band.setEnabled(on)
        if not on:
            self.cb_band.setCurrentIndex(self.cb_band.count() - 1)
            self.lb_rf.setText("--.---.---")
        else:
            self._sync_band()

    # ---------------------------------------------------------------- 밴드
    def set_bands(self, bands):
        self.bands = [list(b) for b in bands]
        self._fill_bands()
        self._sync_band()

    def _fill_bands(self):
        self.cb_band.blockSignals(True)
        self.cb_band.clear()
        for n, hz in self.bands:
            self.cb_band.addItem(n, int(hz))
        self.cb_band.addItem("", None)                     # 목록 밖 주파수 · CAT 없음
        self.cb_band.blockSignals(False)

    def _sync_band(self):
        b = rig.band_of(self.freq.hz) if getattr(self, "connected", False) else ""
        i = self.cb_band.findText(b) if b else -1
        self.cb_band.blockSignals(True)
        self.cb_band.setCurrentIndex(i if i >= 0 else self.cb_band.count() - 1)
        self.cb_band.blockSignals(False)

    def _on_band(self, i):
        hz = self.cb_band.itemData(i)
        if hz and self.connected:
            self.freq.set_freq(int(hz), from_user=True)

    def _on_dial(self, hz):
        self._sync_band()
        self.dialChanged.emit(int(hz))

    # ---------------------------------------------------------------- 표시 (앱이 부름)
    def show_dial(self, hz):
        if hz is not None:
            self.freq.set_freq(hz)
            self._sync_band()

    def show_mode(self, m):
        """무전기 모드는 표시 안 함 (19부, 설정 Radio 의 Mode 로만)"""

    def show_rf(self, hz):
        self.lb_rf.setText(rig.fmt_hz(hz) if (hz is not None and self.connected) else "--.---.---")

    def show_tx(self, on, ptt=False):
        """송신 · PTT 중이면 PTT 버튼을 눌린 모양으로 (글자 표시 없음, 20부)"""
        on = bool(on or ptt)
        if self.btn_ptt.property("on") != on:
            self.btn_ptt.setProperty("on", on)
            self.btn_ptt.style().unpolish(self.btn_ptt)
            self.btn_ptt.style().polish(self.btn_ptt)

    def show_cat(self, text, state):
        """연결됐으면 비움, 아니면 '연결되지 않음' (차분한 글자색, 20부)"""
        t = "" if state == "ok" else tr("vfo.not_connected")
        if self.lb_cat.text() != t:
            self.lb_cat.setText(t)
