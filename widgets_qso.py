"""
8단계 운용 화면 부품 — 수신 · 송신 · 송신 해부 · 수신 로그 패널.
그리기만 한다. 프로토콜 구조 계산은 stream_ui.py, 색 · 글꼴은 theme.py.
"""

from datetime import datetime

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import Qt, Signal, QRectF, QSize
from PySide6.QtGui import QTextCharFormat, QTextCursor, QColor, QShortcut, QKeySequence
from PySide6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QGridLayout, QLabel, QPushButton,
                               QPlainTextEdit, QTextEdit, QProgressBar, QTableWidget, QTableWidgetItem,
                               QHeaderView, QAbstractItemView, QApplication, QToolButton, QSizePolicy,
                               QScrollArea, QFrame)

import theme as T
from widgets import TimelinePanel, _lab, segment, fmt_clock, fit_axes, ElideLabel
from strings_ko import tr, trf

SEG_BYTES = 32


def _u16(s):
    """파이썬 글자 위치 → QTextDocument 위치 (UTF-16 단위)"""
    return len(s.encode("utf-16-le")) // 2


# ====================================================================== 수신
class StatusRow(QWidget):
    """
    머리줄 상태 항목 줄. 글자를 자르지 않는다 — 폭이 모자라면 우선순위 낮은 항목부터 통째로 숨긴다.
    items = [(라벨, 우선순위)] 표시 순서대로, 우선순위 숫자가 작을수록 끝까지 남는다.
    툴팁 = 숨긴 항목 포함 전체. 폭은 버튼을 밀지 않게 0 까지 줄어든다
    """

    def __init__(self, items, spacing):
        super().__init__()
        self.items = items
        self.sp = spacing
        h = QHBoxLayout(self)
        h.setContentsMargins(0, 0, 0, 0)
        h.setSpacing(spacing)
        for lb, _ in items:
            h.addWidget(lb)
        h.addStretch(1)
        self.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
        self.setMinimumWidth(0)

    def minimumSizeHint(self):
        return QSize(0, super().minimumSizeHint().height())

    def refit(self):
        avail, used, show = self.width(), 0, set()
        for lb, pr in sorted(self.items, key=lambda x: x[1]):
            w = lb.sizeHint().width() + (self.sp if show else 0)
            if used + w > avail:
                break                                   # 이 항목이 안 들어가면 더 낮은 항목도 숨김 (우선순위 순서 유지)
            show.add(id(lb))
            used += w
        for lb, _ in self.items:
            lb.setVisible(id(lb) in show)
        self.setToolTip("  ·  ".join(lb.text() for lb, _ in self.items if lb.text()))

    def resizeEvent(self, ev):
        super().resizeEvent(ev)
        self.refit()


class QsoRxPanel(QWidget):
    """
    받은 송신을 지우지 않고 아래로 쌓는다. 새 송신(코스타스 A 검출)마다 구분선
    '====== YYYY-MM-DD HH:MM ======' (현지 시각) 뒤에 본문, 메시지 사이 빈 줄 1개.
    수신 중인 메시지는 맨 아래에서 32바이트 구간이 확정되는 대로 글자를 이어 붙인다
    (실패 구간 □ 벽돌색, 끝의 '▍' = 다음 구간 대기). 부분 복원 · 신호 소실 · 송신 중지는 구분선 끝에 표시.
    문서 끝(현재 메시지 본문)만 고쳐 쓰므로 쌓여도 전체를 다시 그리지 않는다.
    위로 스크롤하면 따라가기가 멈추고, '따라가기' 또는 맨 아래로 내리면 다시 따라간다.
    """
    # 글자 크기 (pt): 값은 self.size 한 곳, 가− / 가+ 는 1 pt 씩 대칭, 범위 안에서 자유롭게 오감
    SIZE_MIN, SIZE_MAX, SIZE_DEFAULT = 10, 40, 18

    def __init__(self, size=18):
        super().__init__()
        v = QVBoxLayout(self)
        v.setContentsMargins(T.sp(2), T.sp(1), T.sp(2), T.sp(2))
        v.setSpacing(T.sp(1))
        head = QHBoxLayout()
        head.setSpacing(T.sp(3))
        self.lb_state = _lab(tr("main.waiting"), "dim")
        self.lb_snr = _lab("SNR -", "num")
        self.lb_df = _lab("Δf -", "num")
        self.lb_seg = _lab(tr("main.seg"), "num")
        # 폭이 모자라면 Δf → SNR → 구간 → 상태 순으로 통째로 숨김 (글자 잘림 없음), 전체는 툴팁
        self.status = StatusRow([(self.lb_state, 1), (self.lb_snr, 3), (self.lb_df, 4), (self.lb_seg, 2)], T.sp(3))
        head.addWidget(self.status, 1)
        self.btn_follow = QToolButton()
        self.btn_follow.setText(tr("qso.follow"))
        self.btn_follow.setCheckable(True)
        self.btn_follow.setChecked(True)
        self.btn_small, self.btn_reset, self.btn_big = QToolButton(), QToolButton(), QToolButton()
        self.btn_small.setText(tr("qso.a"))
        self.btn_reset.setText(tr("qso.a2"))
        self.btn_big.setText(tr("qso.a3"))
        self.btn_small.setToolTip(tr("qso.smaller"))
        self.btn_reset.setToolTip(tr("qso.default_size"))
        self.btn_big.setToolTip(tr("qso.larger"))
        self.btn_copy = QToolButton()
        self.btn_copy.setText(tr("qso.copy"))
        self.btn_stop = QToolButton()
        self.btn_stop.setText(tr("main.stop"))
        self.btn_stop.setEnabled(False)
        self.btn_clear = QToolButton()
        self.btn_clear.setText(tr("qso.clear"))
        seg_box = segment([self.btn_stop, self.btn_follow, self.btn_small, self.btn_reset, self.btn_big,
                           self.btn_copy, self.btn_clear])
        seg_box.setStyleSheet("QToolButton {{ padding: 0 {}px; }}".format(T.sp(2)))    # 글자에 맞는 폭 (항상 전부 보임)
        head.addWidget(seg_box)
        v.addLayout(head)
        self.view = QTextEdit()
        self.view.setObjectName("qsoRx")
        self.view.setReadOnly(True)
        self.view.setTextInteractionFlags(Qt.TextSelectableByMouse | Qt.TextSelectableByKeyboard)
        v.addWidget(self.view, 1)
        self.size = int(min(max(size or self.SIZE_DEFAULT, self.SIZE_MIN), self.SIZE_MAX))
        self._apply_size()
        self.btn_small.clicked.connect(lambda: self._step_size(-1))
        self.btn_big.clicked.connect(lambda: self._step_size(1))
        self.btn_reset.clicked.connect(lambda: self._set_size(self.SIZE_DEFAULT))
        for seq, f_ in (("Ctrl+0", lambda: self._set_size(self.SIZE_DEFAULT)), ("Ctrl+-", lambda: self._step_size(-1)),
                        ("Ctrl+=", lambda: self._step_size(1)), ("Ctrl++", lambda: self._step_size(1))):
            sc = QShortcut(QKeySequence(seq), self)
            sc.setContext(Qt.WidgetWithChildrenShortcut)
            sc.activated.connect(f_)
        self.btn_copy.clicked.connect(self._copy)
        self.btn_clear.clicked.connect(self.clear_view)
        self.view.verticalScrollBar().valueChanged.connect(self._scrolled)
        self.btn_follow.toggled.connect(lambda on: on and self._to_bottom())
        self._programmatic = False
        self.plain = ""
        self.parts = []
        self._sep = None          # 현재 메시지 구분선 (시작 위치, 길이, 시각 문구)
        self._body = 0            # 현재 메시지 본문 시작 위치 (여기부터 문서 끝까지만 고쳐 쓴다)

    sizeChanged = Signal(int)

    def _apply_size(self):
        # 앱 스타일시트의 글꼴이 위젯 setFont 를 덮어써서 처음 화면이 기본 글꼴(10 pt)로 나왔다 (패널 값은 18 pt)
        # → 이 위젯 자체 스타일시트로 크기를 준다 (앱 스타일시트보다 우선). 문서 기본 글꼴도 같은 값으로
        f = T.ui_font(self.size)
        self.view.setStyleSheet('QTextEdit#qsoRx {{ font-family: "{}"; font-size: {}pt; }}'.format(f.family(), self.size))
        self.view.setFont(f)
        self.view.document().setDefaultFont(f)

    def _step_size(self, d):
        self._set_size(self.size + d)

    def _set_size(self, v):
        """크기 바꾸기 (한 곳). 본문 · 구분선 · 이미 있는 글자 · 새 글자 모두 문서 기본 글꼴 크기를 따른다"""
        v = int(min(max(v, self.SIZE_MIN), self.SIZE_MAX))
        self.btn_small.setEnabled(v > self.SIZE_MIN)
        self.btn_big.setEnabled(v < self.SIZE_MAX)
        if v == self.size:
            return
        self.size = v
        self._apply_size()
        self.sizeChanged.emit(self.size)              # 설정 저장 (qso_font) → 다음 실행 때 유지

    def _copy(self):
        c = self.view.textCursor()
        QApplication.clipboard().setText(c.selectedText() if c.hasSelection() else self.plain)

    def _scrolled(self, val):
        if self._programmatic:
            return
        sb = self.view.verticalScrollBar()
        at_bottom = val >= sb.maximum() - 4
        if self.btn_follow.isChecked() != at_bottom:
            self.btn_follow.blockSignals(True)
            self.btn_follow.setChecked(at_bottom)
            self.btn_follow.blockSignals(False)

    def _to_bottom(self):
        self._programmatic = True
        sb = self.view.verticalScrollBar()
        sb.setValue(sb.maximum())
        self._programmatic = False

    # ---------------------------------------------------------------- 내용
    def clear_view(self):
        self.view.clear()
        self.parts, self.plain, self._sep, self._body = [], "", None, 0

    @staticmethod
    def _fmt(color, bold=False):
        f = QTextCharFormat()
        f.setForeground(QColor(T.c(color)))
        f.setFontWeight(700 if bold else 400)
        return f

    @staticmethod
    def _sep_text(stamp, tag):
        return "====== {}{} ======".format(stamp, " · " + tag if tag else "")

    def new_message(self, df=None, rho=None, stamp=None, mode=None):
        """새 송신: 문서 끝에 빈 줄 + 구분선 (stamp = 코스타스 A 검출 현지 시각, 없으면 지금, mode = 모드 이름)"""
        self.parts = []
        self.plain = ""
        doc = self.view.document()
        c = QTextCursor(doc)
        c.movePosition(QTextCursor.End)
        if not doc.isEmpty():
            c.insertText("\n\n", self._fmt("text"))              # 앞 메시지 끝 → 빈 줄 1개
        stamp = (stamp or datetime.now()).strftime("%Y-%m-%d %H:%M") + (" · {}".format(mode) if mode else "")
        txt = self._sep_text(stamp, "")
        self._sep = [c.position(), len(txt), stamp]
        c.insertText(txt, self._fmt("qso_sep", True))
        c.insertText("\n", self._fmt("text"))
        self._body = c.position()
        if self.btn_follow.isChecked():
            self._to_bottom()
        self.set_meta(state=tr("qso.rx_first_crc_ok").format(mode) if mode else tr("qso.rx_first_crc_ok2"), snr=float("nan"),
                      df=df if df is not None else float("nan"), seg=tr("main.seg_0_confirmed"))      # 구분선은 첫 CRC 통과 때만 열림

    def set_meta(self, state=None, snr=None, df=None, seg=None, err=False):
        if state is not None:
            self.lb_state.setText(trf(state))
            self.lb_state.setProperty("error", bool(err))
            self.lb_state.style().unpolish(self.lb_state)
            self.lb_state.style().polish(self.lb_state)
        if snr is not None:
            self.lb_snr.setText("SNR {:+.1f} dB".format(snr) if np.isfinite(snr) else "SNR -")
        if df is not None:
            self.lb_df.setText("Δf {:+.1f} Hz".format(df) if np.isfinite(df) else "Δf -")
        if seg is not None:
            self.lb_seg.setText(trf(seg))
        self.status.refit()

    def discard_current(self):
        """현재 메시지(구분선부터 끝까지)를 지운다 — 오검출 폐기는 수신 패널에 남기지 않는다"""
        if self._sep is None:
            return
        doc = self.view.document()
        pos = self._sep[0]
        c = QTextCursor(doc)
        c.setPosition(max(0, pos - 2) if pos >= 2 else 0)        # 앞 메시지 뒤 빈 줄 포함
        c.movePosition(QTextCursor.End, QTextCursor.KeepAnchor)
        c.removeSelectedText()
        self._sep, self.parts, self.plain = None, [], ""
        self._body = doc.characterCount() - 1

    def set_tag(self, tag):
        """현재 메시지 구분선 끝에 짧은 표시 (부분 복원 · 신호 소실 · 송신 중지)"""
        if self._sep is None:
            return
        pos, n, stamp = self._sep
        txt = self._sep_text(stamp, tag)
        c = QTextCursor(self.view.document())
        c.setPosition(pos)
        c.setPosition(pos + n, QTextCursor.KeepAnchor)
        c.insertText(txt, self._fmt("qso_sep", True))
        self._sep[1] = len(txt)
        self._body += len(txt) - n

    def render(self, parts):
        """parts: [(글, 종류)] 종류 = ok / fail / pending / note. 현재 메시지 본문(문서 끝)만 바꾼다"""
        self.parts = list(parts)
        sb = self.view.verticalScrollBar()
        keep = sb.value()
        follow = self.btn_follow.isChecked()
        col = {"ok": "text", "fail": "err", "pending": "text3", "note": "text3"}
        self._programmatic = True
        c = QTextCursor(self.view.document())
        c.setPosition(min(self._body, self.view.document().characterCount() - 1))
        c.movePosition(QTextCursor.End, QTextCursor.KeepAnchor)
        c.removeSelectedText()
        for s, kind in self.parts:
            c.insertText(s, self._fmt(col.get(kind, "text")))
        self.plain = "".join(s for s, k in self.parts if k in ("ok", "fail"))
        if follow:
            sb.setValue(sb.maximum())
        else:
            sb.setValue(min(keep, sb.maximum()))
        self._programmatic = False


# ====================================================================== 송신
class QsoTxPanel(QWidget):
    """
    메모장형 여러 줄 입력 + 송신/중지 + 진행률 · 남은 시간.
    송신 중에는 입력칸을 잠그고 구간 단위로 색을 칠한다: 송신 완료 (흐림) / 송신 중 (빨강) / 대기 (기본).
    글자 단위 위치는 인터리버 때문에 ±3.2초 흐려지므로 표시는 32바이트 구간 단위다.
    """
    sendRequested = Signal(str)
    stopRequested = Signal()
    resendRequested = Signal()

    def __init__(self):
        super().__init__()
        v = QVBoxLayout(self)
        v.setContentsMargins(T.sp(2), T.sp(1), T.sp(2), T.sp(2))
        v.setSpacing(T.sp(1))
        self.edit = QPlainTextEdit()
        self.edit.setObjectName("qsoTx")
        self.edit.setPlaceholderText(tr("main.message"))
        self.edit.setFont(T.ui_font(15))
        v.addWidget(self.edit, 1)
        row = QHBoxLayout()
        row.setSpacing(T.sp(2))
        self.btn_send = QPushButton(tr("main.tx"))
        self.btn_send.setObjectName("txButton")
        self.btn_stop = QPushButton(tr("main.stop"))
        self.btn_stop.setEnabled(False)
        self.bar = QProgressBar()
        self.bar.setRange(0, 1000)
        self.bar.setTextVisible(False)
        self.bar.setFixedHeight(T.sp(3))
        self.lb = _lab(tr("qso.remaining"), "num")
        self.lb.setMinimumWidth(200)
        self.btn_resend = QPushButton(tr("qso.resend"))
        self.btn_resend.setEnabled(False)
        self.btn_resend.setVisible(False)
        row.addWidget(self.btn_send)
        row.addWidget(self.btn_resend)
        row.addWidget(self.btn_stop)
        row.addWidget(self.bar, 1)
        row.addWidget(self.lb)
        v.addLayout(row)
        self.btn_send.clicked.connect(lambda: self.sendRequested.emit(self.edit.toPlainText()))
        self.btn_stop.clicked.connect(self.stopRequested.emit)
        self.btn_resend.clicked.connect(self.resendRequested.emit)
        self._text = ""
        self._resend_ok = False
        self._chars = []

    def set_transmitting(self, on, text="", char_segs=None):
        self.edit.setReadOnly(on)
        self.btn_send.setEnabled(not on)
        self.btn_resend.setEnabled(not on and self._resend_ok)
        self.btn_stop.setEnabled(on)
        self.btn_stop.setText(tr("main.stop"))
        self._text = text
        self._chars = char_segs or []
        self._sel_key = None
        if not on:
            self.edit.setExtraSelections([])
            self.bar.setValue(0)
            self.lb.setText(tr("qso.remaining"))
            self._stop_note = ""

    def set_resend(self, visible, enabled):
        """TNG1 전용 '다시 보내기' 버튼 (다른 모드에서는 숨김)"""
        self._resend_ok = bool(enabled)
        self.btn_resend.setVisible(bool(visible))
        self.btn_resend.setEnabled(bool(enabled) and self.btn_send.isEnabled())

    def set_stopping(self, keep_bytes, total_bytes):
        self.btn_stop.setText(tr("qso.halt"))
        self._stop_note = tr("qso.stop_first_bytes").format(keep_bytes, total_bytes)

    def reset_note(self):
        self._stop_note = ""

    def set_note(self, msg):
        """송신 전 경고 (진행 막대 오른쪽 줄)"""
        self.lb.setText(trf(msg))

    def set_progress(self, frac, remain_s, seg_state, seg_text, cut_segs=None):
        """seg_state: 구간별 0 대기 / 1 송신 중 / 2 완료. cut_segs 이상 구간은 중지로 안 보냄 (취소선)"""
        self.bar.setValue(int(1000 * np.clip(frac, 0, 1)))
        self.lb.setText(tr("qso.remaining2").format(fmt_clock(np.ceil(max(remain_s - 0.05, 0.0))), seg_text, getattr(self, "_stop_note", "")))
        key = (tuple(int(x) for x in seg_state), cut_segs)
        if key == getattr(self, "_sel_key", None):             # 구간 상태가 같으면 강조도 같다
            return
        self._sel_key = key
        doc = self.edit.document()
        sels = []
        fmts = {2: ("text3", False), 1: ("err", True)}
        runs, cur = [], None
        for i, k0, k1 in self._chars:
            st = max(seg_state[k] if k < len(seg_state) else 0 for k in (k0, k1))
            if cut_segs is not None and k0 >= cut_segs:
                st = -1
            if cur is not None and cur[2] == st:
                cur[1] = i + 1
            else:
                cur = [i, i + 1, st]
                runs.append(cur)
        for a, b, st in runs:
            if st == 0:
                continue
            f = QTextCharFormat()
            if st == -1:
                f.setForeground(QColor(T.c("text3")))
                f.setFontStrikeOut(True)
            else:
                col, bold = fmts[st]
                f.setForeground(QColor(T.c(col)))
                if bold:
                    f.setFontWeight(700)
            c = QTextCursor(doc)
            c.setPosition(_u16(self._text[:a]))
            c.setPosition(_u16(self._text[:b]), QTextCursor.KeepAnchor)
            s = QTextEdit.ExtraSelection()
            s.cursor, s.format = c, f
            sels.append(s)
        self.edit.setExtraSelections(sels)


# ====================================================================== 송신 해부
def _body():
    w = QWidget()
    w.setObjectName("dockBody")
    w.setAttribute(Qt.WA_StyledBackground, True)
    v = QVBoxLayout(w)
    v.setContentsMargins(T.sp(1), T.sp(1), T.sp(1), T.sp(1))
    v.setSpacing(T.sp(1))
    return w, v


class TxAnalysisPanel:
    """
    송신 표시 제어 — 화면 두 개를 함께 갱신한다 (각각 독립 도킹 패널).
      monitor  '송신 모니터': 송신 단계 · 현재 구간 · 시간 · 실효 속도 · 구간 글자 · 바이트, 프레임/구간 타임라인 커서,
                             송신 진폭 · 스펙트럼
      encode   '송신 해부'  : ① 바이트 → ② 정보 비트 → ③ 컨볼루션 부호 비트 → ④ 인터리빙 뒤 송신 위치 (현재 구간)
    self.shown: 부호화 층에 지금 그린 값 (검증용 — 실제 송신 프레임과 코드로 비교한다)
    """
    H_WAVE, H_LAYER, H_TX = 150, 76, 110

    def __init__(self):
        # ---------------- 송신 모니터
        self.monitor, v = _body()
        self.f = {}
        g = QGridLayout()
        g.setHorizontalSpacing(T.sp(3))
        g.setVerticalSpacing(0)
        rows = (("phase", tr("qso.tx_stage")), ("seg", tr("qso.current_seg")), ("el", tr("qso.elapsed")), ("left", tr("qso.remaining3")),
                ("rate_d", tr("qso.throughput_data")), ("rate_a", tr("qso.throughput_total")),
                ("chars", tr("qso.seg_text")), ("hex", tr("qso.seg_bytes")))
        tips = {"rate_d": tr("qso.chars_in_sent_segments"),
                "rate_a": tr("qso.chars_in_sent_segments2")}
        for i, (k, lab) in enumerate(rows):
            lb = _lab(lab, "dim")
            g.addWidget(lb, i, 0)
            w = _lab("—", "num")
            w.setTextInteractionFlags(Qt.TextSelectableByMouse)
            w.setWordWrap(k == "hex")
            g.addWidget(w, i, 1)
            self.f[k] = w
        g.setColumnStretch(1, 1)
        v.addLayout(g)
        self.timeline = TimelinePanel(tr("qso.tx_idle"))
        self.timeline.setMinimumHeight(110)
        v.addWidget(self.timeline, 2)
        row = QHBoxLayout()
        row.setSpacing(T.sp(1))
        self.p_wave = pg.PlotWidget(axisItems=fit_axes("left", "bottom"))
        T.style_plot(self.p_wave, None, tr("w.time_ms"), tr("qso.amplitude"), grid=False)
        self.p_spec = pg.PlotWidget(axisItems=fit_axes("left", "bottom"))
        T.style_plot(self.p_spec, None, tr("w.frequency_hz"), "dBFS", grid=False)
        for p in (self.p_wave, self.p_spec):
            p.setMinimumHeight(self.H_WAVE)
            p.setMouseEnabled(x=False, y=False)
            p.hideButtons()
            row.addWidget(p, 1)
        self.c_wave = self.p_wave.plot(pen=T.pen("tx", 1))
        self.c_spec = self.p_spec.plot(pen=T.pen("tx", 1))
        self.p_wave.setXRange(0, 60, padding=0)
        self.p_spec.setXRange(1000, 2000, padding=0)
        self.p_spec.setYRange(-110, -10, padding=0)
        v.addLayout(row, 3)
        # ---------------- 송신 해부 (부호화 4층)
        self.encode, v2 = _body()
        titles = (("bytes", tr("qso.bytes_32_seg_2")), ("info", tr("qso.info_bits")),
                  ("coded", tr("qso.convolutional_code_bits_k")), ("tx", tr("qso.tx_position_after_interleaving")))
        self.enc = {}
        for k, t in titles:
            pl = pg.PlotWidget(axisItems=fit_axes("left", "bottom"))
            T.style_plot(pl, None, tr("qso.data_frame") if k == "tx" else None, None, grid=False)
            pl.getAxis("left").setStyle(showValues=False)
            pl.getAxis("left").setWidth(8)
            if k != "tx":
                pl.getAxis("bottom").setStyle(showValues=False)
                pl.getAxis("bottom").setHeight(2)
            pl.setTitle('<span style="color:{}; font-size:{}pt">{}</span>'.format(T.c("text2"), T.PT_SMALL, t),
                        justify="left")
            pl.setMouseEnabled(x=False, y=False)
            pl.hideButtons()
            pl.setMinimumHeight(self.H_TX if k == "tx" else self.H_LAYER)
            if k == "tx":                                         # ④ 층: 정해진 프레임 범위 + 페이지 넘김
                v2.addWidget(self._tx_pager())
            v2.addWidget(pl, 3 if k == "tx" else 2)
            self.enc[k] = pl.getPlotItem()
        self._items = []
        self.tx = None
        self._seg = None
        self.shown = None
        self._peak = 1.0
        self._pg = {"lo": 0, "hi": 1, "page": 0, "manual": False, "cur": 0}
        for key_, d_ in ((Qt.Key_Left, -1), (Qt.Key_Right, 1)):
            sc = QShortcut(QKeySequence(key_), self.encode)
            sc.setContext(Qt.WidgetWithChildrenShortcut)
            sc.activated.connect(lambda d_=d_: self._tx_go(self._pg["page"] + d_, manual=True))

    PAGE_F = 12            # ④ 층 한 화면 프레임 수 (프레임 안 96 위치가 뭉개지지 않는 폭)

    def _tx_pager(self):
        w = QWidget()
        h = QHBoxLayout(w)
        h.setContentsMargins(0, 0, 0, 0)
        h.setSpacing(T.sp(1))
        self.b_tx_prev = QPushButton(tr("qso.prev"))
        self.b_tx_prev.clicked.connect(lambda: self._tx_go(self._pg["page"] - 1, manual=True))
        self.lb_tx_page = _lab("—", "num")
        self.b_tx_next = QPushButton(tr("qso.next"))
        self.b_tx_next.clicked.connect(lambda: self._tx_go(self._pg["page"] + 1, manual=True))
        self.b_tx_follow = QPushButton(tr("qso.follow_tx"))
        self.b_tx_follow.clicked.connect(lambda: self._tx_follow())
        self.ov_tx = _TxOverview()
        self.ov_tx.seek.connect(lambda f: self._tx_go(int(f) // self.PAGE_F - self._pg["lo"] // self.PAGE_F, manual=True))
        h.addWidget(self.b_tx_prev)
        h.addWidget(self.lb_tx_page)
        h.addWidget(self.b_tx_next)
        h.addWidget(self.ov_tx, 1)
        h.addWidget(self.b_tx_follow)
        return w

    def _tx_pages(self):
        lo, hi = self._pg["lo"], self._pg["hi"]
        p0 = lo // self.PAGE_F
        return max(1, (hi - 1) // self.PAGE_F - p0 + 1), p0

    def _tx_go(self, page, manual=False):
        n, p0 = self._tx_pages()
        page = int(np.clip(page, 0, n - 1))
        self._pg["page"] = page
        if manual:
            self._pg["manual"] = True
        a = (p0 + page) * self.PAGE_F
        self.enc["tx"].setXRange(a - 0.5, a + self.PAGE_F - 0.5, padding=0)
        self.lb_tx_page.setText("{} / {} · F{}–F{}".format(page + 1, n, a + 1, a + self.PAGE_F))
        self.ov_tx.set_view(a, a + self.PAGE_F)
        self.b_tx_prev.setEnabled(page > 0)
        self.b_tx_next.setEnabled(page < n - 1)

    def _tx_follow(self):
        self._pg["manual"] = False
        n, p0 = self._tx_pages()
        self._tx_go(int(self._pg["cur"]) // self.PAGE_F - p0)

    def set_message(self, tx, tl):
        """새 송신: TxAudio 와 두 줄 타임라인"""
        self.tx = tx
        self._seg = None
        self.timeline.show_packet(tl)
        self.timeline.set_cursor(0)
        self._peak = float(max(np.max(np.abs(tx.audio)), 1e-6))
        self.p_wave.setYRange(-1.1 * self._peak, 1.1 * self._peak, padding=0)

    def clear(self, note=tr("qso.tx_idle")):
        self.tx = None
        self.timeline.show_packet(None)
        self.timeline.set_cursor(None)
        for k in self.f:
            self.f[k].setText("—")

    def phase(self, t):
        """경과 t → 송신 단계 문구"""
        tx = self.tx
        n, fr = tx.plan.n, tx.frame_s
        if hasattr(tx, "t_frame"):                                    # TNG5 v2: 시작 동기 Welch12 · 중간 동기 M · B×1
            import tng5
            t_dend = tx.t_frame(n - 1) + fr
            if t < tx.t_tone:
                return tr("qso.lead_silence")
            if t < tx.t_a:
                return tr("qso.tone_300_ms")
            if t < tx.t_data:
                return tr("qso.start_sync_welch12_costas")
            if t < tx.frame_t0:
                return tr("qso.guard_frame_start")
            if t < t_dend:
                for kq in range(1, tng5.n_mids(n) + 1):
                    tm = tx.t_frame(kq * tng5.MID_EVERY)
                    if tm - tng5.NM / tng5.FS <= t < tm:
                        return tr("qso.mid_sync_m").format(kq)
                f = int(np.searchsorted([tx.t_frame(i) for i in range(n)], t, side="right")) - 1
                return tr("qso.data_frame2").format(max(f, 0) + 1, n)
            if t < tx.t_b:
                return tr("qso.guard_frame_end")
            if t < tx.t_b + tx.t_b_len:
                return tr("qso.costas_b_0_34")
            return tr("qso.tail_silence")
        t_dend = tx.frame_t0 + n * fr
        if t < tx.t_tone:
            return tr("qso.lead_silence")
        if t < tx.t_a:
            return tr("qso.tone_300_ms")
        if t < tx.t_data:
            return tr("qso.costas_a")
        if t < tx.frame_t0:
            return tr("qso.guard_frame_start")
        if t < t_dend:
            return tr("qso.data_frame2").format(int((t - tx.frame_t0) / fr) + 1, n)
        if t < tx.t_b:
            return tr("qso.guard_frame_end")
        if t < tx.t_b + getattr(tx, "t_b_len", 0.224):
            return tr("qso.costas_b")
        return tr("qso.tail_silence")

    def update_tx(self, t, seg_state, frames_done, frames_started, cut_segs=None):
        tx = self.tx
        if tx is None:
            return
        plan = tx.plan
        cur = [k for k in range(plan.n_seg) if seg_state[k] == 1]
        sent = int(np.sum(seg_state == 2))
        self.f["phase"].setText(self.phase(t))
        if cur:
            seg_txt = tr("qso.sending_done").format(", ".join(str(k + 1) for k in cur), plan.n_seg, sent)
        elif sent >= plan.n_seg:
            seg_txt = tr("qso.none_seg_sent").format(plan.n_seg)
        else:
            seg_txt = tr("qso.idle_seg_total").format(plan.n_seg)
        self.f["seg"].setText(seg_txt + ("" if cut_segs is None else tr("qso.stop_up_to_seg").format(cut_segs)))
        self.f["el"].setText(fmt_clock(t))
        self.f["left"].setText(fmt_clock(np.ceil(max(tx.dur - t - 0.05, 0.0))))       # 올림 (끝나는 순간 00:00)
        is5 = hasattr(tx, "seg_view")
        if is5:                                                       # TNG5: 구간 18바이트 = 27자 (첫 구간 24자)
            n_chars = tx.chars_done(sent)
            done_bytes = 2 + 2 * ((n_chars + 2) // 3)
        else:
            done_bytes = min(len(tx.S), SEG_BYTES * sent)
            n_chars = len(tx.S[2:done_bytes].decode("utf-8", errors="ignore"))
        td = t - tx.frame_t0
        self.f["rate_d"].setText("—" if td <= 0 else tr("qso.1f_cps_0f_bps").format(
            n_chars / td, 8 * max(done_bytes - 2, 0) / td, n_chars))
        self.f["rate_a"].setText(tr("qso.1f_cps_0f_bps2").format(n_chars / max(t, 1e-6),
                                                                  8 * max(done_bytes - 2, 0) / max(t, 1e-6)))
        k = cur[0] if cur else min(sent, plan.n_seg - 1)
        if k != self._seg:
            self._seg = k
            self._draw_segment(k)
        a, b, s, n = plan.seg_bits[k]
        txt = tx.seg_chars(k) if is5 else tx.S[max(s, 2):s + n].decode("utf-8", errors="replace").replace("\n", "↵")
        self.f["chars"].setText("[{}] {}{}".format(k + 1, tr("qso.length_b").format(len(tx.P)) if s == 0 else "", txt))
        self.f["hex"].setText(" ".join("{:02X}".format(x) for x in tx.S[s:s + n]))
        # 타임라인 커서 (0 = 톤 시작) · 색
        self.timeline.set_cursor(1000.0 * (t - tx.t_tone))
        fs_ = ["sent" if f < frames_done else ("sending" if f < frames_started else "wait") for f in range(plan.n)]
        ss = ["sent" if x == 2 else ("sending" if x == 1 else "wait") for x in seg_state]
        self.timeline.set_tx_states(fs_, ss)
        # 파형 (최근 60 ms) · 스펙트럼 (최근 128 ms) — 실제로 내보내는 오디오 (송신 음량 곱하기 전)
        i = int(t * tx.fs)
        w = tx.audio[max(0, i - int(0.06 * tx.fs)):i]
        if len(w) > 16:
            self.c_wave.setData(1000 * np.arange(len(w)) / tx.fs, w)
        n2 = int(tx.fs * 0.128)
        seg = tx.audio[max(0, i - n2):i]
        if len(seg) == n2:
            win = np.hanning(n2)
            pw = np.abs(np.fft.rfft(seg * win)) ** 2
            db = 10 * np.log10(pw / (np.sum(win) / 2) ** 2 + 1e-16)
            self.c_spec.setData(np.fft.rfftfreq(n2, 1.0 / tx.fs), db)
        # ④ 층: 이미 나간 위치는 흐리게, 빨간 선 = 현재 프레임
        if self._items and frames_done != self._tx_done:           # 바뀔 때만, 붓 2개 재사용 (심볼 캐시 적중)
            self._tx_done = frames_done
            self._tx_scatter.setBrush([self._br_sent if p < frames_done else self._br_wait for p in self._tx_frames])
            self._tx_line.setValue(min(max(frames_started, 0), plan.n))
            self._pg["cur"] = min(max(frames_started - 1, 0), plan.n - 1)
            self.ov_tx.set_cur(self._pg["cur"])
            if not self._pg["manual"]:                               # 따라가기: 현재 프레임이 범위 밖이면 넘김
                n_, p0 = self._tx_pages()
                pg_ = int(np.clip(self._pg["cur"] // self.PAGE_F - p0, 0, n_ - 1))
                if pg_ != self._pg["page"]:
                    self._tx_go(pg_)

    def _draw_segment(self, k):
        for pl, it in self._items:
            pl.removeItem(it)
        self._items = []
        tx, plan = self.tx, self.tx.plan
        a, e, s, n = plan.seg_bits[k]
        from framing import crc16_ccitt
        add = lambda key, it: (self.enc[key].addItem(it), self._items.append((self.enc[key], it)))
        if hasattr(tx, "seg_view"):                                   # TNG5: 18바이트 + CRC16, K7 R1/4, 반복 4회
            full, kind, bits5, coded5, pos5 = tx.seg_view(k)
        else:
            Pp = tx.si["Pp"]
            data = Pp[s:s + n]
            full = data + crc16_ccitt(data).to_bytes(2, "big")
            kind = ["len" if (s == 0 and i < 2) else ("crc" if i >= n else "body") for i in range(len(full))]
        # ① 바이트 (v2 첫 구간의 앞 2바이트 = 길이 필드)
        nb = len(full)
        col = {"len": "cat3", "body": "tx_dim", "crc": "fec"}
        add("bytes", pg.BarGraphItem(x0=np.arange(nb), width=0.92, y0=0, height=0.8,
                                     brushes=[T.brush(col[c]) for c in kind], pen=None))
        for i, x in enumerate(full):
            ti = pg.TextItem("{:02X}".format(x), color=T.c("bg0") if kind[i] != "body" else T.c("text"),
                             anchor=(0.5, 0.5))
            ti.setFont(T.mono_font(T.PT_SMALL))
            ti.setPos(i + 0.46, 0.4)
            add("bytes", ti)
        self.enc["bytes"].setXRange(-0.2, max(nb, 8), padding=0)
        self.enc["bytes"].setYRange(-0.1, 1.0, padding=0)
        # ② 정보 비트 ③ 부호 비트 — 부호기 상태 = 앞 구간 마지막 6 정보 비트 (= 앞 구간 CRC16 의 끝 6비트)
        if hasattr(tx, "seg_view"):
            bits, coded = bits5, coded5
        else:
            bits = np.unpackbits(np.frombuffer(full, np.uint8)).astype(np.int64)
            prev = (np.unpackbits(np.frombuffer(crc16_ccitt(Pp[s - SEG_BYTES:s]).to_bytes(2, "big"),
                                                np.uint8))[-6:].astype(np.int64) if s > 0 else np.zeros(0, np.int64))
            from link7 import conv_encode_rows
            coded = conv_encode_rows(np.concatenate([prev, bits]))[0][2 * len(prev):]
        lut = np.array([pg.mkColor(T.c("bg3")).getRgb()[:3], pg.mkColor(T.c("text2")).getRgb()[:3]], dtype=np.uint8)
        for key, arr in (("info", bits), ("coded", coded)):
            it = pg.ImageItem(arr.astype(np.float32)[None, :])
            it.setLookupTable(lut)
            it.setLevels((0, 1))
            it.setRect(QRectF(0, 0, len(arr), 1))
            add(key, it)
            self.enc[key].setXRange(0, len(arr), padding=0)
            self.enc[key].setYRange(0, 1.2, padding=0)
        # ④ 송신 위치: 부호 비트 p → 송신 위치 q[p] (프레임 = q // 96, 세로 = 프레임 안 위치)
        pos = pos5 if hasattr(tx, "seg_view") else plan.q[2 * a:2 * e]
        fr = pos / 96.0
        self._tx_frames = fr
        self._br_sent, self._br_wait = T.brush("text3"), T.brush("tx")
        self._tx_done = None
        self._tx_scatter = pg.ScatterPlotItem(fr, (pos % 96) / 96.0, size=4, pen=None, brush=self._br_wait)
        add("tx", self._tx_scatter)
        self._tx_line = pg.InfiniteLine(0, angle=90, pen=T.pen("err", 1.4))
        add("tx", self._tx_line)
        self._pg.update({"lo": int(fr.min()), "hi": int(fr.max()) + 1, "manual": False})
        self.ov_tx.set_frames(np.asarray(pos) // 96, plan.n)
        n_, p0 = self._tx_pages()
        self._tx_go(int(np.clip(self._pg["cur"] // self.PAGE_F - p0, 0, n_ - 1)))
        self.enc["tx"].setYRange(-0.05, 1.05, padding=0)
        self.shown = {"k": k, "bytes": bytes(full), "bits": bits.copy(), "coded": coded.copy(), "pos": pos.copy()}


class _TxOverview(QWidget):
    """④ 층 개요: 전체 데이터 프레임 가로 막대 · 이 구간 비트가 실린 프레임 (진함) · 보이는 범위 테두리 · 현재 프레임 선.
    클릭 · 끌기 = 그 프레임이 있는 범위로"""
    seek = Signal(float)

    def __init__(self):
        super().__init__()
        self.setFixedHeight(T.sp(5))
        self.setMinimumWidth(120)
        self.n, self.fr, self.view, self.cur = 1, np.zeros(0, int), None, None

    def set_frames(self, frames, n):
        self.fr, self.n = np.unique(np.asarray(frames, int)), max(int(n), 1)
        self.update()

    def set_view(self, a, b):
        self.view = (a, b)
        self.update()

    def set_cur(self, f):
        self.cur = f
        self.update()

    def paintEvent(self, ev):
        from PySide6.QtGui import QPainter, QColor, QPen
        from PySide6.QtCore import QRectF
        p = QPainter(self)
        w, h = self.width(), self.height()
        p.fillRect(self.rect(), QColor(T.c("plot")))
        X = lambda f: f / self.n * w
        p.setPen(Qt.NoPen)
        p.fillRect(QRectF(0, 3, w, h - 6), QColor(T.c("bg3")))
        for f in self.fr:
            p.fillRect(QRectF(X(f), 3, max(X(f + 1) - X(f), 1.0), h - 6), QColor(T.c("tx")))
        if self.cur is not None:
            p.setPen(QPen(QColor(T.c("err")), 1.5))
            p.drawLine(int(X(self.cur + 0.5)), 0, int(X(self.cur + 0.5)), h)
        if self.view is not None:
            p.setPen(QPen(QColor(T.c("text")), 1.5))
            p.setBrush(Qt.NoBrush)
            p.drawRect(QRectF(X(self.view[0]) + 0.75, 0.75, max(X(self.view[1]) - X(self.view[0]) - 1.5, 2), h - 1.5))
        p.end()

    def _emit(self, ev):
        self.seek.emit(ev.position().x() / max(self.width(), 1) * self.n)

    def mousePressEvent(self, ev):
        self._emit(ev)

    def mouseMoveEvent(self, ev):
        if ev.buttons() & Qt.LeftButton:
            self._emit(ev)


# ====================================================================== 수신 로그
class RxLog(QTableWidget):
    """
    수신 로그 — 성공 · 실패 · 부분 복원 · A만 검출 · B만 검출 전부. 송신도 한 줄 (황토색).
    행을 고르면 packetSelected(id).
    """
    packetSelected = Signal(object)
    COLS = ["UTC", tr("w.result"), "SNR", "Δf Hz", tr("w.seg"), tr("qso.drift"), tr("qso.resync"), "FEC", "RF", tr("qso.message_reason")]
    C_FEC = 7
    WIDTHS = (70, 120, 54, 56, 70, 76, 52, 50, 96)
    rf_source = None                         # 18부: 행 추가 때 RF 주파수 (Hz) 를 돌려주는 함수 (앱이 넣음)

    def __init__(self):
        super().__init__(0, len(self.COLS))
        self.setHorizontalHeaderLabels(self.COLS)
        self.verticalHeader().setVisible(False)
        self.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.setSelectionMode(QAbstractItemView.SingleSelection)
        self.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.setShowGrid(False)
        self.setWordWrap(False)
        self.setFocusPolicy(Qt.NoFocus)
        h = self.horizontalHeader()
        h.setHighlightSections(False)
        h.setDefaultAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        h.setFixedHeight(24)
        for i, w in enumerate(self.WIDTHS):
            self.setColumnWidth(i, w)
            h.setSectionResizeMode(i, QHeaderView.Fixed)
        h.setSectionResizeMode(len(self.COLS) - 1, QHeaderView.Stretch)
        self.verticalHeader().setDefaultSectionSize(24)
        self.itemSelectionChanged.connect(self._sel)
        self._mono = T.mono_font(T.PT_BODY)

    def _rf(self, row):
        hz = row.get("rf_hz")
        if hz is None and self.rf_source is not None:
            try:
                hz = self.rf_source()
            except Exception:
                hz = None
        if hz is None:
            return ""
        import rig
        return rig.fmt_hz(hz)

    def _item(self, v, pid, num=False, color=None):
        it = QTableWidgetItem(v)
        it.setData(Qt.UserRole, pid)
        if num:
            it.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
            it.setFont(self._mono)
        if color:
            it.setForeground(T.qcolor(color))
        return it

    def add(self, row):
        """row: id, utc_hms, dir, result, result_col, snr, df, seg, drift, relock, fec, text, text_col, tip"""
        r = self.rowCount()
        self.insertRow(r)
        pid = row.get("id")
        f1 = lambda v, fmt: "" if v is None or (isinstance(v, float) and not np.isfinite(v)) else fmt.format(v)
        vals = [(row.get("utc_hms", ""), False, "text2"),
                (trf(row.get("result", "")), False, row.get("result_col")),
                (f1(row.get("snr"), "{:+.1f}"), True, None),
                (f1(row.get("df"), "{:+.1f}"), True, None),
                (trf(row.get("seg") or ""), True, None),
                (f1(row.get("drift"), "{:+.0f} ppm"), True, None),
                ("" if row.get("relock") is None else str(row["relock"]), True, None),
                ("" if row.get("fec") is None else str(row["fec"]), True, None),
                (self._rf(row), True, "text2")]
        for c, (v, num, col) in enumerate(vals):
            self.setItem(r, c, self._item(v, pid, num, col))
        it = self._item(trf(row.get("text", "")), pid, color=row.get("text_col", "text"))       # 사용자 글은 57자 문자표라 한글 없음
        it.setToolTip(trf(row.get("tip") or row.get("text", "")))
        self.setItem(r, len(self.COLS) - 1, it)
        self.scrollToBottom()

    def remove_id(self, pid):
        """이 id 의 행을 모두 지운다 (TNG1 수신 중 행 → 최종 결과 행으로 바꿀 때)"""
        for row in range(self.rowCount() - 1, -1, -1):
            it = self.item(row, 0)
            if it is not None and it.data(Qt.UserRole) == pid:
                self.removeRow(row)

    def add_result_note(self, pid, note):
        """결과 칸 뒤에 확인 표시를 덧붙인다 (예: TNG5 길이 필드 결과 뒤 코스타스 B 확인)"""
        note = trf(note)
        for row in range(self.rowCount() - 1, -1, -1):
            it = self.item(row, 0)
            if it is not None and it.data(Qt.UserRole) == pid:
                r_ = self.item(row, 1)
                if r_ is not None and note not in r_.text():
                    r_.setText(r_.text() + note)
                    r_.setToolTip((r_.toolTip() + "\n" if r_.toolTip() else "") + note.strip(" ·"))
                return True
        return False

    def set_fec(self, pid, v):
        for row in range(self.rowCount() - 1, -1, -1):
            it = self.item(row, 0)
            if it is not None and it.data(Qt.UserRole) == pid:
                c = self._item(str(v), pid, num=True, color="fec" if v else "text3")
                self.setItem(row, self.C_FEC, c)
                break

    def _sel(self):
        items = self.selectedItems()
        if items:
            self.packetSelected.emit(items[0].data(Qt.UserRole))
