"""
TNG1 전용 전문가 패널 (16-GFSK + 처프). 모드가 TNG1 일 때 기존 성상도 · 아이 · 동기 맵 · 해부 · 송신 해부 자리에 대신 보인다.

  ToneGridPanel   16음 격자: 세로 16음 · 가로 시간 (기호 1개 = 1칸, 160 ms). 역처프 뒤 에너지. 위 = 수신, 아래 = 송신 (같은 시간축)
  ConfPanel       기호별 확신도 (1등 칸 ÷ 2등 칸, dB) 시간 그래프
  FoldPanel       동기 누적 지도 (블록 주기 위상 × 주파수 오프셋, 누적 블록 수 K)
  SyncTonesPanel  블록 안 흩뿌린 동기 12음의 위치 · 검출 점수
  Dissect1Panel   패킷 해부: 블록 → 기호 → 비터비 비트 → CRC16 → 가변 길이 문자
  TxDissect1Panel 송신 해부: 문자 → 가변 길이 비트 → 블록 + CRC16 → 부호화 → 16음 기호
"""
import math
import html

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import Qt, QRectF, Signal
from PySide6.QtGui import QPainterPath, QColor, QFont
from PySide6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel, QComboBox, QTextBrowser, QGraphicsPathItem)

import theme as T
import charset1
import tng1
from strings_ko import tr

C = tng1.CFGS["TNG1"]
SPAN_S = 32.0
_LUT = (pg.colormap.get("inferno").getLookupTable(0.0, 1.0, 256))


def _img(a):
    """[x (시간 · 위상), y (음 · 주파수)] 배열 → ImageItem 축 순서에 맞게 (앱은 row-major 로 설정)"""
    return np.ascontiguousarray(a.T) if pg.getConfigOption("imageAxisOrder") == "row-major" else a


def _style(p, xl=None, yl=None):
    p.showGrid(x=False, y=False)
    for ax in ("left", "bottom"):
        a = p.getAxis(ax)
        a.setPen(pg.mkPen(T.c("line2")))
        a.setTextPen(pg.mkPen(T.c("text2")))
        a.setStyle(tickFont=T.mono_font(T.PT_SMALL))
    if xl:
        p.setLabel("bottom", '<span style="color:{}; font-size:{}pt">{}</span>'.format(T.c("text2"), T.PT_SMALL, xl))
    if yl:
        p.setLabel("left", '<span style="color:{}; font-size:{}pt">{}</span>'.format(T.c("text2"), T.PT_SMALL, yl))
    p.getViewBox().setMouseEnabled(x=False, y=False)
    p.hideButtons()


def _cells_path(t0s, tones, T_):
    """칸 테두리 경로: 칸 (시작 시각, 음) 목록"""
    path = QPainterPath()
    for t0, k in zip(t0s, tones):
        path.addRect(QRectF(float(t0), float(k), T_, 1.0))
    return path


def _path_item(col, width=1.2, alpha=255):
    it = QGraphicsPathItem()
    it.setPen(T.pen(col, width, alpha))
    return it


# ====================================================================== 16음 격자
class ToneGridPanel(QWidget):
    blockClicked = Signal(object)            # {"abs": 수신 입력 절대 샘플 (블록 시작), "label": "B3"}

    def __init__(self):
        super().__init__()
        v = QVBoxLayout(self)
        v.setContentsMargins(T.sp(1), T.sp(1), T.sp(1), T.sp(1))
        v.setSpacing(T.sp(1))
        self.info = QLabel(tr("main.waiting"))
        self.info.setObjectName("panelNote")
        v.addWidget(self.info)
        self.gl = pg.GraphicsLayoutWidget()
        v.addWidget(self.gl, 1)
        self.prx = self.gl.addPlot(row=0, col=0)
        self.ptx = self.gl.addPlot(row=1, col=0)
        self.ptx.setXLink(self.prx)
        for p, name in ((self.prx, tr("main.rx")), (self.ptx, tr("main.tx"))):
            _style(p, tr("t1.time_s_0_now") if p is self.ptx else None, name + tr("t1.tone"))
            p.setYRange(0, 16, padding=0)
            p.setXRange(-SPAN_S, 0, padding=0)
            p.getAxis("left").setTicks([[(k + 0.5, str(k)) for k in (0, 4, 8, 12, 15)]])
        self.img_rx = pg.ImageItem()
        self.img_rx.setLookupTable(_LUT)
        self.prx.addItem(self.img_rx)
        self.img_tx = pg.ImageItem()
        self.img_tx.setLookupTable(_LUT)
        self.ptx.addItem(self.img_tx)
        self.p_sync = _path_item("wf_band", 1.0, 230)          # 동기 음 (다른 색 테두리)
        self.p_final = _path_item("text", 1.2, 230)            # 최종 판단 (CRC 통과 블록 재부호화)
        self.p_tsync = _path_item("wf_band", 1.0, 230)
        self.p_tcur = _path_item("accent", 2.0, 255)           # 지금 보내는 칸
        for it in (self.p_sync, self.p_final):
            self.prx.addItem(it)
        for it in (self.p_tsync, self.p_tcur):
            self.ptx.addItem(it)
        self.mis = pg.ScatterPlotItem(symbol="x", size=9, pen=pg.mkPen(T.c("err"), width=2), brush=None)
        self.prx.addItem(self.mis)                             # 에너지 1등 칸 ≠ 최종 (오류정정이 고친 자리)
        self._blk_items = []
        self._blocks = []
        self.gl.scene().sigMouseClicked.connect(self._click)

    def _clear_blocks(self):
        for it, p in self._blk_items:
            p.removeItem(it)
        self._blk_items = []

    def set_rx(self, s):
        self._clear_blocks()
        if not s:
            self.img_rx.clear()
            for it in (self.p_sync, self.p_final):
                it.setPath(QPainterPath())
            self.mis.setData([], [])
            self.info.setText(tr("t1.no_input"))
            return
        t, T_, E = s["t"], s["T"], s["E"]
        db = 10 * np.log10(np.maximum(E, 1e-3))
        img = np.clip((db + 3.0) / 18.0, 0, 1)                  # -3 ~ +15 dB (잡음 평균 = 0 dB)
        self.img_rx.setImage(_img(img), autoLevels=False, levels=(0, 1))
        self.img_rx.setRect(QRectF(float(t[0]), 0.0, T_ * len(t), 16.0))
        sm = s["sync"]
        self.p_sync.setPath(_cells_path(t[sm], s["tone_sync"][sm], T_))
        fin = s["final"]
        k = fin >= 0
        self.p_final.setPath(_cells_path(t[k], fin[k], T_))
        am = E.argmax(1)
        bad = k & (am != fin) & ~sm
        self.mis.setData(t[bad] + T_ / 2, am[bad] + 0.5)
        self._blocks = s["blocks"]
        for tb, lab, a_in in s["blocks"]:
            ln = pg.InfiniteLine(tb, angle=90, pen=T.pen("text3", 1, 200, Qt.DashLine))
            tx = pg.TextItem(lab, color=T.c("text2"), anchor=(0, 0))
            tx.setFont(T.mono_font(T.PT_SMALL))
            tx.setPos(tb, 16)
            for it in (ln, tx):
                self.prx.addItem(it)
                self._blk_items.append((it, self.prx))
        self.info.setText(s.get("note", ""))

    def set_tx(self, s):
        if not s:
            self.img_tx.clear()
            for it in (self.p_tsync, self.p_tcur):
                it.setPath(QPainterPath())
            return
        t, T_, sym, cur = s["t"], s["T"], s["sym"], s["cur"]
        img = np.zeros((len(t), 16))
        sent = np.arange(len(t)) <= cur
        img[np.nonzero(sent)[0], sym[sent]] = 0.85
        img[np.nonzero(~sent)[0], sym[~sent]] = 0.18            # 앞으로 보낼 칸 (흐리게)
        self.img_tx.setImage(_img(img), autoLevels=False, levels=(0, 1))
        self.img_tx.setRect(QRectF(float(t[0]), 0.0, T_ * len(t), 16.0))
        sm = s["sync"]
        self.p_tsync.setPath(_cells_path(t[sm], sym[sm], T_))
        if 0 <= cur < len(t):
            self.p_tcur.setPath(_cells_path([t[cur]], [sym[cur]], T_))
        else:
            self.p_tcur.setPath(QPainterPath())

    def _click(self, ev):
        vb = self.prx.getViewBox()
        if not vb.sceneBoundingRect().contains(ev.scenePos()):
            return
        x = vb.mapSceneToView(ev.scenePos()).x()
        for tb, lab, a_in in self._blocks:
            if tb <= x < tb + C.block_s:
                self.blockClicked.emit({"abs": a_in, "label": lab})
                return


# ====================================================================== 기호별 확신도
class ConfPanel(QWidget):
    def __init__(self):
        super().__init__()
        v = QVBoxLayout(self)
        v.setContentsMargins(T.sp(1), T.sp(1), T.sp(1), T.sp(1))
        self.p = pg.PlotWidget()
        _style(self.p.getPlotItem(), tr("t1.time_s_0_now"), tr("t1.1st_2nd_cell_db"))
        self.p.setXRange(-SPAN_S, 0, padding=0)
        self.p.setYRange(0, 15, padding=0)
        v.addWidget(self.p, 1)
        self.sd = pg.ScatterPlotItem(size=5, pen=None, brush=pg.mkBrush(T.c("rx")))
        self.ss = pg.ScatterPlotItem(size=6, pen=None, brush=pg.mkBrush(T.c("wf_band")))
        self.sm = pg.ScatterPlotItem(symbol="x", size=8, pen=pg.mkPen(T.c("err"), width=2), brush=None)
        for it in (self.sd, self.ss, self.sm):
            self.p.addItem(it)
        self.p.addItem(pg.InfiniteLine(3.0, angle=0, pen=T.pen("text3", 1, 150, Qt.DashLine)))

    def set_data(self, s):
        if not s:
            for it in (self.sd, self.ss, self.sm):
                it.setData([], [])
            return
        E, t = s["E"], s["t"] + s["T"] / 2
        srt = np.sort(E, 1)
        m = 10 * np.log10(np.maximum(srt[:, -1], 1e-9) / np.maximum(srt[:, -2], 1e-9))
        sm = s["sync"]
        fin = s["final"]
        bad = (fin >= 0) & (E.argmax(1) != fin) & ~sm
        self.sd.setData(t[~sm & ~bad], m[~sm & ~bad])
        self.ss.setData(t[sm], m[sm])
        self.sm.setData(t[bad], m[bad])


# ====================================================================== 동기 누적 지도
class FoldPanel(QWidget):
    def __init__(self):
        super().__init__()
        v = QVBoxLayout(self)
        v.setContentsMargins(T.sp(1), T.sp(1), T.sp(1), T.sp(1))
        self.note = QLabel("—")
        self.note.setObjectName("panelNote")
        v.addWidget(self.note)
        self.p = pg.PlotWidget()
        _style(self.p.getPlotItem(), tr("t1.block_phase_s"), tr("t1.freq_offset_hz"))
        v.addWidget(self.p, 1)
        self.img = pg.ImageItem()
        self.img.setLookupTable(_LUT)
        self.p.addItem(self.img)
        self.cand = pg.ScatterPlotItem(symbol="o", size=10, pen=pg.mkPen(T.c("text"), width=1.5), brush=None)
        self.p.addItem(self.cand)

    def set_data(self, f):
        if not f:
            self.img.clear()
            self.cand.setData([], [])
            self.note.setText("—")
            return
        S, fr, hop_s, K, thr = f["S"], f["fr"], f["hop_s"], f["K"], f["thr"]
        img = np.clip((S - np.median(S)) / max(thr - np.median(S), 1e-6), 0, 1.3) / 1.3
        self.img.setImage(_img(img), autoLevels=False, levels=(0, 1))
        self.img.setRect(QRectF(0.0, float(fr[0]), S.shape[0] * hop_s, float(fr[-1] - fr[0])))
        self.p.setXRange(0, S.shape[0] * hop_s, padding=0)
        self.p.setYRange(float(fr[0]), float(fr[-1]), padding=0)
        c = f.get("cands") or []
        self.cand.setData([x[0] for x in c], [x[1] for x in c])
        self.note.setText(tr("t1.blocks_max_1f_threshold").format(K, float(S.max()), thr))


# ====================================================================== 동기 12음
class SyncTonesPanel(QWidget):
    def __init__(self):
        super().__init__()
        v = QVBoxLayout(self)
        v.setContentsMargins(T.sp(1), T.sp(1), T.sp(1), T.sp(1))
        self.note = QLabel("—")
        self.note.setObjectName("panelNote")
        v.addWidget(self.note)
        self.p = pg.PlotWidget()
        _style(self.p.getPlotItem(), tr("t1.symbol_in_block"), tr("t1.tone2"))
        self.p.setXRange(-1, C.nsym + 1, padding=0)
        self.p.setYRange(-0.5, 15.5, padding=0)
        v.addWidget(self.p, 1)
        pos, tone, _, _, _ = tng1.lay(C)
        self.pos, self.tone = pos, tone
        self.ref = pg.ScatterPlotItem(symbol="s", size=14, pen=pg.mkPen(T.c("line2")), brush=None)
        self.ref.setData(pos, tone)
        self.p.addItem(self.ref)
        self.sc = pg.ScatterPlotItem(symbol="s", size=12, pen=None)
        self.p.addItem(self.sc)
        self.txt = []

    def set_data(self, d):
        if not self.txt:                                      # 점수 글자 12개는 한 번 만들고 글만 바꿈 (갱신 12 → 수 ms)
            for x, y in zip(self.pos, self.tone):
                it = pg.TextItem("", color=T.c("text2"), anchor=(0.5, 1.2))
                it.setFont(T.mono_font(T.PT_SMALL))
                it.setPos(float(x), float(y))
                self.p.addItem(it)
                self.txt.append(it)
        if not d:
            self.sc.setData([], [])
            for it in self.txt:
                it.setText("")
            self.note.setText("—")
            return
        sc = d["score"]
        brushes = [pg.mkBrush(T.c("ok") if s >= 3.0 else (T.c("fec") if s >= 0.0 else T.c("err"))) for s in sc]
        self.sc.setData(self.pos, self.tone, brush=brushes)
        for it, s in zip(self.txt, sc):
            it.setText("{:+.0f}".format(s))
        self.note.setText(tr("t1.12_sync_tone_score").format(
            d.get("label", ""), 10 * math.log10(max(np.mean(10 ** (np.asarray(sc) / 10)), 1e-9))))


from tng1_app import sync_scores  # noqa: E402  (동기 12음 점수)


# ====================================================================== 해부 (HTML)
def _span(txt, col=None, bold=False):
    s = html.escape(txt)
    st = []
    if col:
        st.append("color:{}".format(T.c(col)))
    if bold:
        st.append("font-weight:600")
    return '<span style="{}">{}</span>'.format(";".join(st), s) if st else s


def _char_name(ch):
    return {"\n": "↵", " ": "␣"}.get(ch, ch)


def _h(n, title):
    return '<p style="margin:10px 0 4px 0; color:{}; font-weight:600">{} {}</p>'.format(T.c("text"), n, html.escape(title))


def _pre(body):
    return '<pre style="margin:0; font-family:{}; font-size:{}pt; color:{}">{}</pre>'.format(
        T.mono_font().family(), T.PT_BODY, T.c("text2"), body)


def huffman_rows(bits):
    """정보 비트 → [(부호 문자열, 글자 또는 None)] (EOT · 채움 포함)"""
    out, cur = [], ()
    for i, b in enumerate(bits):
        cur += (int(b),)
        ch = charset1._DEC.get(cur)
        if ch is not None:
            out.append(("".join(map(str, cur)), ch))
            cur = ()
            if ch == charset1.EOT:
                rest = "".join(str(int(x)) for x in bits[i + 1:])
                if rest:
                    out.append((rest, None))
                return out
    if cur:
        out.append(("".join(map(str, cur)), None))
    return out


def block_html(b):
    """수신 블록 해부. b: {"k", "abs", "f", "info", "E", "n_comb", "first", "text"}"""
    pos, tone, data_pos, _, ilv = tng1.lay(C)
    info = np.asarray(b["info"], int)
    first = bool(b.get("first"))
    syms = tng1.encode_block(C, list(info), first=first)
    E = np.asarray(b["E"]) if b.get("E") is not None else None
    am = E.argmax(1) if E is not None else None
    o = []
    o.append(_h("①", tr("t1.block")))
    o.append(_pre(tr("t1.block_b_start_2f").format(
                      b["k"] + 1, b.get("t_s", 0.0), b.get("f", 0.0),
                      "0x5A5A" if first else "0xFFFF", tr("t1.first_block") if first else tr("t1.next_block"),
                      tr("t1.combined_x").format(b["n_comb"]) if b.get("n_comb", 1) > 1 else tr("t1.single_decode"))))
    o.append(_h("②", tr("t1.92_symbols_s_sync")))
    rows, nfix = [], 0
    for r0 in range(0, C.nsym, 12):
        cells = []
        for j in range(r0, min(r0 + 12, C.nsym)):
            if j in set(pos.tolist()):
                cells.append(_span("S{:>2}   ".format(int(syms[j])), "wf_band"))
            elif am is not None and am[j] != syms[j]:
                nfix += 1
                cells.append(_span("{:>2}({:>2})".format(int(syms[j]), int(am[j])), "err"))
            else:
                cells.append("{:>3}   ".format(int(syms[j])))
        rows.append("{:>3} ".format(r0) + " ".join(cells))
    o.append(_pre("\n".join(rows)))
    if am is not None:
        o.append(_pre(tr("t1.strongest_cell_final_80").format(nfix)))
    o.append(_h("③", tr("t1.viterbi_bits_64_text")))
    crc = tng1.crc16_bits(list(info), tng1.crc_init(C, first))
    bstr = "".join(map(str, info))
    o.append(_pre(" ".join(bstr[i:i + 8] for i in range(0, 64, 8)) + "  |  " +
                  _span(" ".join("".join(map(str, crc))[i:i + 8] for i in range(0, 16, 8)), "ok")))
    o.append(_h("④", "CRC16"))
    o.append(_pre(tr("t1.received_computed").format("".join(map(str, crc)), _span(tr("t1.ok"), "ok", True))))
    o.append(_h("⑤", tr("t1.variable_length_text_decode")))
    lines = []
    for code, ch in huffman_rows(info):
        if ch is None:
            lines.append(tr("t1.14_padding_partial_code").format(code))
        elif ch == charset1.EOT:
            lines.append("{:<14} {}".format(code, _span(tr("t1.eot_end_of_message"), "fec")))
        else:
            lines.append("{:<14} {}".format(code, _span(_char_name(ch), "text")))
    o.append(_pre("\n".join(lines)))
    o.append(_pre(tr("t1.text") + _span(repr(b.get("text", "")), "ok")))
    return "".join(o)


class _HtmlPanel(QWidget):
    def __init__(self, note):
        super().__init__()
        v = QVBoxLayout(self)
        v.setContentsMargins(T.sp(1), T.sp(1), T.sp(1), T.sp(1))
        v.setSpacing(T.sp(1))
        h = QHBoxLayout()
        self.note = QLabel(note)
        self.note.setObjectName("panelNote")
        self.cb = QComboBox()
        self.cb.setMinimumWidth(160)
        h.addWidget(self.note, 1)
        h.addWidget(self.cb)
        v.addLayout(h)
        self.tb = QTextBrowser()
        self.tb.setOpenLinks(False)
        v.addWidget(self.tb, 1)


class Dissect1Panel(_HtmlPanel):
    """패킷 해부 (TNG1): 블록 고르기 → ①~⑤"""

    def __init__(self):
        super().__init__(tr("t1.no_rx_packet"))
        self.blocks = []
        self.cb.currentIndexChanged.connect(self._show)

    def set_packet(self, blocks, title="", select_abs=None):
        self.blocks = [b for b in blocks if b.get("info") is not None]
        self.cb.blockSignals(True)
        self.cb.clear()
        for b in self.blocks:
            self.cb.addItem("B{}{}".format(b["k"] + 1, tr("t1.x_combined").format(b["n_comb"]) if b.get("n_comb", 1) > 1 else ""))
        self.cb.blockSignals(False)
        self.note.setText(title or tr("t1.tng1_packet"))
        i = 0
        if select_abs is not None:
            d = [abs(b.get("abs_in", 0) - select_abs) for b in self.blocks]
            i = int(np.argmin(d)) if d else 0
        if self.blocks:
            self.cb.setCurrentIndex(i)
        self._show()

    def _show(self):
        i = self.cb.currentIndex()
        if not (0 <= i < len(self.blocks)):
            self.tb.setHtml(_pre(tr("t1.no_crc_ok_block")))
            return
        self.tb.setHtml(block_html(self.blocks[i]))


class TxDissect1Panel(_HtmlPanel):
    """송신 해부 (TNG1): 문자 → 가변 길이 비트 → 블록 + CRC16 → 부호화 → 16음 기호"""

    def __init__(self):
        super().__init__(tr("qso.tx_idle"))
        self.txa = None
        self.cb.currentIndexChanged.connect(self._show)

    def set_message(self, txa):
        self.txa = txa
        self.cb.blockSignals(True)
        self.cb.clear()
        for i, (s, t) in enumerate(txa.blocks):
            self.cb.addItem(tr("t1.b_chars").format(i + 1, len(t)))
        self.cb.blockSignals(False)
        self.note.setText(tr("t1.tng1_tx_chars_blocks").format(len(txa.text), len(txa.blocks), txa.dur))
        self._show()

    def set_current(self, k):
        if self.txa is not None and 0 <= k < self.cb.count() and k != self.cb.currentIndex():
            self.cb.setCurrentIndex(k)

    def _show(self):
        txa = self.txa
        i = self.cb.currentIndex()
        if txa is None or not (0 <= i < len(txa.blocks)):
            self.tb.setHtml(_pre("—"))
            return
        syms, text = txa.blocks[i]
        first = i == 0
        eot = i == len(txa.blocks) - 1
        bits, n, e = charset1.pack_block(text, C.KI, end=eot)
        o = [_h("①", tr("t1.text_variable_length_bits"))]
        lines, used = [], 0
        for ch in text:
            code = "".join(map(str, charset1.CODE[ch]))
            used += len(code)
            lines.append("{:<3} {}".format(_span(_char_name(ch), "text"), code))
        if e:
            code = "".join(map(str, charset1.CODE[charset1.EOT]))
            used += len(code)
            lines.append("{} {}".format(_span("EOT", "fec"), code))
        lines.append(tr("t1.padding_bits_1").format(C.KI - used))
        o.append(_pre("\n".join(lines)))
        o.append(_h("②", tr("t1.block_text_64_bits")))
        crc = tng1.crc16_bits(bits, tng1.crc_init(C, first))
        bstr = "".join(map(str, bits))
        o.append(_pre(" ".join(bstr[j:j + 8] for j in range(0, 64, 8)) + "  |  " +
                      _span(" ".join("".join(map(str, crc))[j:j + 8] for j in range(0, 16, 8)), "ok") +
                      tr("t1.crc16_init").format("0x5A5A" if first else "0xFFFF", tr("t1.first_block") if first else tr("t1.next_block"))))
        o.append(_h("③", tr("t1.coding_tail_biting_convolutional")))
        pos, tone, data_pos, _, ilv = tng1.lay(C)
        lab = syms[data_pos]
        o.append(_pre("\n".join("{:>3} ".format(r0) + " ".join("{:>2}".format(int(x)) for x in lab[r0:r0 + 20])
                                for r0 in range(0, len(lab), 20))))
        o.append(_h("④", tr("t1.92_16_tone_symbols")))
        rows = []
        for r0 in range(0, C.nsym, 12):
            cells = []
            for j in range(r0, min(r0 + 12, C.nsym)):
                cells.append(_span("S{:>2}".format(int(syms[j])), "wf_band") if j in set(pos.tolist())
                             else "{:>3}".format(int(syms[j])))
            rows.append("{:>3} ".format(r0) + " ".join(cells))
        o.append(_pre("\n".join(rows)))
        self.tb.setHtml("".join(o))
