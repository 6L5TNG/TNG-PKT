"""
TNG5 패킷 해부 (수신 로그에서 TNG5 행 클릭 → '패킷 해부' 창). 자료는 tng5_app.analyze 가 만든 dict.

  위   : 패킷 두 줄 (톤 · A×4 · 가드 · 프레임 · 가드 · B×4 / 18바이트 구간 CRC16), 구간 클릭 = 그 구간 선택
  가운데: 구간 표 (CRC16 · 길이 필드 · 비트가 실린 프레임 범위 · 평균 |LLR| · 정정 비트 · 글자)
  아래 : 선택 구간 층별 — ① 바이트 (길이 · 본문 · CRC) ② base-40 묶음 (2바이트 → 3글자) ③ 반복 사본 4개 |LLR|
         ④ 부호 비트 송신 위치 (인터리빙 뒤 프레임), 프레임별 평균 |LLR|
"""
import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel, QTableWidget, QTableWidgetItem,
                               QHeaderView, QSplitter, QPlainTextEdit, QAbstractItemView)

from tngpkt import theme as T
from tngpkt.widgets import TimelinePanel
from tngpkt.strings_ko import tr, trf


def trf_fmt(f):
    """조각 번역을 틀에 먼저 (사용자 글 안의 글자는 건드리지 않게)"""
    return trf(f)


class Dissect5Panel(QWidget):
    COLS = ["구간", "CRC16", "글자", "프레임", "평균 |LLR|", "정정", "바이트 (16진)", "내용"]

    def __init__(self):
        super().__init__()
        self.d = None
        v = QVBoxLayout(self)
        v.setContentsMargins(6, 6, 6, 6)
        v.setSpacing(6)
        self.title = QLabel(trf("TNG5 패킷 해부"))
        self.title.setObjectName("panelNote")
        v.addWidget(self.title)
        self.tl = TimelinePanel("패킷 없음")
        self.tl.setMinimumHeight(96)
        self.tl.setMaximumHeight(120)
        v.addWidget(self.tl)
        sp = QSplitter(Qt.Vertical)
        self.table = QTableWidget(0, len(self.COLS))
        self.table.setHorizontalHeaderLabels([tr(k) if k else c for c, k in zip(self.COLS, ("dis5.col.seg", None, "dis5.col.chars", "dis5.col.frames", "dis5.col.llr", "dis5.col.fixed", "dis5.col.hex", "dis5.col.text"))])
        self.table.verticalHeader().setVisible(False)
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SingleSelection)
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        hh = self.table.horizontalHeader()
        for i in range(len(self.COLS) - 1):
            hh.setSectionResizeMode(i, QHeaderView.ResizeToContents)
        hh.setSectionResizeMode(len(self.COLS) - 1, QHeaderView.Stretch)
        self.table.itemSelectionChanged.connect(self._sel)
        sp.addWidget(self.table)
        low = QWidget()
        h = QHBoxLayout(low)
        h.setContentsMargins(0, 0, 0, 0)
        self.layers = QPlainTextEdit()
        self.layers.setReadOnly(True)
        self.layers.setFont(T.mono_font(T.PT_SMALL))
        h.addWidget(self.layers, 3)
        g = pg.GraphicsLayoutWidget()
        g.setBackground(T.c("bg1"))
        self.p_rep = g.addPlot(row=0, col=0, title="③ 반복 사본별 평균 |LLR|")
        self.p_pos = g.addPlot(row=1, col=0, title="④ 부호 비트 송신 위치 (프레임)")
        self.p_fr = g.addPlot(row=2, col=0, title="프레임별 평균 |LLR|")
        for p in (self.p_rep, self.p_pos, self.p_fr):
            p.showGrid(x=True, y=True, alpha=0.15)
            p.setMenuEnabled(False)
        h.addWidget(g, 4)
        sp.addWidget(low)
        sp.setSizes([220, 320])
        v.addWidget(sp, 1)
        self.tl.frameClicked.connect(lambda f: None)

    # ---------------------------------------------------------------- 표시
    def show_packet(self, d, title=""):
        self.d = d
        self.table.setRowCount(0)
        self.layers.clear()
        for p in (self.p_rep, self.p_pos, self.p_fr):
            p.clear()
        if d is None:
            self.title.setText(trf("TNG5 패킷 해부 · 자료 없음 · {}".format(title)))
            self.tl.show_packet(None)
            return
        segs = d["segments"]
        n_ok = sum(1 for r in segs if r["ok"])
        self.title.setText(trf_fmt("TNG5 · {} · 프레임 {} · 구간 {}/{} 통과 · 글자 수 {} · Δf {:+.1f} Hz{}".format(
            title, d["n_frames"], n_ok, len(segs), d["chars"] if d["chars"] is not None else "모름",
            d["df"] if d["df"] is not None and np.isfinite(d["df"]) else 0.0, "" if d["has_b"] else " · B 없음")))
        self.tl.show_packet(d["timeline"], d.get("frame_err"), d.get("frame_llr"))
        self.table.setRowCount(len(segs))
        for i, r in enumerate(segs):
            vals = ["S{}".format(r["k"] + 1), "통과" if r["ok"] else "실패",
                    "24" if r["k"] == 0 else "27", "F{}–F{}".format(*r["frames"]), "{:.2f}".format(r["llr_abs"]),
                    "-" if r["err"] is None else str(r["err"]),
                    " ".join("{:02X}".format(x) for x in r["data"]), r["text"] or "□"]
            for j, v in enumerate(vals):
                it = QTableWidgetItem(v)
                if j == 1:
                    it.setForeground(pg.mkColor(T.c("ok" if r["ok"] else "err")))
                if j == 6:
                    it.setFont(T.mono_font(T.PT_SMALL))
                self.table.setItem(i, j, it)
        fl = np.asarray(d["frame_llr"])
        self.p_fr.addItem(pg.BarGraphItem(x=np.arange(1, len(fl) + 1), height=fl, width=0.8, brush=T.brush("rx_dim"),
                                          pen=None))
        if segs:
            self.table.selectRow(0)

    def _sel(self):
        d = self.d
        rows = self.table.selectionModel().selectedRows() if d is not None else []
        if not rows:
            return
        r = d["segments"][rows[0].row()]
        k = r["k"]
        data = r["data"]
        lines = ["구간 S{}  ({})".format(k + 1, "CRC16 통과" if r["ok"] else "CRC16 실패"),
                 "",
                 "① 바이트 18 + CRC16 2 (계산 CRC {:04X})".format(r["crc_calc"])]
        if k == 0:
            lines.append("   길이 필드  {:02X} {:02X}  → {}구간".format(data[0], data[1], r["len_field"]))
        lines.append("   본문       " + " ".join("{:02X}".format(x) for x in (data[2:] if k == 0 else data)))
        lines += ["", "② 글자 부호 (charset1 가변 길이, 구간마다 글자 단위 · 나머지 1 채움, 마지막 구간 EOT)"]
        for i, (code, nb, s) in enumerate(r["groups"]):
            lines.append("   {:>2}  {:<9}  {}비트  → '{}'".format(i + 1, code, nb, "↵" if s == "\n" else s))
        lines += ["", "③ 부호: K=7 R1/4 (생성 117 127 155 171) 꼬리 6 → 664비트 × 4회 반복 = 2656비트",
                  "   반복 사본별 평균 |LLR|: " + " · ".join("{:.2f}".format(x) for x in r["rep_llr"]),
                  "   비터비 입력 = 사본 4개 LLR 합",
                  "", "④ 인터리버: 구간 안 무작위 + 지연 (최대 약 9.6 s) → 프레임 F{}–F{}".format(*r["frames"])]
        if r["err"] is not None:
            lines.append("   hard 판정 오류 (정정된 부호 비트) {} / 2656".format(r["err"]))
        self.layers.setPlainText("\n".join(lines))
        self.p_rep.clear()
        self.p_rep.addItem(pg.BarGraphItem(x=np.arange(1, 5), height=r["rep_llr"], width=0.6,
                                           brush=T.brush("ok" if r["ok"] else "err"), pen=None))
        self.p_pos.clear()
        pos = np.asarray(d["coded_pos"][k])
        self.p_pos.addItem(pg.ScatterPlotItem(pos / 96.0 + 1, (pos % 96) / 96.0, size=3, pen=None,
                                              brush=T.brush("rx_dim")))
