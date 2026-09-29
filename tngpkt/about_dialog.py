"""TNG PKT About 창 (영어 고정). 내용은 registry 한 곳에서 모은다. 버전 숫자는 고정폭 글꼴."""
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QColor, QGuiApplication
from PySide6.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QWidget, QScrollArea,
                               QTableWidget, QTableWidgetItem, QHeaderView, QAbstractItemView, QSizePolicy)

from tngpkt import registry
from tngpkt import theme as T


def about_text():
    """버그 신고용 평문 (Copy Version Info 와 같음)"""
    return registry.version_report()


class AboutDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        r = registry
        self.setWindowTitle("About " + r.APP_NAME)
        self.resize(760, 700)
        v = QVBoxLayout(self)
        v.setContentsMargins(T.sp(5), T.sp(5), T.sp(5), T.sp(4))
        v.setSpacing(T.sp(3))

        # ---- 머리: 이름 · 버전 · 채널 · 빌드 날짜 · 만든 이
        head = QHBoxLayout()
        head.setSpacing(T.sp(3))
        col = QVBoxLayout()
        col.setSpacing(T.sp(1))
        name = QLabel(r.APP_NAME)
        name.setStyleSheet("color: {}; font-size: 20pt; font-weight: bold;".format(T.c("text")))
        col.addWidget(name)
        row = QHBoxLayout()
        row.setSpacing(T.sp(2))
        ver = QLabel("Version {} (Build {})".format(r.APP_VERSION, r.APP_BUILD))
        ver.setFont(T.mono_font(T.PT_BODY))
        ver.setStyleSheet("color: {};".format(T.c("text")))
        ver.setTextInteractionFlags(Qt.TextSelectableByMouse)
        ch = QLabel(r.CHANNEL)
        ch.setFont(T.ui_font(T.PT_SMALL, True))
        ch.setStyleSheet("color: {a}; border: 1px solid {a}; border-radius: {r}px; padding: 0 {p}px;".format(
            a=T.c("accent"), r=T.RADIUS, p=T.sp(1)))
        row.addWidget(ver)
        row.addWidget(ch)
        row.addStretch(1)
        col.addLayout(row)
        sub = QLabel("Built {} · by {}".format(r.BUILD_DATE, r.AUTHOR))
        sub.setFont(T.ui_font(T.PT_SMALL))
        sub.setStyleSheet("color: {};".format(T.c("text2")))
        col.addWidget(sub)
        head.addLayout(col, 1)
        v.addLayout(head)
        v.addWidget(self._hline())

        # ---- 표 3개 (스크롤 한 번에)
        body = QWidget()
        body.setObjectName("aboutBody")
        body.setStyleSheet("QWidget#aboutBody {{ background: {}; }}".format(T.c("bg0")))
        bv = QVBoxLayout(body)
        bv.setContentsMargins(0, 0, T.sp(1), 0)
        bv.setSpacing(T.sp(2))
        rows_p = []
        for p in r.PROTOCOLS:
            st = p.status
            rows_p.append([(p.name, None, None), (p.version + " " + r.CHANNEL, "mono", None), (str(p.wire_rev) if p.wire_rev else "—", "mono", None),
                           (p.model_id, "mono", p.model_hash or None), (st, None, "ok" if st == "Active" else "text3")])
        self.t_proto = self._table(["Name", "Version", "Wire rev", "Model ID", "Status"], rows_p, stretch=0)
        self.t_proto.cellDoubleClicked.connect(self._copy_hash)
        for i, p in enumerate(r.PROTOCOLS):
            if p.spec is not None:
                pass
        bv.addWidget(self._section("Protocols"))
        bv.addWidget(self.t_proto)
        self.lb_hint = QLabel("Model ID: first 8 hex digits of SHA-256. Double-click to copy the full hash.")
        self.lb_hint.setObjectName("note")
        bv.addWidget(self.lb_hint)

        rows_a = []
        for m in r.AI_MODELS:
            ver = m.version + (" " + r.CHANNEL if m.version != "—" else "")
            mid = m.model_id if m.ok else m.model_id + " ≠ " + (m.actual[:8] or "missing")
            rows_a.append([(m.name, None, m.path or ("57 chars + EOT, canonical Huffman (charset1.py)" if m.role == "Charset" else None)),
                           (ver, "mono", None), (m.used_by, "mono", None), (m.role, None, None),
                           (mid, "mono", (m.expected or None) if m.ok else "err")])
        self.t_ai = self._table(["Name", "Version", "Used by", "Role", "Model ID"], rows_a, stretch=0)
        self.t_ai.cellDoubleClicked.connect(self._copy_ai_hash)
        bv.addWidget(self._section("AI Models"))
        bv.addWidget(self.t_ai)

        rows_m = [[(m.name, None, m.module), (m.version + " " + r.CHANNEL, "mono", None),
                   ("{:<5} {}".format(m.protocol, m.protocol_versions.replace(">=", "≥").replace(",", ", ")), "mono", None)] for m in r.EXPERT_MODULES]
        bv.addWidget(self._section("Expert Modules"))
        bv.addWidget(self._table(["Name", "Version", "Compatible with"], rows_m, stretch=0))

        rows_l = [[(t, None, None), (ver_, "mono", None), (lic, None, None)] for t, ver_, lic in r.library_versions()]
        bv.addWidget(self._section("Third-party Libraries"))
        bv.addWidget(self._table(["Name", "Version", "License"], rows_l, stretch=0))
        bv.addStretch(1)
        sc = QScrollArea()
        sc.setWidgetResizable(True)
        sc.setFrameShape(QScrollArea.NoFrame)
        sc.setWidget(body)
        sc.viewport().setStyleSheet("background: {};".format(T.c("bg0")))
        v.addWidget(sc, 1)
        v.addWidget(self._hline())

        # ---- 아래: 설정 위치 · 복사 · 닫기
        foot = QHBoxLayout()
        loc = QLabel("Settings and logs: %APPDATA%\\" + r.APP_NAME)
        loc.setObjectName("note")
        loc.setStyleSheet("font-family: '{}';".format(T.mono_font().family()))   # 맑은 고딕은 역슬래시를 ₩ 로 그린다
        foot.addWidget(loc)
        foot.addStretch(1)
        self.b_copy = QPushButton("Copy Version Info")
        self.b_copy.clicked.connect(self._copy_all)
        b_close = QPushButton("Close")
        b_close.setDefault(True)
        b_close.clicked.connect(self.accept)
        foot.addWidget(self.b_copy)
        foot.addWidget(b_close)
        v.addLayout(foot)

    # ------------------------------------------------------------------ 부품
    @staticmethod
    def _hline():
        w = QWidget()
        w.setObjectName("hline")
        return w

    @staticmethod
    def _section(text):
        lb = QLabel(text)
        lb.setFont(T.ui_font(T.PT_BODY, True))
        lb.setStyleSheet("color: {}; padding-top: {}px;".format(T.c("text2"), T.sp(1)))
        return lb

    @staticmethod
    def _table(headers, rows, stretch=0):
        """읽기 전용 표 (내용 높이에 맞춤, 안쪽 스크롤 없음). 칸 = (글자, 'mono'|None, 도움말 또는 색 이름)"""
        t = QTableWidget(len(rows), len(headers))
        t.setHorizontalHeaderLabels(headers)
        t.verticalHeader().setVisible(False)
        t.setEditTriggers(QAbstractItemView.NoEditTriggers)
        t.setSelectionBehavior(QAbstractItemView.SelectRows)
        t.setSelectionMode(QAbstractItemView.SingleSelection)
        t.setFocusPolicy(Qt.NoFocus)
        t.setShowGrid(False)
        t.setWordWrap(False)
        t.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        t.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        hh = t.horizontalHeader()
        hh.setDefaultAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        hh.setHighlightSections(False)
        mono, ui = T.mono_font(T.PT_BODY), T.ui_font(T.PT_BODY)
        for i, cells in enumerate(rows):
            for j, (txt, kind, extra) in enumerate(cells):
                it = QTableWidgetItem(txt)
                it.setFont(mono if kind == "mono" else ui)
                it.setTextAlignment(Qt.AlignLeft | Qt.AlignVCenter)
                if extra in T.PALETTE:
                    it.setForeground(QColor(T.c(extra)))
                elif extra:
                    it.setToolTip(extra)
                t.setItem(i, j, it)
            t.setRowHeight(i, T.CTRL_H)
        for j in range(len(headers)):
            hh.setSectionResizeMode(j, QHeaderView.Stretch if j == stretch else QHeaderView.ResizeToContents)
        t.setMinimumWidth(0)
        t.setFixedHeight(hh.sizeHint().height() + T.CTRL_H * len(rows) + 2)
        t.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        return t

    # ------------------------------------------------------------------ 복사
    def _flash(self, btn_or_label, text, back):
        btn_or_label.setText(text)
        QTimer.singleShot(1500, lambda: btn_or_label.setText(back))

    def _copy_all(self):
        QGuiApplication.clipboard().setText(about_text())
        self._flash(self.b_copy, "Copied", "Copy Version Info")

    def _copy_ai_hash(self, row, col):
        m = registry.AI_MODELS[row]
        if col == 4 and m.expected:
            QGuiApplication.clipboard().setText(m.expected)
            self._flash(self.lb_hint, "Copied {} SHA-256: {}".format(m.name, m.expected),
                        "Model ID: first 8 hex digits of SHA-256. Double-click to copy the full hash.")

    def _copy_hash(self, row, col):
        p = registry.PROTOCOLS[row]
        if col == 3 and p.model_hash:
            QGuiApplication.clipboard().setText(p.model_hash)
            self._flash(self.lb_hint, "Copied {} model SHA-256: {}".format(p.name, p.model_hash),
                        "Model ID: first 8 hex digits of SHA-256. Double-click to copy the full hash.")
