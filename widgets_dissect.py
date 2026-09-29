"""
패킷 해부 뷰 — 수신 패킷 하나를 처리 단계별(층별)로 펼친다. 계산은 dissect.py (분석 스레드), 여기서는 그리기만.

  왼쪽: 트리 요약 (층별 핵심 수치)
  오른쪽: 8층, 가로축은 모두 같은 시간축 (톤 시작 = 0 s), 왼쪽 축 폭 고정으로 픽셀 단위 정렬
      1 오디오   2 기저대역 I/Q   3 구간   4 수신 NN LLR (송신 순서, 심볼마다 비트 2개)
      5 디인터리빙 후 부호 비트   6 비터비 출력 정보 비트   7 CRC · 프레임별 오류   8 바이트 → 텍스트
      (5 · 6 · 8 은 인터리빙 뒤라 시간 순서가 없다 → 데이터 구간 폭에 부호어 순서대로 펼친다)
  7단계 스트리밍 형식은 5 컨볼루션 디인터리빙, 6 비터비(제한 깊이 48), 7 32바이트 구간별 CRC16 으로 층 이름이 바뀐다.
  어느 층이든 클릭 → 인터리버 치환 · 부호율 1/2 · 8비트 묶음을 따라 모든 층의 대응 위치를 강조
"""

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import Qt, QRectF, QPointF, Signal
from PySide6.QtGui import QPolygonF, QPainterPath, QTransform, QImage, QPainter, QPen
import threading
from PySide6.QtCore import QTimer
from PySide6.QtWidgets import (QWidget, QHBoxLayout, QVBoxLayout, QSplitter, QTreeWidget, QTreeWidgetItem, QGraphicsItem,
                               QPushButton, QSpinBox, QLabel)
from PySide6.QtGui import QColor, QKeySequence, QShortcut

import theme as T
from widgets import FitAxis, _lab, ElideLabel
from dissect import links_from_tx, links_from_coded, links_from_info, links_from_byte
from strings_ko import tr, trf

# (키, 층 제목, 그래프 최소 높이) — 제목은 그래프 바깥 위쪽 (PlotItem 제목 줄)
# 7단계 스트리밍 형식에서 바뀌는 층 제목
LAYERS_STREAM = {"segs": "3  구간 (프레임 색: 확인 가능한 비트 기준 정정 여부)",
                 "llr": "4  수신 NN 출력 LLR (송신 순서)",
                 "coded": "5  컨볼루션 디인터리빙 후 부호 비트 (부호어 순서)",
                 "info": "6  비터비 출력 정보 비트 (제한 깊이 48)",
                 "crc": "7  32바이트 구간별 CRC16",
                 "bytes": "8  바이트 · 텍스트"}
LAYERS_TNG5 = {"segs": "3  구간 (시작 동기 · 20프레임마다 중간 동기 M · B×1)",
               "llr": "4  수신 NN 출력 LLR (송신 순서, 중간 동기 틈 포함)",
               "coded": "5  디인터리빙 + 반복 4회 결합 (구간마다 R1/4 부호어 664비트)",
               "info": "6  비터비 출력 정보 비트 (구간마다 18바이트 + CRC16 + 꼬리 6)",
               "crc": "7  18바이트 구간별 CRC16 (S1 앞 2바이트 = 길이 필드)",
               "bytes": "8  바이트 · 글자"}
LAYERS = [("audio", "1  수신 원본 파형", 96),
          ("base", "2  기저대역 I/Q (주파수 보정 후)", 96),
          ("segs", "3  구간", 64),
          ("llr", "4  수신 NN 출력 LLR", 72),
          ("coded", "5  디인터리빙 후 부호 비트", 64),
          ("info", "6  비터비 출력 정보 비트", 64),
          ("crc", "7  CRC · 프레임별 오류 비트", 64),
          ("bytes", "8  바이트 (16진수)", 72)]


def _diverging_lut(n=256):
    """LLR 색: 음수(0) 분홍 ← 0 어두움 → 양수(1) 파랑"""
    from PySide6.QtGui import QColor
    a = np.array(QColor(T.c("cat1")).getRgb()[:3], float)
    m = np.array(QColor(T.c("bg2")).getRgb()[:3], float)
    b = np.array(QColor(T.c("cat0")).getRgb()[:3], float)
    t = np.linspace(-1, 1, n)[:, None]
    lut = np.where(t < 0, m + (a - m) * (-t), m + (b - m) * t)
    return lut.astype(np.uint8)


def env_geometry(vb, view):
    """오디오 층 ViewBox 의 기하 (크기 · 화면 변환 · 그리기 설정). 화면 스레드에서 떠 둔다"""
    return {"rect": QRectF(vb.rect()), "dev": QTransform(vb.deviceTransform(view.viewportTransform())), "hints": view.renderHints(),
            "dpr": float(view.devicePixelRatioF()), "yinv": vb.state["yInverted"], "xinv": vb.state["xInverted"]}


def env_full_transform(g, xr, yr):
    """ViewBox.updateMatrix 와 같은 순서의 계산 → 채움 항목 좌표 → 화면 좌표 (보기 범위 xr · yr)"""
    b = g["rect"]
    vr = QRectF(xr[0], yr[0], xr[1] - xr[0], yr[1] - yr[0])
    sx, sy = b.width() / vr.width(), b.height() / vr.height()
    if not g["yinv"]:
        sy = sy * -1
    if g["xinv"]:
        sx = sx * -1
    m = QTransform()
    c = b.center()
    m.translate(c.x(), c.y())
    m.scale(sx, sy)
    vc = vr.center()
    m.translate(-vc.x(), -vc.y())
    return m * g["dev"]


def env_fill_path(a):
    """채움 경로 — 화면과 같은 코드 (PlotDataItem 곡선 → FillBetweenItem.updatePath). 작업 스레드에서 한 번"""
    c1, c2 = pg.PlotDataItem(a["t"], a["lo"]), pg.PlotDataItem(a["t"], a["hi"])
    return pg.FillBetweenItem(c1, c2).path()


def render_env_full(a, g, brush, xr, yr, path=None):
    """
    작업 스레드: 오디오 채움의 전체 보기 그림을 미리 그린다.
    그리는 방식은 화면과 같다 — 같은 곡선 경로(PlotDataItem) · 같은 채움 경로(FillBetweenItem) · 같은 변환 · 같은 그리기 설정,
    투명 이미지에 그린 뒤 합성 (= Qt 항목 캐시가 하는 방식, 원래 직접 그리기와 0화소 차이로 확인됨).
    """
    if g is None or g["dpr"] != 1.0:
        return None                                        # 배율 화면은 원래 경로로 (정렬 보장 못 함)
    D = env_full_transform(g, xr, yr)
    if path is None:
        path = env_fill_path(a)
    r = D.mapRect(path.boundingRect()).toAlignedRect() & g["dev"].mapRect(g["rect"]).toAlignedRect().adjusted(-2, -2, 2, 2)
    if r.isEmpty():
        return None
    img = QImage(r.size(), QImage.Format_ARGB32_Premultiplied)
    img.fill(0)
    p = QPainter(img)
    p.setRenderHints(g["hints"], True)
    p.setTransform(D * QTransform.fromTranslate(-r.x(), -r.y()))
    p.setPen(QPen(Qt.NoPen))
    p.setBrush(brush)
    p.drawPath(path)
    p.end()
    return {"D": D, "img": img, "x": r.x(), "y": r.y()}


def _same_linear(A, B):
    return all(abs(u - v) <= 1e-9 * max(1.0, abs(v)) for u, v in
               ((A.m11(), B.m11()), (A.m12(), B.m12()), (A.m21(), B.m21()), (A.m22(), B.m22())))


class _EnvFill(pg.FillBetweenItem):
    _given = None                                    # 작업 스레드가 미리 만든 같은 채움 경로 (있으면 다시 만들지 않음)

    _defer = False                                   # 만들 때 경로를 미루기 (전체 보기에서 처음 필요할 때 같은 코드로 만든다)

    def updatePath(self):
        if _EnvFill._given is not None:
            self.setPath(_EnvFill._given)
            return
        if _EnvFill._defer:
            return
        super().updatePath()

    def boundingRect(self):
        if getattr(self, "_deferred", False):        # 경로 = lo · hi 꼭짓점 다각형 → 경계 = 데이터 범위 (펜 없음)
            return self._br
        return super().boundingRect()

    """
    오디오 1 ms 최소/최대 사이 채움. 확대 상태에서는 보이는 가로 범위(+양쪽 4화소 · 2점)의 다각형만 채운다.
    잘라낸 부분과 닫는 변은 전부 보이는 범위 밖에 있어서 범위 안 화소의 짝홀 판정 · 경계선은 원래 다각형과 같다.
    (전체 다각형 약 15만 꼭짓점을 매번 채우던 것이 해부 창 다시 그리기 시간의 대부분이었다)
    """

    def __init__(self, c1, c2, t, lo, hi, vb, **kw):
        super().__init__(c1, c2, **kw)
        self._t, self._lo, self._hi, self._vb = t, lo, hi, vb
        self._clip = (None, None)
        self.pre = None                                      # render_env_full 결과 (전체 보기 그림)
        self.pre_hits = 0

    def paint(self, p, opt, widget=None):
        vb = self._vb
        t = self._t
        if vb is None or len(t) < 4:
            return super().paint(p, opt, widget)
        x0, x1 = vb.viewRange()[0]
        m = 4.0 * (x1 - x0) / max(vb.width(), 1.0)                  # 닫는 변이 경계 화소 안티앨리어싱에 닿지 않게 4화소 여유
        i0 = max(int(np.searchsorted(t, x0 - m)) - 2, 0)
        i1 = min(int(np.searchsorted(t, x1 + m)) + 2, len(t))
        if i0 == 0 and i1 == len(t):
            if getattr(self, "_deferred", False):
                self._deferred = False
                super().updatePath()
            pre = self.pre
            if pre is not None:                              # 미리 그린 전체 보기: 변환이 정확히 같을 때만 (정수 화소 이동)
                T_ = p.transform()
                D = pre["D"]
                dx, dy = T_.dx() - D.dx(), T_.dy() - D.dy()
                if _same_linear(T_, D) and abs(dx - round(dx)) < 1e-6 and abs(dy - round(dy)) < 1e-6:
                    p.save()
                    p.setTransform(QTransform.fromTranslate(pre["x"] + round(dx), pre["y"] + round(dy)))
                    p.drawImage(0, 0, pre["img"])
                    p.restore()
                    self.pre_hits += 1
                    return
            return super().paint(p, opt, widget)
        key, path = self._clip
        if key != (i0, i1):
            x = np.concatenate([t[i0:i1], t[i0:i1][::-1]])
            y = np.concatenate([self._lo[i0:i1], self._hi[i0:i1][::-1]])
            poly = QPolygonF([QPointF(a, b) for a, b in zip(x.tolist(), y.tolist())])
            path = QPainterPath()
            path.setFillRule(self.fillRule())
            path.addPolygon(poly)
            self._clip = ((i0, i1), path)
        p.setPen(self.pen())
        p.setBrush(self.brush())
        p.drawPath(path)


OV_H = 30          # 개요 막대 높이 (위 줄 = 프레임, 아래 줄 = 32바이트 구간 CRC)


def render_overview(d, w, dpr):
    """
    개요 막대 그림 한 장 (작업 스레드에서 호출: QImage · QPainter 만 사용).
    위 줄: 톤 · A · 가드 · 프레임 (정정 여부 색) · B,  아래 줄: 구간별 CRC16 통과 / 실패 / 미확정, ▼ = 재탐색
    가로 = 0 (톤 시작) ~ 패킷 끝
    """
    w = max(int(w), 10)
    img = QImage(int(round(w * dpr)), int(round(OV_H * dpr)), QImage.Format_ARGB32_Premultiplied)
    img.setDevicePixelRatio(dpr)
    img.fill(QColor(T.c("plot")))
    t_end = max(float(d["t_end"]), 0.1)
    X = lambda t: t / t_end * w
    p = QPainter(img)
    p.setPen(Qt.NoPen)
    stream = d.get("format") == "스트리밍"
    col = {"tone": "tx_dim", "costas": "rx_dim", "guard": "bg4", "missing": "err_dim"}
    ff = d["frame_fixed"]
    fi = 0
    for name, a0, a1, kind in d["segs"]:
        c = col.get(kind)
        if kind == "frame":
            c = "fec" if ff[fi] > 0 else ("bg4" if stream and d["frame_known"][fi] == 0 else "ok")
            fi += 1
        p.fillRect(QRectF(X(a0), 2, max(X(a1) - X(a0), 0.6), 11), QColor(T.c(c)))
    if stream:
        t0, t1 = d["t_data"]
        ni, dur = d["n_info"], t1 - t0
        for r in d["segments"]:
            a, e = r["info"]
            x0, x1 = X(t0 + a / ni * dur), X(t0 + e / ni * dur)
            c = "ok" if r["ok"] else ("err" if r["decided"] else "bg3")
            p.fillRect(QRectF(x0, 16, max(x1 - x0 - 1, 0.6), 12), QColor(T.c(c)))
            if r["relock"]:
                m = (x0 + x1) / 2
                p.setBrush(QColor(T.c("fec")))
                p.drawPolygon(QPolygonF([QPointF(m - 3, 14), QPointF(m + 3, 14), QPointF(m, 19)]))
                p.setBrush(Qt.NoBrush)
    p.end()
    return img


class OverviewBar(QWidget):
    """개요 막대: 미리 그린 그림 + 현재 상세 범위 테두리 + 층 클릭 대응 위치 눈금. 클릭 · 끌기 = 상세 위치 이동"""
    seek = Signal(float)

    def __init__(self):
        super().__init__()
        self.setFixedHeight(OV_H)
        self.img, self.t_end, self.view, self.marks = None, 1.0, None, []

    def set_image(self, img, t_end):
        self.img, self.t_end = img, max(float(t_end), 0.1)
        self.update()

    def set_view(self, a, b):
        self.view = (a, b)
        self.update()

    def set_marks(self, ts):
        self.marks = list(ts)
        self.update()

    def paintEvent(self, ev):
        p = QPainter(self)
        p.fillRect(self.rect(), QColor(T.c("plot")))
        if self.img is not None:
            p.drawImage(0, 0, self.img)
        w = self.width()
        X = lambda t: t / self.t_end * w
        p.setPen(QPen(QColor(T.c("text")), 1))
        for t in self.marks:
            x = X(t)
            p.drawLine(QPointF(x, 0), QPointF(x, 3))
            p.drawLine(QPointF(x, OV_H - 3), QPointF(x, OV_H))
        if self.view is not None:
            a, b = X(self.view[0]), X(self.view[1])
            p.setPen(QPen(QColor(T.c("text")), 1.5))
            p.setBrush(Qt.NoBrush)
            p.drawRect(QRectF(a + 0.75, 0.75, max(b - a - 1.5, 2), OV_H - 1.5))
        p.end()

    def _emit(self, ev):
        self.seek.emit(ev.position().x() / max(self.width(), 1) * self.t_end)

    def mousePressEvent(self, ev):
        self._emit(ev)

    def mouseMoveEvent(self, ev):
        if ev.buttons() & Qt.LeftButton:
            self._emit(ev)


class DissectPanel(QWidget):
    PAGE_SEGS = 1          # 상세 보기 기본 구간 수 (근거: README — 8층 16진수 라벨 칸 20화소 조건 + 저사양 측정)
    selected = Signal(object)
    _preReady = Signal(object)

    def __init__(self):
        super().__init__()
        v = QVBoxLayout(self)
        v.setContentsMargins(T.sp(1), T.sp(1), T.sp(1), T.sp(1))
        v.setSpacing(T.sp(1))
        self.head = ElideLabel(trf("패킷 선택 대기 (수신 로그)"))
        self.head.setObjectName("dim")
        hr = QHBoxLayout()
        hr.setContentsMargins(0, 0, 0, 0)
        hr.addWidget(self.head, 1)
        self.btn_prev = QPushButton(tr("dis.prev"))
        self.btn_prev.clicked.connect(lambda: self.go_page(self.page - 1))
        self.btn_next = QPushButton(tr("dis.next"))
        self.btn_next.clicked.connect(lambda: self.go_page(self.page + 1))
        self.lb_page = QLabel("—")
        self.lb_page.setObjectName("num")
        self.sp_n = QSpinBox()
        self.sp_n.setRange(1, 64)
        self.sp_n.setValue(self.PAGE_SEGS)
        self.sp_n.setPrefix("구간 ")
        self.sp_n.setSuffix("개씩")
        self.sp_n.valueChanged.connect(self._set_n)
        self.btn_fit = QPushButton(tr("dis.to_start"))
        self.btn_fit.clicked.connect(self.fit_all)
        for w_ in (self.btn_prev, self.lb_page, self.btn_next, self.sp_n, self.btn_fit):
            hr.addWidget(w_)
        v.addLayout(hr)
        self.ov = OverviewBar()
        self.ov.seek.connect(self.seek_time)
        v.addWidget(self.ov)
        for key_, f_ in ((Qt.Key_Left, lambda: self.go_page(self.page - 1)),
                         (Qt.Key_Right, lambda: self.go_page(self.page + 1))):
            sc = QShortcut(QKeySequence(key_), self)
            sc.setContext(Qt.WidgetWithChildrenShortcut)
            sc.activated.connect(f_)
        self.page = 0
        self._pages = [(0.0, 1.0)]
        sp = QSplitter(Qt.Horizontal)
        self.tree = QTreeWidget()
        self.tree.setColumnCount(2)
        self.tree.setHeaderLabels(["층 · 항목", "값"])
        self.tree.setObjectName("dissectTree")
        self.tree.setMinimumWidth(300)
        self.tree.setColumnWidth(0, 150)
        self.tree.setIndentation(T.sp(4))
        self.tree.setRootIsDecorated(True)
        self.tree.setUniformRowHeights(True)
        sp.addWidget(self.tree)
        self.gl = pg.GraphicsLayoutWidget()
        self.gl.setBackground(T.c("plot"))
        self.gl.ci.layout.setSpacing(T.sp(2))              # 층 사이 간격
        sp.addWidget(self.gl)
        sp.setStretchFactor(1, 1)
        sp.setSizes([320, 900])
        v.addWidget(sp, 1)
        self.p = {}
        first = None
        for i, (key, title, h) in enumerate(LAYERS):
            last = i == len(LAYERS) - 1
            pl = self.gl.addPlot(row=i, col=0, axisItems={"left": FitAxis("left"), "bottom": FitAxis("bottom")})
            T.style_plot(pl, None, "시간 [s, 0 = 톤 시작]" if last else None, None, grid=False)
            pl.getAxis("left").setWidth(T.AXIS_W)
            pl.getAxis("left").setStyle(showValues=key in ("audio", "base"))
            if not last:
                pl.getAxis("bottom").setStyle(showValues=False)
                pl.getAxis("bottom").setHeight(2)
            pl.setMinimumHeight(h)
            pl.setMouseEnabled(x=True, y=False)
            pl.hideButtons()
            self._set_title(pl, title)
            if first is None:
                first = pl
            else:
                pl.setXLink(first)
            self.p[key] = pl
        self.gl.scene().sigMouseClicked.connect(self._click)
        self.p["bytes"].getViewBox().sigXRangeChanged.connect(self._view_changed)
        self.d = None
        self._items = []
        self._hl = []
        self._lut = _diverging_lut()
        self._fill = None
        self._pend = None
        self.pend_max = 0.0
        self._pend_timer = QTimer(self)
        self._pend_timer.setInterval(0)
        self._pend_timer.timeout.connect(self._pend_tick)
        self._preReady.connect(self._pre_ready)
        self.gl.sigDeviceRangeChanged.connect(self._geom_changed)      # 창 크기 변경 → 그 크기로 다시 준비

    # ---------------------------------------------------------------- 오디오 채움 미리 그리기
    def _geom_key(self):
        vb = self.p["audio"].getViewBox()
        g = env_geometry(vb, self.gl)
        D = g["dev"]
        r = g["rect"]
        return g, (r.width(), r.height(), D.m11(), D.m12(), D.m21(), D.m22(), D.dx(), D.dy(), g["dpr"])

    @staticmethod
    def full_ranges(d):
        a = d["audio"]
        m = float(max(np.max(np.abs(a["lo"])), np.max(np.abs(a["hi"])), 1e-6))
        return (0.0, max(float(d["t_end"]), 0.1)), (-m * 1.1, m * 1.3)

    def prepare(self, d):
        """패킷 분석 결과가 오면: 지금 창 크기의 전체 보기 오디오 채움을 작업 스레드에서 미리 그린다"""
        if d is None or d.get("audio") is None or len(d["audio"]["t"]) < 4:
            return
        shown = getattr(self, "_shown_once", False)
        # 한 번도 열지 않은 창은 배치가 정해지지 않았다 (숨긴 채 미리 배치하면 축 제목이 1화소 옮겨졌다: 아래 축 폭
        # 967.9999… vs 968.0) → 전체 보기 채움 그림은 첫 열기 이후부터. 채움 경로 · 개요 그림은 항상 미리
        g, key = self._geom_key() if shown else (None, None)
        ow, dpr = self.ov.width(), float(self.devicePixelRatioF())
        if d.get("_fill_path") is not None and (d.get("_pre") or (None,))[0] == key and \
                (d.get("_ov") or (None,))[0] == (ow, dpr):
            return
        xr, yr = self.full_ranges(d)
        a = {k: np.asarray(d["audio"][k], float) for k in ("t", "lo", "hi")}
        br = T.brush("rx_dim")

        def job():
            if d.get("_fill_path") is None:
                d["_fill_path"] = env_fill_path(a)
            d["_ov"] = ((ow, dpr), render_overview(d, ow, dpr))
            if g is not None:
                d["_pre"] = (key, render_env_full(a, g, br, xr, yr, d["_fill_path"]))
            if not getattr(self, "_closing", False):              # 17부: 창 닫는 중이면 신호 안 보냄 (지워진 창 RuntimeError)
                self._preReady.emit(d)
        if getattr(self, "_closing", False):
            return
        th = threading.Thread(target=job, daemon=True, name="dissect-prerender")
        d["_pre_thr"] = th
        self.__dict__.setdefault("_pre_threads", []).append(th)
        self._pre_threads = [t for t in self._pre_threads if t.is_alive() or t is th]
        th.start()

    def shutdown(self, timeout=2.0):
        """창 닫기 전 (17부): 새 미리 그리기 막고, 돌고 있는 미리 그리기 스레드가 끝나길 기다린 뒤 신호 연결을 끊는다"""
        self._closing = True
        for t in list(getattr(self, "_pre_threads", [])):
            t.join(timeout)
        try:
            self._preReady.disconnect(self._pre_ready)
        except (RuntimeError, TypeError):
            pass

    def _pre_ready(self, d):
        if d is self.d:
            self._set_overview(d)
            if self._fill is not None and d.get("_pre") is not None:
                self._fill.pre = d["_pre"][1]
                self._fill.update()

    def _set_overview(self, d):
        ow, dpr = self.ov.width(), float(self.devicePixelRatioF())
        ov = d.get("_ov")
        if ov is None or ov[0] != (ow, dpr):
            thr = d.get("_pre_thr")
            if thr is not None and thr.is_alive() and d.get("_ov") is None:
                return                                   # 작업 스레드가 곧 붙인다 (_pre_ready)
            d["_ov"] = ov = ((ow, dpr), render_overview(d, ow, dpr))   # 크기가 바뀐 경우: 그 크기로 다시 (수 ms)
        self.ov.set_image(ov[1], d["t_end"])

    def _geom_changed(self, *_):
        if self.d is not None:
            self.prepare(self.d)

    def _set_head(self, text):
        self.head.setText(trf(text))

    @staticmethod
    def _set_title(pl, text, extra="", color="text2"):
        """층 제목 (그래프 바깥 위쪽, 왼쪽 정렬). extra = 제목 줄 오른쪽 상태 (예: CRC 통과)"""
        text, extra = trf(text), trf(extra)
        html = '<span style="color:{}; font-size:{}pt">{}</span>'.format(T.c("text2"), T.PT_SMALL, text)
        if extra:
            html += '&nbsp;&nbsp;&nbsp;<span style="color:{}; font-size:{}pt; font-family:{}">{}</span>'.format(
                T.c(color), T.PT_SMALL, T.font_names()[1], extra)
        pl.setTitle(html, justify="left")

    # ---------------------------------------------------------------- 그리기
    def _add(self, key, item):
        self.p[key].addItem(item)
        self._items.append((key, item))
        return item

    def clear(self):
        for key, it in self._items + self._hl:
            self.p[key].removeItem(it)
        self._items, self._hl = [], []
        self._byte_labels, self._seg_text_labels, self._thin_items = [], [], []
        self._lazy = []
        self.tree.clear()
        self.ov.set_marks([])

    def show_packet(self, d, title="", wait_pre=True):
        """즉시 전부 그리기 (정상 경로)"""
        self.cancel_pending()
        for _ in self._build(d, title, wait_pre):
            pass

    # 미리 적용 (숨은 창): 같은 _build 를 이벤트 틱마다 약 15 ms 씩 나눠 실행
    SLICE_S = 0.015

    def pre_apply(self, d, title):
        self.cancel_pending()                        # 앞 패킷 작업이 남았으면 취소 → 새 패킷으로
        self._pend = {"d": d, "gen": self._build(d, title, False), "max": 0.0}
        self._pend_timer.start(0)

    def cancel_pending(self):
        self._pend = None
        self._pend_timer.stop()

    def finish_pending(self):
        """남은 작업을 즉시 끝낸다 (반쯤 만든 창을 보이지 않게)"""
        pd = self._pend
        if pd is None:
            return
        self._pend_timer.stop()
        for _ in pd["gen"]:
            pass
        self._pend = None

    def pending_d(self):
        return None if self._pend is None else self._pend["d"]

    def _pend_tick(self):
        import time as _t
        pd = self._pend
        if pd is None:
            return
        t0 = _t.perf_counter()
        while True:
            ts = _t.perf_counter()
            try:
                v = next(pd["gen"])
            except StopIteration:
                self._pend = None
                self._pend_timer.stop()
                return
            finally:
                pd["max"] = max(pd["max"], _t.perf_counter() - ts)
                self.pend_log = pd.setdefault("log", [])
                pd["log"].append(round((_t.perf_counter() - ts) * 1000, 1))
                self.pend_max = pd["max"]
            if v == "wait" or _t.perf_counter() - t0 >= self.SLICE_S:
                return                                # 다음 틱에 이어서

    def _build(self, d, title="", wait_pre=True):
        self.clear()
        yield
        self.d = d
        if d is None:
            self._set_head("해부 자료 없음" if title else
                              "패킷 선택 대기 (수신 메시지 표)")
            return
        s = d["summary"]
        stream = d.get("format") == "스트리밍"
        self._stream = stream
        tng5 = bool(d.get("tng5"))
        for key, t_, _ in LAYERS:
            self._set_title(self.p[key], (LAYERS_TNG5 if tng5 else LAYERS_STREAM).get(key, t_) if stream else t_)
        if tng5:
            cf = d.get("confirm") or {}
            st = "복조 성공" if s["ok"] else ("부분 복원 ({}/{} 구간)".format(s["segments_ok"], d["used"])
                                          if s["segments_ok"] else "복조 실패")
            how = {"mid": "M1 확인", "crc": "첫 CRC 구제"}.get(cf.get("by"), "파일 복조" if not cf else "미확정")
            self._set_head("{} · TNG5 {} · {} · 수신 확정: {}".format(title.replace(chr(10), " ↵ "),
                                                                        "v1 파일" if d.get("v1") else "v2", st, how))
        elif stream:
            used = d.get("used") or len(d["segments"])
            st = "복조 성공" if s["ok"] else ("부분 복원 ({}/{} 구간)".format(s["segments_ok"], used)
                                          if s["segments_ok"] else "복조 실패")
            self._set_head("{} · 스트리밍 형식 · {}".format(title, st))
        else:
            self._set_head("{} · 단일 블록 형식 · {}".format(title, "복조 성공" if s["ok"] else "복조 실패"))
        t0, t1 = d["t_data"]
        n, S, ntx, ni = d["n"], d["S"], len(d["llr_tx"]), d["n_info"]
        dur = t1 - t0
        # 1 오디오
        a = d["audio"]
        yield
        lo = self._add("audio", pg.PlotDataItem(a["t"], a["lo"], pen=T.pen("rx", 1)))
        yield
        hi = self._add("audio", pg.PlotDataItem(a["t"], a["hi"], pen=T.pen("rx", 1)))
        yield
        _EnvFill._given = d.get("_fill_path")             # 작업 스레드가 만들어 둔 같은 경로, 없으면 전체 보기 때 만든다
        _EnvFill._defer = _EnvFill._given is None
        try:
            fill = _EnvFill(lo, hi, np.asarray(a["t"], float), np.asarray(a["lo"], float), np.asarray(a["hi"], float),
                            self.p["audio"].getViewBox(), brush=T.brush("rx_dim"))
        finally:
            _EnvFill._given = None
            _EnvFill._defer = False
        if fill.path().isEmpty() and len(a["t"]):
            ta, la, ha = np.asarray(a["t"], float), np.asarray(a["lo"], float), np.asarray(a["hi"], float)
            y0, y1 = float(min(la.min(), ha.min())), float(max(la.max(), ha.max()))
            fill._br = QRectF(float(ta.min()), y0, float(ta.max() - ta.min()), y1 - y0)
            fill._deferred = True
        fill.setCacheMode(QGraphicsItem.DeviceCoordinateCache)     # 보기 범위가 그대로면 다시 채우지 않는다
        yield
        self._add("audio", fill)
        self._fill = fill
        # 전체 보기 그림이 아직이면 기다리지 않는다: 준비되면 _pre_ready 가 붙이고, 그 전에는 같은 채움을 직접 그린다
        fill.pre = (d.get("_pre") or (None, None))[1]
        m = float(max(np.max(np.abs(a["lo"])), np.max(np.abs(a["hi"])), 1e-6))
        self.p["audio"].setYRange(-m * 1.1, m * 1.3, padding=0)
        yield
        # 2 기저대역
        b = d["base"]
        self._add("base", pg.PlotDataItem(b["t"], b["q"], pen=T.pen("text3", 1)))
        self._add("base", pg.PlotDataItem(b["t"], b["i"], pen=T.pen("rx", 1)))
        self.p["base"].setYRange(-3.2, 4.0, padding=0)
        yield
        # 3 구간
        col = {"tone": "tx_dim", "costas": "rx_dim", "guard": "bg4", "missing": "err_dim"}
        ff = d["frame_fixed"]
        fi = 0
        bars, labs = {}, []                                      # 같은 색 막대는 한 항목으로 (그리는 방식 동일)
        for name, a0, a1, kind in d["segs"]:
            c = col.get(kind)
            if kind == "frame":
                c = "fec" if ff[fi] > 0 else ("bg4" if stream and d["frame_known"][fi] == 0 else "ok")
                fi += 1
            b = bars.setdefault((c, 200 if kind == "frame" else 255), ([], []))
            b[0].append(a0)
            b[1].append((a1 - a0) * 0.985)
            short = {"톤": "톤", "코스타스 A": "A", "가드": "G", "코스타스 B": "B", "B 없음": "B?",
                     "시작 동기 Welch12": "W12", "코스타스 A×4": "A×4"}.get(
                name, name.replace("프레임 ", "F"))
            labs.append(((a0 + a1) / 2, a1 - a0, [short], T.c("bg0") if kind == "frame" else T.c("text")))
        for (c, al), (x0, w) in bars.items():
            self._add("segs", pg.BarGraphItem(x0=x0, width=w, y0=0, height=0.8, brush=T.brush(c, al), pen=None))
        self._lazy_group("segs", 0.4, labs, T.ui_font(T.PT_SMALL))
        self.p["segs"].setYRange(-0.1, 1.5, padding=0)
        yield
        # 4 LLR (행 0 = 심볼의 첫 비트, 행 1 = 둘째 비트)
        img = np.clip(d["llr_tx"].reshape(n * S, 2).T, -15, 15)
        for f0_, f1_, ta_, tb_ in (d.get("llr_blocks") or [(0, n, t0, t1)]):     # TNG5 v2: 중간 동기 틈에서 끊음
            it = pg.ImageItem(img[:, f0_ * S:f1_ * S])
            it.setLookupTable(self._lut)
            it.setLevels((-15, 15))
            it.setRect(QRectF(ta_, 0, tb_ - ta_, 2))
            self._add("llr", it)
        e = np.nonzero(d["err_tx"])[0]
        if len(e):
            self._add("llr", pg.ScatterPlotItem(d["t_bit"][e], (e % 2) + 0.5, symbol="t1", size=7,
                                                pen=None, brush=T.brush("err")))
        self.p["llr"].setYRange(0, 2.8, padding=0)
        yield
        # 5 디인터리빙 후 부호 비트
        it = pg.ImageItem(np.clip(d["llr_coded"], -15, 15)[None, :])
        it.setLookupTable(self._lut)
        it.setLevels((-15, 15))
        it.setRect(QRectF(t0, 0, dur, 1))
        self._add("coded", it)
        nco = len(d["llr_coded"])
        xc = lambda i: t0 + (np.asarray(i) + 0.5) / nco * dur
        if len(d["fixed_coded"]):
            self._add("coded", pg.ScatterPlotItem(xc(d["fixed_coded"]), np.full(len(d["fixed_coded"]), 1.25),
                                                  symbol="t1", size=8, pen=None, brush=T.brush("err")))
        self.p["coded"].setYRange(0, 2.0, padding=0)
        yield
        # 6 정보 비트
        ib = d["info_bits"].astype(np.float32)
        it = pg.ImageItem(ib[None, :])
        if stream:                                              # -1 = 실패 구간 (값 모름)
            it.setLookupTable(np.array([pg.mkColor(T.c("err_dim")).getRgb()[:3], pg.mkColor(T.c("bg3")).getRgb()[:3],
                                        pg.mkColor(T.c("text2")).getRgb()[:3]], dtype=np.uint8))
            it.setLevels((-1, 1))
        else:
            it.setLookupTable(np.array([pg.mkColor(T.c("bg3")).getRgb()[:3], pg.mkColor(T.c("text2")).getRgb()[:3]],
                                       dtype=np.uint8))
            it.setLevels((0, 1))
        it.setRect(QRectF(t0, 0, dur, 1))
        self._add("info", it)
        xi = lambda j: t0 + (np.asarray(j) + 0.5) / ni * dur
        if len(d["fixed_info"]):
            self._add("info", pg.ScatterPlotItem(xi(d["fixed_info"]), np.full(len(d["fixed_info"]), 1.25),
                                                 symbol="t1", size=8, pen=None, brush=T.brush("fec")))
        self.p["info"].setYRange(0, 2.0, padding=0)
        yield
        # 7 CRC · 프레임별 오류 (스트리밍: 32바이트 구간별 CRC16)
        if stream:
            self._draw_seg_crc(d, t0, dur, ni)
            yield
            self._draw_bytes(d, t0, dur, stream=True)
            yield
            self.fit_all()
            yield
            self._thin_labels()
            yield
            self._fill_tree_stream(d, title)
            return
        fr_dur = dur / n
        crc = d["crc"]
        for f in range(n):
            c = "fec" if ff[f] > 0 else "ok"
            self._add("crc", pg.BarGraphItem(x0=[t0 + f * fr_dur], width=[fr_dur * 0.97], y0=0, height=0.8,
                                             brush=T.brush(c, 170), pen=None))
            ti = pg.TextItem(trf("오류 {}").format(int(ff[f])), color=T.c("bg0"), anchor=(0.5, 0.5))
            ti.setFont(T.ui_font(T.PT_SMALL))
            ti.setPos(t0 + (f + 0.5) * fr_dur, 0.4)
            self._add("crc", ti)
        crc_txt = ("CRC 통과 (수신 {:04X} = 계산 {:04X})".format(crc["rx"], crc["calc"]) if crc["ok"] else
                   "CRC 실패 ({})".format("수신 {:04X} ≠ 계산 {:04X}".format(crc["rx"], crc["calc"])
                                         if crc["rx"] is not None else crc["reason"]))
        self._set_title(self.p["crc"], dict((k_, t_) for k_, t_, _ in LAYERS)["crc"], crc_txt,
                        "ok" if crc["ok"] else "err")          # 프레임 칸을 가리지 않게 제목 줄에
        self.p["crc"].setYRange(-0.1, 1.5, padding=0)
        # 8 바이트
        self._draw_bytes(d, t0, dur, stream=False)
        self.fit_all()
        self._thin_labels()
        self._fill_tree(d, title)

    def _draw_bytes(self, d, t0, dur, stream):
        nb = len(d["bytes"])
        bcol = {"len": "cat3", "body": "rx_dim", "crc": "fec", "pad": "bg3", "unk": "err_dim"}
        bw = dur / max(nb, 1)
        for kind in bcol:
            idx = [i for i, k in enumerate(d["byte_kinds"]) if k == kind]
            if idx:
                self._add("bytes", pg.BarGraphItem(x0=[t0 + i * bw for i in idx], width=[bw * 0.92] * len(idx),
                                                   y0=0, height=0.8, brush=T.brush(bcol[kind], 200), pen=None))
        kinds = d["byte_kinds"]
        self._lazy_group("bytes", 0.4, [(t0 + (i + 0.5) * bw, bw, ["··" if kinds[i] == "unk" else "{:02X}".format(int(val))],
                                         T.c("bg0") if kinds[i] == "crc" else T.c("text"))
                                        for i, val in enumerate(d["bytes"])], T.mono_font(T.PT_SMALL), min_px=20)
        if d.get("groups5"):                         # TNG5: 2바이트 묶음 → 3글자 (base-40), 실패 구간은 구간 표시
            labs = [(t0 + (b0 + 1) * bw, 2 * bw, ["↵" if ch == "\n" else ('"{}"'.format(ch) if not ch.startswith("(") else ch)],
                     T.c("text")) for b0, ch, _ in d["groups5"]]
            for r in d["segments"]:
                if not r["ok"]:
                    a0 = r["info"][0] // 8
                    labs.append((t0 + (a0 + 10) * bw, 20 * bw, ["CRC 실패"], T.c("err")))
            self._lazy_group("bytes", 1.15, labs, T.ui_font(T.PT_SMALL))
        elif stream:                                   # 구간 텍스트 (바이트 줄 위)
            labs = []
            for r in d["segments"]:
                a0 = r["info"][0] // 8
                x = t0 + (a0 + (r["bytes"][1] + 2) / 2) * bw
                txt = r["text"] if r["ok"] else ("CRC 실패" if r["decided"] else "미확정")
                labs.append((x, (r["bytes"][1] + 2) * bw, [txt],
                             T.c("text") if r["ok"] else T.c("err")))
            self._lazy_group("bytes", 1.15, labs, T.ui_font(T.PT_SMALL))
        self.p["bytes"].setYRange(-0.1, 1.5, padding=0)
        if not getattr(self, "_thin_connected", False):          # 한 번만 연결
            self.p["bytes"].getViewBox().sigXRangeChanged.connect(self._thin_labels)
            self._thin_connected = True

    def _draw_seg_crc(self, d, t0, dur, ni):
        """7층 (스트리밍): 구간마다 CRC16 통과 / 실패 / 미확정 막대, 라벨 = 구간 번호 · 정렬, ▼ = 재탐색"""
        bars, labs, rl = {}, [], []
        for r in d["segments"]:
            a, e = r["info"]
            x0, w = t0 + a / ni * dur, (e - a) / ni * dur
            c = "ok" if r["ok"] else ("err" if r["decided"] else "bg3")
            b = bars.setdefault(c, ([], []))
            b[0].append(x0)
            b[1].append(w * 0.97)
            lab = "S{}".format(r["k"] + 1) + ("" if r["offset"] is None else " {:+d}".format(int(r["offset"])))
            labs.append((x0 + w / 2, w, [lab, "S{}".format(r["k"] + 1), str(r["k"] + 1)],
                         T.c("bg0") if r["ok"] else T.c("text")))
            if r["relock"]:
                rl.append(x0 + w / 2)
        for c, (x0, w) in bars.items():
            self._add("crc", pg.BarGraphItem(x0=x0, width=w, y0=0, height=0.8, brush=T.brush(c, 170), pen=None))
        self._lazy_group("crc", 0.4, labs, T.mono_font(T.PT_SMALL))
        if rl:
            self._add("crc", pg.ScatterPlotItem(rl, [1.05] * len(rl), symbol="t", size=9, pen=None,
                                                brush=T.brush("fec")))
        n_ok = sum(r["ok"] for r in d["segments"])
        dr = d.get("drift_ppm")
        extra = "통과 {}/{} · 드리프트 {} · 재탐색 {}회 (▼)".format(
            n_ok, d.get("used") or len(d["segments"]), "-" if dr is None else "{:+.0f} ppm".format(dr),
            d.get("relocks", 0))
        if d.get("tng5"):
            lf = next((r["len_field"] for r in d["segments"] if r.get("len_field") is not None), None)
            extra = "통과 {}/{} · 길이 필드 {}".format(n_ok, d.get("used") or len(d["segments"]),
                                                  "-" if lf is None else "{}{}".format(lf, "구간" if d.get("charset") == "charset1" else "자"))
        self._set_title(self.p["crc"], (LAYERS_TNG5 if d.get("tng5") else LAYERS_STREAM)["crc"], extra,
                        "ok" if d["crc"]["ok"] else "err")
        self.p["crc"].setYRange(-0.1, 1.5, padding=0)

    def fit_all(self):
        """
        가로축 = 패킷 시작(0, 톤 시작) ~ 패킷 끝. 확대 · 이동은 이 범위 안에서만 (8층 가로축 연동).
        제한이 없던 때는 마우스 휠 축소만으로 축이 수십만 초까지 벌어졌다 (12회 축소 → ±73,600 s 재현).
        """
        if self.d is None:
            return
        t1 = max(float(self.d["t_end"]), 0.1)
        for pl in self.p.values():
            vb = pl.getViewBox()
            vb.setLimits(xMin=0.0, xMax=t1, minXRange=min(0.05, t1), maxXRange=t1)
            vb.enableAutoRange(x=False)
        self._pages = self._make_pages(self.d, self.sp_n.value())
        self._set_overview(self.d)
        self.go_page(0)

    @staticmethod
    def _make_pages(d, n):
        """상세 범위 목록: 32바이트 구간 n 개씩 (첫 범위는 톤 시작부터, 끝 범위는 패킷 끝까지). 구간 n 개 이하면 전체 하나"""
        t_end = max(float(d["t_end"]), 0.1)
        segs = d.get("segments") if d.get("format") == "스트리밍" else None
        if not segs or len(segs) <= n:
            return [(0.0, t_end)]
        t0, t1 = d["t_data"]
        ni, dur = d["n_info"], t1 - t0
        xs = [(t0 + r["info"][0] / ni * dur, t0 + r["info"][1] / ni * dur) for r in segs]
        pages = []
        for i in range(0, len(xs), n):
            pages.append([xs[i][0], xs[min(i + n, len(xs)) - 1][1]])
        pages[0][0] = 0.0
        pages[-1][1] = t_end
        return [tuple(p_) for p_ in pages]

    def n_pages(self):
        return len(self._pages)

    def go_page(self, k):
        if self.d is None:
            return
        k = int(min(max(k, 0), len(self._pages) - 1))
        self.page = k
        a, b = self._pages[k]
        self.p["bytes"].setXRange(a, b, padding=0)            # 8층 가로축 연동
        n = len(self._pages)
        self.lb_page.setText("{} / {}".format(k + 1, n))
        self.btn_prev.setEnabled(k > 0)
        self.btn_next.setEnabled(k < n - 1)
        self._thin_labels()

    def seek_time(self, t):
        """개요 클릭 · 끌기: 그 시각이 들어 있는 상세 범위로"""
        k = next((i for i, (a, b) in enumerate(self._pages) if t < b), len(self._pages) - 1)
        if k != self.page:
            self.go_page(k)

    def _set_n(self, n):
        if self.d is not None:
            self._pages = self._make_pages(self.d, n)
            self.go_page(0)

    def _view_changed(self, *_):
        a, b = self.p["bytes"].getViewBox().viewRange()[0]
        self.ov.set_view(a, b)

    def showEvent(self, ev):
        # 보이기 전에 그리면 폭 0 인 상태에서 범위가 잡혀 처음 범위가 어긋났다 (1.5 ~ 35.9 s 재현) → 보일 때 다시 맞춘다
        super().showEvent(ev)
        self._shown_once = True
        QTimer.singleShot(0, self._shown)

    def _shown(self):
        self.fit_all()
        self.prepare(self.d)                          # 보인 뒤 확정된 크기로 전체 보기 채움 준비

    def _thin_labels(self, *_):
        """바이트 칸이 좁으면 16진수 글자를 숨긴다 (겹치지 않게)"""
        if self.d is None:
            return
        vb = self.p["bytes"].getViewBox()
        x0, x1 = vb.viewRange()[0]
        pxu = vb.width() / max(x1 - x0, 1e-9)                         # 데이터 1단위(초)당 화소
        for g in getattr(self, "_lazy", []):
            # 칸 폭에 맞는 가장 긴 라벨 (바이트: 칸 20 화소 이상), 없으면 숨김.
            # 보이는 가로 범위 밖 라벨은 어차피 잘려 안 보이므로 만들지 않는다 (범위에 들어오면 그때 만든다)
            wpx = g["w"] * pxu
            if g["min_px"]:
                vi = np.where(wpx >= g["min_px"], 0, -1)
            else:
                ok = g["tw"] + 6 <= wpx[:, None]
                vi = np.where(ok.any(1), ok.argmax(1), -1)
            show = (g["x"] + g["w"] >= x0) & (g["x"] - g["w"] <= x1) & (vi >= 0)
            made = g["items"]
            for i, ti in made.items():
                if not show[i]:
                    ti.setVisible(False)
            for i in np.nonzero(show)[0]:
                txt = g["var"][i][vi[i]]
                ti = made.get(i)
                if ti is None:
                    ti = pg.TextItem(trf(txt), color=g["col"][i], anchor=(0.5, 0.5))
                    ti.setFont(g["font"])
                    ti.setPos(g["x"][i], g["y"])
                    self._add(g["key"], ti)
                    made[i] = ti
                else:
                    if ti.textItem.toPlainText() != trf(txt):
                        ti.setText(trf(txt))
                    ti.setVisible(True)

    def _lazy_group(self, key, y, labs, font, min_px=0):
        """라벨 묶음 [(x 중심, 칸 폭, 후보 문구들, 색)] — TextItem 은 _thin_labels 가 보이는 것만 만든다"""
        if not labs:
            return
        var = [v for _, _, v, _ in labs]
        tw = np.full((len(var), max(len(v) for v in var)), np.inf)
        for i, v in enumerate(var):
            tw[i, :len(v)] = [sum(12 if ord(c) > 0x2000 else 7 for c in s_) for s_ in v]
        self._lazy.append({"key": key, "y": y, "font": font, "min_px": min_px, "var": var, "tw": tw,
                           "x": np.array([l[0] for l in labs], float), "w": np.array([l[1] for l in labs], float),
                           "col": [l[3] for l in labs], "items": {}})

    def _fill_tree(self, d, title):
        s, crc = d["summary"], d["crc"]
        f2 = lambda v, u="": "-" if v is None or not np.isfinite(v) else "{:+.1f}{}".format(v, u)

        def node(name, kv):
            it = QTreeWidgetItem([name, ""])
            for k, v_ in kv:
                QTreeWidgetItem(it, [k, str(v_)])
            self.tree.addTopLevelItem(it)
            it.setExpanded(True)
            return it
        a = d["audio"]
        node("패킷", [("형식", "단일 블록 (4~6단계)"), ("상태", "복조 성공" if s["ok"] else "복조 실패"),
                     ("경로", s["mode"]), ("수신", title)])
        node("1 오디오", [("길이", "{:.2f} s".format(d["t_end"])),
                         ("최대 레벨", "{:.1f} dBFS".format(20 * np.log10(max(np.max(np.abs(a["hi"])), 1e-9))))])
        node("2 기저대역", [("주파수 오차", f2(s["df"], " Hz")), ("추정 SNR", f2(s["snr"], " dB"))])
        node("3 구간", [("A ρ²", "{:.3f}".format(s["rho_a"])),
                       ("B ρ²", "-" if s["rho_b"] is None else "{:.3f}".format(s["rho_b"])),
                       ("데이터 프레임", s["frames"])])
        node("4 LLR", [("비트 수", s["coded_bits"]), ("평균 |LLR|", "{:.1f}".format(s["mean_abs_llr"])),
                      ("hard 판정 오류", s["fixed"])])
        node("5 디인터리빙", [("부호 비트 수", s["coded_bits"])])
        node("6 비터비", [("정보 비트 수", s["info_bits"]), ("정정 부호 비트", s["fixed"])])
        node("7 CRC", [("결과", "CRC 통과" if crc["ok"] else "CRC 실패 ({})".format(crc["reason"])),
                      ("수신 / 계산", "-" if crc["rx"] is None else "{:04X} / {:04X}".format(crc["rx"], crc["calc"])),
                      ("프레임별 오류", " ".join(str(int(x)) for x in d["frame_fixed"]))])
        node("8 바이트", [("길이 필드", s["plen"]), ("전체 바이트", s["nbytes"]), ("텍스트", d["text"] or "-")])
        self.sel_item = node("선택", [("", "층 클릭 대기")])

    def _fill_tree_stream(self, d, title):
        if d.get("tng5"):
            return self._fill_tree_tng5(d, title)
        s = d["summary"]
        f2 = lambda v, u="": "-" if v is None or not np.isfinite(v) else "{:+.1f}{}".format(v, u)

        def node(name, kv, parent=None):
            it = QTreeWidgetItem([name, ""])
            for k, v_ in kv:
                QTreeWidgetItem(it, [k, str(v_)])
            (parent.addChild(it) if parent is not None else self.tree.addTopLevelItem(it))
            it.setExpanded(True)
            return it
        used = d.get("used") or len(d["segments"])
        st = "복조 성공" if s["ok"] else ("부분 복원 ({}/{} 구간)".format(s["segments_ok"], used)
                                      if s["segments_ok"] else "복조 실패")
        node("패킷", [("형식", "스트리밍 (7단계)"), ("상태", st), ("경로", s["mode"]), ("수신", title)])
        a = d["audio"]
        node("1 오디오", [("길이", "{:.2f} s".format(d["t_end"])),
                         ("최대 레벨", "{:.1f} dBFS".format(20 * np.log10(max(np.max(np.abs(a["hi"])), 1e-9))))])
        node("2 기저대역", [("주파수 오차", f2(s["df"], " Hz")), ("추정 SNR", f2(s["snr"], " dB"))])
        dr = d.get("drift_ppm")
        node("3 구간", [("A ρ²", "{:.3f}".format(s["rho_a"])),
                       ("B ρ²", "-" if s["rho_b"] is None else "{:.3f}".format(s["rho_b"])),
                       ("데이터 프레임", s["frames"]),
                       ("클럭 드리프트 추정", "-" if dr is None else "{:+.0f} ppm (A~B 간격)".format(dr))])
        node("4 LLR", [("비트 수", s["coded_bits"]), ("평균 |LLR|", "{:.1f}".format(s["mean_abs_llr"])),
                      ("정정 비트 / 확인 가능", "{} / {}".format(s["fixed"], s["known_coded"]))])
        node("5 디인터리빙", [("부호 비트 수", len(d["llr_coded"])),
                           ("인터리버", "지연순 정렬 (가지 12 · 지연 단위 24 · 최대 이동 3.2 s)")])
        node("6 비터비", [("정보 비트 수", s["info_bits"]), ("결정 깊이", "구간 끝 + 48 정보 비트 (B 뒤 최종 복조는 끝까지)"),
                        ("정렬 가설", "19개 (±18 샘플) · 추적 ±4 · 연속 2 실패 시 전 범위")])
        segn = node("7 구간 CRC16", [("통과", "{} / {}".format(s["segments_ok"], used)),
                                    ("재탐색 (잠금 복구)", "{}회".format(d.get("relocks", 0)))])
        for r in d["segments"]:
            fr = r["frames"]
            st_ = "통과" if r["ok"] else ("실패" if r["decided"] else "미확정")
            QTreeWidgetItem(segn, ["S{}".format(r["k"] + 1), "{} · 정렬 {} · 프레임 {}–{}{} · {}".format(
                st_, "-" if r["offset"] is None else "{:+d}".format(int(r["offset"])),
                int(fr.min()) + 1, int(fr.max()) + 1, " · 재탐색" if r["relock"] else "", r["text"][:24])])
        node("8 바이트", [("전체 바이트", s["nbytes"]), ("텍스트", d["text"] or "-")])
        self.sel_item = node("선택", [("", "층 클릭 대기")])

    def _fill_tree_tng5(self, d, title):
        s = d["summary"]
        f2 = lambda v, u="": "-" if v is None or not np.isfinite(v) else "{:+.1f}{}".format(v, u)

        def node(name, kv, parent=None):
            it = QTreeWidgetItem([name, ""])
            for k, v_ in kv:
                QTreeWidgetItem(it, [k, str(v_)])
            (parent.addChild(it) if parent is not None else self.tree.addTopLevelItem(it))
            it.setExpanded(True)
            return it
        used = d["used"]
        st = "복조 성공" if s["ok"] else ("부분 복원 ({}/{} 구간)".format(s["segments_ok"], used)
                                      if s["segments_ok"] else "복조 실패")
        cf = d.get("confirm") or {}
        how = {"mid": "중간 동기 M1 확인", "crc": "첫 CRC 구제 (M1 확인 실패)"}.get(
            cf.get("by"), "파일 복조 (실시간 확정 없음)" if not cf else "미확정")
        node("패킷", [("형식", "TNG5 v1 파일 (A×4 · B×4)" if d.get("v1") else "TNG5 v2 (Welch12 · 중간 동기 · B×1)"),
                     ("상태", st), ("수신 확정", how), ("경로", s["mode"]), ("수신", title)])
        a = d["audio"]
        node("1 오디오", [("길이", "{:.2f} s".format(d["t_end"])),
                         ("최대 레벨", "{:.1f} dBFS".format(20 * np.log10(max(np.max(np.abs(a["hi"])), 1e-9))))])
        node("2 기저대역", [("주파수 오차", f2(s["df"], " Hz")), ("추정 SNR", f2(s["snr"], " dB"))])
        kv = [("시작 동기 점수", "{:.1f}".format(s["rho_a"]) if np.isfinite(s["rho_a"]) else "-"),
              ("데이터 프레임", s["frames"]), ("중간 동기", "{}개 (20프레임 = 3.84 s 마다)".format(len(d.get("mids") or [])))]
        for mc in d.get("mid_checks") or []:
            kv.append(("M{}".format(mc["k"]), "ρ² {:.3f} (기준 {:.3f}) · {} · 위치 {:+.1f} ms".format(
                mc["rho"], mc["need"], "통과" if mc["ok"] else "실패", mc["t_s"] * 1e3)))
        kv.append(("B", "검출 (길이 필드 결과 뒤 인정)" if d.get("b_state") == "ok" else
                   "검출" if s["rho_b"] is not None or any(x[0] == "코스타스 B" for x in d["segs"]) else
                   "미검출" if any(x[0] == "B 없음" for x in d["segs"]) else "-"))
        node("3 구간", kv)
        node("4 LLR", [("비트 수", s["coded_bits"]), ("평균 |LLR|", "{:.2f}".format(s["mean_abs_llr"])),
                      ("정정 비트 (사본 기준) / 확인 가능", "{} / {}".format(int(d["err_tx"].sum()), s["known_coded"] * 4))])
        rn = node("5 반복 4회 결합", [("결합 부호 비트", len(d["llr_coded"])), ("부호", "K=7 R1/4 (117 127 155 171) × 반복 4 = R1/16"),
                                  ("인터리버", "구간 안 무작위 + 지연 (최대 약 9.6 s)"),
                                  ("정정 비트 (결합 후)", s["fixed"])])
        for r in d["segments"]:
            QTreeWidgetItem(rn, ["S{} 사본 |LLR|".format(r["k"] + 1), " · ".join("{:.2f}".format(v) for v in r["rep_llr"])])
        node("6 비터비", [("정보 칸", "{} (구간마다 160 + 꼬리 6 + 빈 2)".format(s["info_bits"]))])
        segn = node("7 구간 CRC16", [("통과", "{} / {}".format(s["segments_ok"], used))])
        for r in d["segments"]:
            fr = r["frames"]
            st_ = "통과" if r["ok"] else "실패"
            QTreeWidgetItem(segn, ["S{}".format(r["k"] + 1), "{} · 프레임 {}–{}{} · {}".format(
                st_, int(fr.min()) + 1, int(fr.max()) + 1,
                "" if r.get("len_field") is None else " · 길이 필드 {}{}".format(
                    r["len_field"], "구간" if d.get("charset") == "charset1" else "자"),
                "" if r["crc_calc"] is None else "CRC {:04X}".format(r["crc_calc"]))])
        if d.get("charset") == "charset1":
            bn = node("8 글자", [("부호", "charset1 57자 가변 길이 (구간마다 글자 단위 · 마지막 구간 EOT · 바이트 흰색화)"),
                               ("텍스트", (d["text"] or "-").replace("\n", " ↵ "))])
        else:
            bn = node("8 base-40", [("묶음", "2바이트 값 v (0–63999) → 3글자 (v = c1·1600 + c2·40 + c3)"),
                                   ("텍스트", d["text"] or "-")])
        for r in d["segments"]:
            if r["ok"]:
                QTreeWidgetItem(bn, ["S{}".format(r["k"] + 1), r["text"].replace("\n", " ↵ ")])
        self.sel_item = node("선택", [("", "층 클릭 대기")])

    # ---------------------------------------------------------------- 클릭 → 층 사이 대응
    def _click(self, ev):
        d = self.d
        if d is None:
            return
        pos = ev.scenePos()
        key = next((k for k, pl in self.p.items() if pl.getViewBox().sceneBoundingRect().contains(pos)), None)
        if key is None:
            return
        pt = self.p[key].getViewBox().mapSceneToView(pos)
        x, y = pt.x(), pt.y()
        t0, t1 = d["t_data"]
        dur = t1 - t0
        ntx, ni, nb = len(d["llr_tx"]), d["n_info"], len(d["bytes"])
        u = (x - t0) / dur
        if not (0 <= u < 1):
            self._highlight(None, x, key)
            return
        if key == "coded":
            L = links_from_coded(d, [int(u * len(d["llr_coded"]))])
        elif key == "info":
            L = links_from_info(d, [int(u * ni)])
        elif key == "bytes":
            L = links_from_byte(d, int(u * nb))
        else:                                                  # 시간 층: 그 심볼의 두 비트 (LLR 층은 행으로 한 비트)
            if d.get("frame_t") is not None:                    # TNG5: 중간 동기 틈 반영
                ft = d["frame_t"]
                f_ = int(np.clip(np.searchsorted(ft, x, side="right") - 1, 0, d["n"] - 1))
                w_ = (x - ft[f_]) / d["frame_s"]
                if not (0 <= w_ < 1):
                    self._highlight(None, x, key)
                    return
                sym = f_ * d["S"] + int(w_ * d["S"])
            else:
                sym = int(u * d["n"] * d["S"])
            p0 = 2 * sym
            tx = [p0 + (1 if y >= 1 else 0)] if key == "llr" and 0 <= y < 2 else [p0, p0 + 1]
            L = links_from_tx(d, tx)
        self._highlight(L, x, key)

    def _highlight(self, L, x, key):
        d = self.d
        for k, it in self._hl:
            self.p[k].removeItem(it)
        self._hl = []

        def add(k, it):
            self.p[k].addItem(it)
            self._hl.append((k, it))
        pen = T.pen("text", 1.2)
        if L is None:
            for k in ("audio", "base", "segs"):
                add(k, pg.InfiniteLine(x, angle=90, pen=pen))
            self.ov.set_marks([x])
            self.sel_item.takeChildren()
            QTreeWidgetItem(self.sel_item, ["시각", "{:.3f} s (데이터 구간 외)".format(x)])
            return
        t0, t1 = d["t_data"]
        dur = t1 - t0
        ntx, ni, nb = len(d["llr_tx"]), d["n_info"], len(d["bytes"])
        tt = d["t_bit"][L["tx"]]
        self.ov.set_marks(tt)                               # 다른 범위에 있는 대응 위치도 개요에 (송신 위치 시각)
        for k in ("audio", "base", "segs", "crc"):
            xs = np.repeat(tt, 3)
            ys = np.tile([-1e3, 1e3, np.nan], len(tt))
            add(k, pg.PlotDataItem(xs, ys, pen=pen, connect="finite"))
        add("llr", pg.ScatterPlotItem(tt, (L["tx"] % 2) + 0.5, symbol="s", size=9, pen=pen, brush=None))
        add("coded", pg.ScatterPlotItem(t0 + (L["coded"] + 0.5) / len(d["llr_coded"]) * dur,
                                        np.full(len(L["coded"]), 0.5),
                                        symbol="s", size=9, pen=pen, brush=None))
        add("info", pg.ScatterPlotItem(t0 + (L["info"] + 0.5) / ni * dur, np.full(len(L["info"]), 0.5),
                                       symbol="s", size=9, pen=pen, brush=None))
        bw = dur / max(nb, 1)
        add("bytes", pg.BarGraphItem(x0=[t0 + b * bw for b in L["byte"]], width=[bw * 0.92] * len(L["byte"]),
                                     y0=0, height=0.8, brush=None, pen=T.pen("text", 1.6)))
        # 트리: 따라간 길
        self.sel_item.takeChildren()
        fr = sorted(set(int(p) // 96 + 1 for p in L["tx"]))
        kinds = [d["byte_kinds"][b] for b in L["byte"]]
        stream = d.get("format") == "스트리밍"
        byte_s = ", ".join("{}{} ({:02X}, {})".format(
            int(b), " · S{}".format(int(b) // d.get("seg_nbytes", 34) + 1) if stream else "", int(d["bytes"][b]),
            {"len": "길이", "body": "본문", "crc": "CRC", "pad": "패딩", "unk": "모름"}[k])
            for b, k in zip(L["byte"], kinds))
        lay_t = dict((k_, t_) for k_, t_, _ in LAYERS)
        if stream:
            lay_t.update(LAYERS_TNG5 if d.get("tng5") else LAYERS_STREAM)
        rows = [("선택 층", lay_t[key].split("  ")[1]),
                ("송신 위치", "{}개 (프레임 {})".format(len(L["tx"]), ", ".join(map(str, fr)))),
                ("시각", ", ".join("{:.3f}".format(v) for v in sorted(tt)[:4]) + (" …" if len(tt) > 4 else "") + " s"),
                ("LLR", ", ".join("{:+.1f}".format(v) for v in d["llr_tx"][L["tx"]][:4]) + (" …" if len(L["tx"]) > 4 else "")),
                ("부호 비트", ", ".join(map(str, L["coded"][:6])) + (" …" if len(L["coded"]) > 6 else "")),
                ("정보 비트", ", ".join(map(str, L["info"][:6])) + (" …" if len(L["info"]) > 6 else "")),
                ("정정 비트", "{}개".format(int(np.isin(L["coded"], d["fixed_coded"]).sum()))),
                ("바이트", byte_s)]
        for k_, v_ in rows:
            QTreeWidgetItem(self.sel_item, [k_, v_])
        self.selected.emit(L)
