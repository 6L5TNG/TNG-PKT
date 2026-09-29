"""
TNG PKT — 신경망 변조 디지털 모드 교신 프로그램 (모드 TNG44)

설정 · 레이아웃 · 로그: %APPDATA%\TNG PKT (app_paths, app_settings). 버전 · 프로토콜 목록: registry.

    python neuromod_app.py                 # 실행
    python neuromod_app.py --rxlog rx.log  # 실시간 수신 사건을 파일에도 기록
    python selftest.py                     # 자가검사 (빠른 모드)

구성
  · 모든 화면 요소는 도킹 패널. 끌어서 옮기기, 떼어 별도 창, 다른 모니터로 옮기기 가능
  · 레이아웃 프리셋 3개 (운용 / 분석 / 전체), 분석 창 분리, 종료할 때 배치 저장 → 다음 실행 때 복원
  · 연속 실시간(LIVE) 패널과 패킷 단위 패널을 제목줄에서 구분
  · 운용: 워터폴 + 수신(구간 확정되는 대로 글자 표시) + 송신(구간 단위 진행 색)
  · 7단계 스트리밍 형식(32바이트 구간 + CRC16)과 옛 단일 블록 형식을 결과의 'format' 으로 나눠 표시
엔진(변조/복조/FEC/코스타스)과 모델은 건드리지 않는다. 오류는 상태 막대에 표시한다.
무전기 CAT · PTT (18부, rig.py): Hamlib rigctld · 시리얼 DTR/RTS · VOX. VFO 패널 · 설정 '무전기'. 송신 끝 · 오류 · 종료 때 PTT 해제.
"""

import os
os.environ.setdefault("PYQTGRAPH_QT_LIB", "PySide6")

import _io_utf8  # noqa: F401
import sys
if __name__ == "__main__" and "--offscreen" not in sys.argv[1:] and "--no-setup" not in sys.argv[1:]:
    import first_run                                  # 24부: 설정 파일 없으면 초기 설정 (창 모듈을 읽기 전: 고른 언어가 문구에 바로)
    first_run.maybe_run()
import threading
import time
from collections import deque
from datetime import datetime, timezone

import numpy as np
import sounddevice as sd
from PySide6.QtCore import Qt, QObject, QTimer, Signal, QSettings, QSize, QEvent
from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
                               QLabel, QComboBox, QPushButton, QLineEdit, QSlider, QFileDialog,
                               QCheckBox, QDockWidget, QToolBar, QToolButton, QButtonGroup, QMenu,
                               QSizePolicy, QTableWidgetItem, QScrollArea, QFrame, QSpinBox)

import theme as T
from widgets import (DockTitle, LEDMeter, CenterMeter, MarginGauge, TXIndicator, StatGrid, SpectrumPanel,
                     WaterfallWidget, TimelinePanel, TrendPanel, MessageTable, MacroBar, ViewToggle,
                     ConstellationPanel, EyePanel, SyncMapPanel, LLRPanel, FeaturePanel,
                     segment, hline, mono_when_filled, short_device, ElideLabel, fmt_clock)
from session import SessionLog, expand_macro
from app_settings import Settings, migrate_old_locations
import app_paths
import registry
from strings_ko import tr, trf
from widgets_zoom import ZoomPanel
from widgets_dissect import DissectPanel
from widgets_qso import QsoRxPanel, QsoTxPanel, TxAnalysisPanel, RxLog

T.setup_pyqtgraph()
APP_NAME = registry.APP_NAME
LAYOUT_VER = 11                   # 도킹 배치 형식 번호. 기본 배치가 바뀌면 올린다 (옛 저장본은 무시)
UI_FPS = 20                        # 화면 갱신 상한
FAIL_SNR_DB = -6.5
CENTER = 1500.0
ANALYSIS_KEYS = ("const", "eye", "feat", "llr", "sync", "trend")
SEP = 4                            # 패널 사이 틈 (style.qss 의 QMainWindow::separator 와 같게)
# 패널 내용 최소 크기 (px). 그래프 · 축 제목 · 글자가 제대로 그려지는 최소치.
# 창이 이보다 작아지면 패널이 눌리지 않고 그 패널 안에 스크롤 막대가 생긴다.
PANEL_MIN = {"spectrum": (360, 140), "waterfall": (360, 220), "meters": (280, 226), "timeline": (360, 112),
             "rx": (548, 150), "tx": (300, 190), "quality": (360, 150), "trend": (280, 290),
             "const": (260, 210), "eye": (260, 210), "feat": (260, 210), "llr": (260, 200),
             "sync": (270, 214), "dissect": (760, 560), "zoom": (440, 380),
             "qrx": (420, 200), "vfo": (300, 92), "qtx": (420, 180), "txmon": (560, 470), "txana": (520, 400), "tones": (560, 320)}
FLOAT_KEYS = ("dissect", "zoom", "tones")
LAYOUT_KEY = {"운용": "layout.op", "분석": "layout.ana", "전체": "layout.all"}      # 프리셋 이름 → 화면 글자 키
# 패널 제목줄 '?' 도움말 (개조식, 한 줄 한 항목, 명사형)
HELP = {
    "vfo": tr("vfo.help"),
    "tones": tr("main.vertical_16_tones_horizontal"),
    "spectrum": tr("main.vertical_dbfs_solid_offset"),
    "waterfall": tr("main.horizontal_frequency_vertical_time"),
    "meters": tr("main.input_level_peak_hold"),
    "timeline": tr("main.top_frames_bottom_segment"),
    "rx": tr("main.row_select_analysis_panels"),
    "tx": tr("main.right_click_edit_delete"),
    "qrx": tr("main.crc_fail_segment_waiting"),
    "qtx": tr("main.stop_send_the_current"),
    "txmon": tr("main.throughput_sent_chars_elapsed"),
    "txana": tr("main.bytes_info_bits_coded"),
    "quality": tr("main.ber_hard_decision_fec"),
    "trend": tr("main.last_60_packets_red"),
    "const": tr("main.eq_channel_estimate_from"),
    "eye": tr("main.two_symbol_trace_green"),
    "feat": tr("main.nn_features_2d_pca"),
    "llr": tr("main.llr_bit_decision_confidence"),
    "sync": tr("main.normalized_correlation_same_as"),
    "dissect": tr("main.click_matching_position_in"),
    "zoom": tr("main.dotted_sync_tones_dashed"),
}   # 프리셋에 넣지 않는 패널: 처음 열 때 떠 있는 창으로 (배치를 누르지 않게)


class Bridge(QObject):
    """다른 스레드 → UI 스레드"""
    engine_ready = Signal(object, str)
    decoded = Signal(object, str)
    status = Signal(str, bool)
    rx_status = Signal(str, bool)          # 수신기 상태 문구: CRC 통과 전이면 로그만 (_on_rx_status)
    saved = Signal(str, float)
    rt_marker = Signal(object)
    live = Signal(object)
    rig_state = Signal(object)             # 18부: 무전기 CAT · PTT 상태 (rig 작업 스레드 → UI)


class DockScroll(QScrollArea):
    """
    패널 내용 스크롤 영역. 내용은 보이는 영역을 꽉 채우되 최소 크기(mn) 아래로는 줄이지 않는다.
    (QScrollArea 기본 방식은 줄바꿈 라벨의 heightForWidth 때문에 여유가 있어도 스크롤이 생긴다)
    """

    def __init__(self, w, mn):
        super().__init__()
        self.setObjectName("dockScroll")
        self.setFrameShape(QFrame.NoFrame)
        self.setWidgetResizable(False)
        self.mn = mn
        self._min = None
        self.setWidget(w)
    def minimumSizeHint(self):
        # 최소 크기 = 패널 내용 최소치. 창을 줄이면 Qt 가 여유 있는 패널부터 줄이고 이 아래로는 누르지 않는다.
        # 레이아웃이 1초에도 여러 번 묻기 때문에 값을 기억해 두고, 보일 때 · 스타일/글꼴이 바뀔 때만 다시 잰다
        # (매번 재면 UI 스레드가 GIL 을 오래 잡아 복조 스레드가 늦어진다)
        if self._min is None:
            m = self.widget().minimumSizeHint()
            self._min = QSize(max(self.mn[0], m.width()), max(self.mn[1], m.height()))
        return self._min

    def _remeasure(self):
        self._min = None
        self.updateGeometry()

    def showEvent(self, ev):
        super().showEvent(ev)
        self._remeasure()

    def changeEvent(self, ev):
        super().changeEvent(ev)
        if ev.type() in (QEvent.StyleChange, QEvent.FontChange):
            self._remeasure()

    def sizeHint(self):
        return self.widget().sizeHint().expandedTo(QSize(*self.mn))

    def _fit(self):
        w = self.widget()
        mw = max(self.mn[0], w.minimumSizeHint().width())
        mh = max(self.mn[1], w.minimumSizeHint().height())
        full = self.size()
        sb = self.style().pixelMetric(self.style().PixelMetric.PM_ScrollBarExtent)
        need_h = full.width() < mw          # 가로 스크롤이 생기면 세로 여유가 줄어든다
        need_v = full.height() - (sb if need_h else 0) < mh
        need_h = full.width() - (sb if need_v else 0) < mw
        vw = full.width() - (sb if need_v else 0)
        vh = full.height() - (sb if need_h else 0)
        w.resize(max(vw, mw), max(vh, mh))

    def overflow(self):
        vp = self.viewport().size()
        return self.widget().width() > vp.width() + 1 or self.widget().height() > vp.height() + 1

    def resizeEvent(self, ev):
        super().resizeEvent(ev)
        self._fit()


class AnalysisWindow(QMainWindow):
    """'분석 창 분리' 로 분석 패널들을 옮겨 담는 별도 창 (다른 모니터로 옮길 수 있다)"""

    def __init__(self, owner):
        super().__init__()
        self.owner = owner
        self.setWindowTitle(tr("main.tng_analysis"))
        self.setObjectName("analysisWindow")
        self.setDockNestingEnabled(True)
        c = QWidget()
        c.setMaximumSize(0, 0)
        self.setCentralWidget(c)
        self.resize(1100, 800)

    def closeEvent(self, ev):
        if self.owner is not None:
            self.owner.attach_analysis()
        ev.accept()


class MainWindow(QMainWindow):
    def __init__(self, engine=None, open_input=True, rx_sync=False, restore=True,
                 settings_scope=None, settings_path=None, log_dir=None):
        super().__init__()
        __import__("engine").app_threads()                         # 27부: torch CPU 스레드 1 (실시간 수신, 바쁜 PC 에서 스레드 경합 방지)
        self.setWindowTitle("{} {} {}".format(APP_NAME, registry.APP_VERSION, registry.CHANNEL))
        self.setObjectName("mainWindow")
        self._open_input, self._rx_sync = open_input, rx_sync
        notes = []
        if settings_path is None and settings_scope is None:        # 실제 실행: 옛 위치에서 한 번 가져오기
            m = migrate_old_locations()
            if m:
                notes.append(m)
        # 레이아웃: 설정 폴더의 layout.ini (시험 스크립트는 따로 준 범위를 그대로 씀)
        self.qs = QSettings(*settings_scope) if settings_scope else QSettings(app_paths.layout_path(), QSettings.IniFormat)
        self.settings = Settings(settings_path)
        if self.settings.notice:
            notes.append(self.settings.notice)
        self.session = SessionLog(log_dir or self.settings.log_dir())
        self.session.extra = lambda: dict(app_version=registry.APP_VERSION, app_build=registry.APP_BUILD, **(self.rig_log() if hasattr(self, "vfo") else {}))
        from latency_log import LatencyLog
        self.latlog = LatencyLog(log_dir or self.settings.log_dir())     # 검출 표시 지연 요약 (항상)
        self._start_notes = notes
        self.rxlog_path = None

        self.engine = None
        self.bridge = Bridge()
        self.packets = {}                      # 패킷 id → 분석 결과 (최근 200개)
        self._pid = 0
        self._shown_pid = None
        self._in_stream, self._rx, self._in_fs, self._in_count = None, None, None, 0
        self._in_q = deque()
        self._in_level = 0.0
        self._tx_audio, self._tx_start, self._tx_dur, self._transmitting = None, None, 0.0, False
        self._tx = None                        # 송신 중: {'player','txa','text','src','cut','stop_i','t_end'}
        self.tx_dry = False                    # 시험용: 장치 없이 tx_advance() 로 재생 위치를 옮긴다
        self._ov = None                        # 워터폴 구간 표시 (현재 메시지)
        self._plan_max = None
        self._live_segs = []
        self._freq_hold = (None, 0.0, "")
        self._snr_hold = (None, 0.0)
        self._audio_ok = {"in": None, "out": None}
        self.pkt_an = None
        self.live_an = None
        self.analysis_win = None
        self.detached = False
        self.preset = "운용"
        self.engine5 = None                    # TNG5 엔진 (tng5_app.Tng5Engine)
        st = self.settings
        self.tx_mode = st["last_tx_mode"] if st["tx_start_mode"] == "last" else st["tx_start_mode"]
        if self.tx_mode not in registry.ENABLED:
            self.tx_mode = "TNG44"
        self._live_mode = "TNG44"
        self._bad_ok = None                    # TNG5 집합 밖 글자 경고를 본 텍스트 (같은 텍스트 두 번째 송신은 진행)

        QApplication.instance().setFont(T.ui_font())   # 그래프 안 글자 · 축 제목도 같은 글꼴
        self.setStyleSheet(T.load_qss())
        self.setDockNestingEnabled(True)
        c = QWidget()
        c.setMaximumSize(0, 0)                  # 가운데 위젯 없이 도킹 패널만
        self.setCentralWidget(c)
        self._build_panels()
        self._build_docks()
        self._build_toolbar()
        self._build_statusbar()
        self._wire()

        # 기본 창 크기: 1600x980, 화면(작업 표시줄 뺀 영역)이 더 작으면 화면에 맞춘다 (배율 125% 등)
        scr = QApplication.primaryScreen().availableGeometry() if QApplication.primaryScreen() else None
        self.resize(min(1600, scr.width()) if scr else 1600, min(980, scr.height() - 40) if scr else 980)
        if not (restore and self._restore_layout()):
            self.apply_preset("운용")
        # 창이 화면에 뜬 뒤(스타일 · 글꼴이 적용된 뒤) 같은 함수로 한 번 더 — 시작과 버튼 전환이 같은 경로
        QTimer.singleShot(0, lambda: self.apply_preset(self.preset))

        self._fill_hostapis()
        self.set_busy(True)
        if engine is not None:
            self.bridge.engine_ready.emit(engine, "재사용")
        else:
            self.show_status(tr("main.loading_models"))
            threading.Thread(target=self._load_engine, daemon=True).start()

        self._ui_timer = QTimer(self)
        self._ui_timer.timeout.connect(self._ui_tick)
        self._ui_timer.start(int(1000 / UI_FPS))
        self._sec_timer = QTimer(self)
        self._sec_timer.timeout.connect(self._sec_tick)
        self._sec_timer.timeout.connect(self._ptt_watch)            # 18부: 최대 송신 시간
        self._sec_timer.start(1000)
        self._sec_tick()

    # ================================================================== 패널 내용
    def _build_panels(self):
        s = self.settings
        # 워터폴
        self.waterfall = WaterfallWidget(cmap=s["cmap"] or "inferno")
        w = QWidget()
        v = QVBoxLayout(w)
        v.setContentsMargins(T.sp(1), T.sp(1), T.sp(1), T.sp(1))
        v.setSpacing(T.sp(1))
        v.addWidget(self.waterfall, 1)
        row = QHBoxLayout()
        row.setContentsMargins(T.sp(1), 0, 0, 0)
        row.setSpacing(T.sp(2))
        self.chk_live = QCheckBox(tr("main.live_waterfall"))
        self.chk_live.setChecked(True)
        row.addWidget(self.chk_live)
        self.chk_segs = QCheckBox(tr("main.segments"))
        self.chk_segs.setChecked(True if s["wf_segs"] is None else bool(s["wf_segs"]))
        row.addWidget(self.chk_segs)
        row.addStretch(1)
        lb = QLabel(tr("main.speed"))
        lb.setObjectName("dim")
        row.addWidget(lb)
        self.cb_wf_speed = QComboBox()
        for k_, n_ in (("느림", "wf.speed.slow"), ("보통", "wf.speed.normal"), ("빠름", "wf.speed.fast")):
            self.cb_wf_speed.addItem(tr(n_), k_)        # 값 = 설정에 저장하는 이름, 글자 = 언어 따라
        self.cb_wf_speed.setCurrentIndex(max(0, self.cb_wf_speed.findData(s["wf_speed"] or "보통")))
        self.waterfall.speed = self.cb_wf_speed.currentData()
        self.cb_wf_speed.currentIndexChanged.connect(lambda *_: self._set_wf_speed(self.cb_wf_speed.currentData()))
        row.addWidget(self.cb_wf_speed)
        lb = QLabel(tr("main.palette"))
        lb.setObjectName("dim")
        row.addWidget(lb)
        self.cb_cmap = QComboBox()
        self.cb_cmap.addItems(T.WATERFALL_CMAPS)
        self.cb_cmap.setCurrentText(self.waterfall.cmap_name)
        row.addWidget(self.cb_cmap)
        v.addLayout(row)
        self.w_waterfall = w
        # 스펙트럼
        self.spectrum = SpectrumPanel(tuple(s["spec_range"]) if s["spec_range"] else None)
        self.spectrum.rangeChanged.connect(lambda lo, hi: self.settings.__setitem__("spec_range", [lo, hi]))
        self.spectrum.link_x(self.waterfall)            # 가로축 연동 (왼쪽 축 폭은 둘 다 theme.AXIS_W)
        # VFO (18부): 다이얼 · 밴드 · 무전기 모드 · 중심 (상단바에서 옮김) · RF · 상태 · PTT · CAT
        self.sp_center = QSpinBox()
        self.sp_center.setObjectName("dbfs")
        self.sp_center.setSuffix(" Hz")
        self.sp_center.setSingleStep(10)
        self.sp_center.setKeyboardTracking(False)
        from widgets_vfo import VfoPanel
        self.vfo = VfoPanel(self.sp_center, s["bands"])
        self.dial_hz, self.rig_mode = int(s["dial_hz"] or 14078000), s["rig_mode"] or "USB"
        self.rig, self._ptt_manual, self._ptt_test = None, False, False
        self._fake_split = None
        # 미터
        m = QWidget()
        g = QVBoxLayout(m)
        g.setContentsMargins(T.sp(2), T.sp(2), T.sp(2), T.sp(2))
        g.setSpacing(T.sp(1))
        self.m_level = LEDMeter(tr("meter.level"), -60, 0, "dBFS", [(-12, "ok"), (-3, "fec"), (0, "err")], fmt="{:.1f}")
        self.m_snr = LEDMeter(tr("meter.snr"), -15, 30, "dB",
                              [(FAIL_SNR_DB + 3, "err"), (FAIL_SNR_DB + 6, "fec"), (30, "ok")])
        self.m_freq = CenterMeter(tr("meter.freq"), 100, "Hz")
        self.m_margin = MarginGauge()
        for x in (self.m_level, self.m_snr, self.m_freq, self.m_margin):
            g.addWidget(x)
        g.addStretch(1)
        self.w_meters = m
        # 채널 품질 · 통계
        self.q_grid = StatGrid([("ber", tr("main.hard_ber")), ("fec", tr("main.fec_fixed_checked")),
                                ("snr", tr("meter.snr")), ("evm", tr("main.evm_after_eq")),
                                ("rho", tr("main.costas_a")), ("drift", tr("main.residual_drift"))])
        self.s_grid = StatGrid([("rx", tr("main.rx_packets")), ("rate", tr("main.success_rate")), ("snr", tr("main.avg_snr")), ("tx", tr("main.tx"))])
        q = QWidget()
        qv = QVBoxLayout(q)
        qv.setContentsMargins(0, 0, 0, 0)
        qv.setSpacing(0)
        qv.addWidget(self.q_grid)
        qv.addWidget(hline())
        cap = QLabel(tr("main.session_stats"))
        cap.setObjectName("section")
        qv.addWidget(cap)
        qv.addWidget(self.s_grid)
        qv.addStretch(1)
        self.w_quality = q
        # 수신 로그
        r = QWidget()
        rv = QVBoxLayout(r)
        rv.setContentsMargins(0, 0, 0, T.sp(1))
        rv.setSpacing(T.sp(1))
        self.table = RxLog()
        self.table.rf_source = lambda: self.rf_hz() if hasattr(self, "vfo") else None
        rv.addWidget(self.table, 1)
        rv.addWidget(hline())
        rr = QHBoxLayout()
        rr.setContentsMargins(T.sp(1), 0, T.sp(1), 0)
        rr.setSpacing(T.sp(3))
        self.btn_open = QPushButton(tr("main.open_wav"))
        self.chk_rx = QCheckBox(tr("main.live_rx"))
        self.chk_rx.setChecked(True)
        self.chk_loop = QCheckBox(tr("main.loopback_rx"))
        self.btn_csv = QPushButton(tr("main.export_csv"))
        for x in (self.btn_open, self.chk_rx, self.chk_loop):
            rr.addWidget(x)
        rr.addStretch(1)
        rr.addWidget(self.btn_csv)
        rv.addLayout(rr)
        self.w_rx = r
        # 송신
        t = QWidget()
        tg = QGridLayout(t)
        tg.setContentsMargins(T.sp(2), T.sp(2), T.sp(2), T.sp(2))
        tg.setHorizontalSpacing(T.sp(2))
        tg.setVerticalSpacing(T.sp(2))
        self.tx_ind = TXIndicator()
        tg.addWidget(self.tx_ind, 0, 0, 1, 3)
        self.ed_my = QLineEdit(s["mycall"] or "")
        self.ed_my.setObjectName("call")
        self.ed_my.setPlaceholderText(tr("main.e_g_6l5tng"))
        self.ed_dx = QLineEdit(s["dxcall"] or "")
        self.ed_dx.setObjectName("call")
        self.ed_dx.setPlaceholderText(tr("main.dx_call"))
        mono_when_filled(self.ed_my)
        mono_when_filled(self.ed_dx)
        calls = QHBoxLayout()
        calls.setSpacing(T.sp(2))
        for lab, ed in ((tr("main.dx_call"), self.ed_dx),):          # 내 호출부호: 설정 창 '일반' 
            lb = QLabel(lab)
            lb.setObjectName("dim")
            calls.addWidget(lb)
            calls.addWidget(ed, 1)
        tg.addLayout(calls, 1, 0, 1, 3)
        self.macros = MacroBar(s["macros"] or [])
        tg.addWidget(self.macros, 2, 0, 1, 3)
        self.chk_macro_now = QCheckBox(tr("main.tx_now"))
        self.chk_macro_now.setChecked(bool(s["macro_now"]))
        tg.addWidget(self.chk_macro_now, 5, 0, 1, 3)
        self.tx_edit = QLineEdit()
        self.tx_edit.setObjectName("txText")
        self.tx_edit.setPlaceholderText(tr("main.message"))
        mono_when_filled(self.tx_edit)
        tg.addWidget(self.tx_edit, 3, 0)
        self.btn_tx = QPushButton(tr("main.tx"))
        self.btn_tx.setObjectName("txButton")
        self.btn_save = QPushButton(tr("main.save_wav"))
        tg.addWidget(self.btn_tx, 3, 1)
        tg.addWidget(self.btn_save, 3, 2)
        vol = QHBoxLayout()
        vol.setSpacing(T.sp(2))
        lb = QLabel(tr("main.tx_level"))
        lb.setObjectName("dim")
        vol.addWidget(lb)
        self.vol = QSlider(Qt.Horizontal)
        self.vol.setRange(0, 100)
        self.vol.setValue(int(s["volume"]))
        self.vol_label = QLabel("{}%".format(int(s["volume"])))
        self.vol_label.setObjectName("num")
        self.vol_label.setMinimumWidth(36)
        self.vol_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        vol.addWidget(self.vol, 1)
        vol.addWidget(self.vol_label)
        tg.addLayout(vol, 4, 0, 1, 3)
        tg.setColumnStretch(0, 1)
        tg.setRowStretch(6, 1)
        self.w_tx = t
        # 오디오 설정
        a = QWidget()
        ag = QGridLayout(a)
        ag.setContentsMargins(T.sp(2), T.sp(2), T.sp(2), T.sp(2))
        ag.setHorizontalSpacing(T.sp(2))
        ag.setVerticalSpacing(T.sp(1))
        self.cb_api, self.cb_in, self.cb_out = QComboBox(), QComboBox(), QComboBox()
        for i, (lab, cb) in enumerate(((tr("main.audio_api"), self.cb_api), (tr("main.input"), self.cb_in),
                                       (tr("main.output"), self.cb_out))):
            lb = QLabel(lab)
            lb.setObjectName("dim")
            ag.addWidget(lb, i, 0)
            ag.addWidget(cb, i, 1)
            cb.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        ag.setRowStretch(3, 1)
        self.w_audio = a
        # 타임라인 · 추이 · 분석
        self.timeline = TimelinePanel()
        self.trend = TrendPanel()
        self.p_const = ConstellationPanel()
        self.p_eye = EyePanel()
        self.p_sync = SyncMapPanel(0.18, cmap=s["sync_cmap"] or T.SYNC_CMAP_DEFAULT,
                                   smooth=True if s["sync_smooth"] is None else bool(s["sync_smooth"]))
        self.p_sync.cmapChanged.connect(lambda n: self.settings.__setitem__("sync_cmap", n))
        self.p_sync.smoothChanged.connect(lambda on: self.settings.__setitem__("sync_smooth", bool(on)))
        self.p_llr = LLRPanel()
        self.p_feat = FeaturePanel()
        self.view_toggle = ViewToggle([("eq", tr("main.eq"), tr("main.signal_restored_by_channel")),
                                       ("raw", tr("main.raw"), tr("main.rx_signal_with_freq"))])
        cw = QWidget()
        cv = QVBoxLayout(cw)
        cv.setContentsMargins(0, 0, 0, 0)
        cv.setSpacing(0)
        cv.addWidget(self.view_toggle)
        cv.addWidget(self.p_const, 1)
        self.w_const = cw
        # 패킷 해부 · 대역 확대 워터폴
        self.p_dissect = DissectPanel()
        from widgets_dissect5 import Dissect5Panel
        self.p_dissect5 = Dissect5Panel()
        self.p_dissect5.setVisible(False)
        self.w_dissect = QWidget()
        lay_ = QVBoxLayout(self.w_dissect)
        lay_.setContentsMargins(0, 0, 0, 0)
        lay_.setSpacing(0)
        lay_.addWidget(self.p_dissect, 1)
        lay_.addWidget(self.p_dissect5, 1)
        from analysis import ZoomAnalyzer
        self.zoom_an = ZoomAnalyzer(self._live_source)
        zp = s["zoom_preset"] if s["zoom_preset"] in ZoomAnalyzer.PRESETS else ZoomAnalyzer.DEFAULT
        self.zoom_an.speed = s["zoom_speed"] or "보통"
        self.zoom_an.set_preset(zp)
        tones = []                                                  # 모드별 코스타스 톤 안내선은 _zoom_tones
        self.p_zoom = ZoomPanel(list(ZoomAnalyzer.PRESETS), zp, span=int(s["zoom_span"] or 300),
                                rng=tuple(s["zoom_range"]) if s["zoom_range"] else None,
                                cmap=s["cmap"] or "inferno", tones=tones, speed=self.zoom_an.speed)
        self._zoom_tones(self.tx_mode)
        self.p_zoom.speedChanged.connect(self._set_zoom_speed)
        self.p_zoom.presetChanged.connect(self._zoom_preset)
        self.p_zoom.spanChanged.connect(lambda v: self.settings.__setitem__("zoom_span", int(v)))
        self.p_zoom.rangeChanged.connect(lambda lo, hi: self.settings.__setitem__("zoom_range", [lo, hi]))
        # 수신 · 송신 · 송신 해부
        self.qrx = QsoRxPanel(int(s["qso_font"] or 18))
        self.qrx.sizeChanged.connect(lambda v: self.settings.__setitem__("qso_font", int(v)))
        self.qrx.btn_stop.clicked.connect(self._stop_rx_panel)
        self.qtx = QsoTxPanel()
        self.txana = TxAnalysisPanel()
        # TNG1 전문가 패널 (모드가 TNG1 일 때 기존 패널 자리에 대신 보임)
        from PySide6.QtWidgets import QStackedWidget
        from widgets_tng1 import ToneGridPanel, ConfPanel, FoldPanel, SyncTonesPanel, Dissect1Panel, TxDissect1Panel
        self.p_tones = ToneGridPanel()
        self.p_conf1, self.p_fold1, self.p_sync1 = ConfPanel(), FoldPanel(), SyncTonesPanel()
        self.p_dissect1, self.p_txd1 = Dissect1Panel(), TxDissect1Panel()
        self.p_tones.blockClicked.connect(self._tng1_block_clicked)
        self._stk = {}
        for key_, a_, b_ in (("const", self.w_const, self.p_conf1), ("eye", self.p_eye, self.p_fold1),
                             ("sync", self.p_sync, self.p_sync1), ("dissect", self.w_dissect, self.p_dissect1),
                             ("txana", self.txana.encode, self.p_txd1)):
            st_ = QStackedWidget()
            st_.addWidget(a_)
            st_.addWidget(b_)
            self._stk[key_] = st_
        self._expert1 = False
        self._t1_tick = 0.0

    # ================================================================== 도킹
    def _build_docks(self):
        spec = [("spectrum", tr("main.fft_spectrum"), "live", self.spectrum),
                ("vfo", tr("vfo.title"), None, self.vfo),
                ("waterfall", tr("main.waterfall"), "live", self.w_waterfall),
                ("meters", tr("main.meters"), "live", self.w_meters),
                ("timeline", tr("main.packet_timeline"), "packet", self.timeline),
                ("rx", tr("main.rx_log"), None, self.w_rx),
                ("tx", tr("set.page.macro"), None, self.w_tx),
                ("qrx", tr("main.rx"), "live", self.qrx),
                ("qtx", tr("main.tx"), None, self.qtx),
                ("txmon", tr("main.tx_monitor"), None, self.txana.monitor),
                ("txana", tr("main.tx_anatomy"), None, self._stk["txana"]),
                ("quality", tr("main.channel_quality_stats"), "packet", self.w_quality),
                ("trend", tr("main.trend"), "packet", self.trend),
                ("const", tr("main.iq_constellation"), "packet", self._stk["const"]),
                ("eye", tr("main.eye_diagram"), "packet", self._stk["eye"]),
                ("feat", tr("main.nn_features_phase_trace"), "packet", self.p_feat),
                ("llr", tr("main.llr_distribution"), "packet", self.p_llr),
                ("sync", tr("main.costas_sync_map"), "live", self._stk["sync"]),
                ("tones", tr("main.16_tone_grid_tng1"), "live", self.p_tones),
                ("dissect", tr("main.packet_anatomy"), "packet", self._stk["dissect"]),
                ("zoom", tr("main.zoom_waterfall"), "live", self.p_zoom)]
        self.docks, self.titles = {}, {}
        for key, title, kind, w in spec:
            d = QDockWidget(title, self)
            d.setObjectName("dock_" + key)
            d.setFeatures(QDockWidget.DockWidgetMovable | QDockWidget.DockWidgetFloatable |
                          QDockWidget.DockWidgetClosable)
            t = DockTitle(d, title, kind, HELP.get(key))
            d.setTitleBarWidget(t)
            d.setWidget(self._scroll_body(key, w))
            self.docks[key], self.titles[key] = d, t
            self.addDockWidget(Qt.LeftDockWidgetArea, d)
        self.docks["zoom"].visibilityChanged.connect(lambda on: setattr(self.zoom_an, "active", bool(on)))
        self._set_expert_mode(self.tx_mode)
        QTimer.singleShot(2500, self._check_models)

    def _scroll_body(self, key, w):
        """패널 내용을 스크롤 영역에 담는다: 내용은 PANEL_MIN 아래로 눌리지 않고, 모자라면 스크롤"""
        w.setObjectName(w.objectName() or "dockBody")
        w.setAttribute(Qt.WA_StyledBackground, True)
        return DockScroll(w, PANEL_MIN[key])

    def titles_truncated(self):
        """보이는 패널 중 제목줄 글자가 잘린 것 (이름 목록)"""
        return [k for k, d in self.docks.items() if d.isVisible() and self.titles[k].truncated()]

    def panel_squeezed(self):
        """보이는 패널 중 창이 모자라 내용이 스크롤로 넘어간 것 (이름 목록)"""
        return [k for k, d in self.docks.items()
                if d.isVisible() and not d.isFloating() and d.widget().overflow()]

    def apply_preset(self, name):
        """레이아웃 프리셋: 운용 / 분석 / 전체"""
        self.preset = name
        D = self.docks
        for d in D.values():                          # 최소 크기를 지금(스타일 적용 뒤) 값으로 다시 잰다
            d.widget()._remeasure()
        for b in self.preset_btns.buttons():
            b.setChecked(b.property("preset") == name)
        L, R = Qt.LeftDockWidgetArea, Qt.RightDockWidgetArea
        H, V = Qt.Horizontal, Qt.Vertical
        keep = {k for k in FLOAT_KEYS if D[k].isFloating() and D[k].isVisible()}
        for k, d in D.items():
            if not (self.detached and k in ANALYSIS_KEYS) and k not in keep:
                self.removeDockWidget(d)
        used = set()

        def ok(k):
            return not (self.detached and k in ANALYSIS_KEYS)

        def add(area, key):
            if not ok(key):
                return
            self.addDockWidget(area, D[key])
            D[key].setFloating(False)
            D[key].show()
            used.add(key)

        def split(a, b, o):
            if not (ok(a) and ok(b)):
                return
            self.splitDockWidget(D[a], D[b], o)
            D[b].show()
            used.add(b)

        def sizes(keys, vals, o):
            """
            비율로 나눈 크기를 최소 크기(+제목줄) 아래로 내리지 않는다.
            값 하나를 None 으로 두면 그 패널이 남는 길이를 전부 가진다 (합이 창에 딱 맞아 다른 패널이 눌리지 않게).
            """
            ks = [k for k in keys if ok(k) and k in used]
            if len(ks) < 2:
                return
            full = W if o == H else Hh
            mins = {k: PANEL_MIN[k][0] if o == H else PANEL_MIN[k][1] + T.TITLE_H for k in ks}
            vs = {k: None if v is None else max(int(v), mins[k]) for k, v in zip(keys, vals) if k in ks}
            flex = [k for k in ks if vs[k] is None]
            if flex:
                rest = full - sum(v for v in vs.values() if v is not None) - SEP * (len(ks) - 1)
                vs[flex[0]] = max(rest, mins[flex[0]])
            self.resizeDocks([D[k] for k in ks], [vs[k] for k in ks], o)

        # 도킹 영역 크기 = 창 - 도구 막대 - 상태 막대
        W = self.width()
        Hh = self.height() - self.toolbar.sizeHint().height() - self.statusBar().sizeHint().height()
        if name == "운용":
            # 워터폴 + 수신 패널 + 송신 패널 (전문가 패널은 숨김 — '패널 보기' 에서 열기)
            add(L, "spectrum")
            split("spectrum", "waterfall", V)
            add(R, "vfo")                              # 18부: 중심 입력이 상단바에서 VFO 로 옮겨져 운용에도 둠
            split("vfo", "qrx", V)
            split("qrx", "qtx", V)
            sizes(["spectrum", "vfo"], [W * 0.40, None], H)
            sizes(["spectrum", "waterfall"], [Hh * .18, None], V)
            sizes(["vfo", "qrx", "qtx"], [0, None, Hh * .34], V)
        elif name == "분석":
            if self.detached:          # 분석 패널이 별도 창에 있으면 본창은 운용 요소만 크게
                add(L, "waterfall")
                split("waterfall", "timeline", V)
                split("timeline", "rx", V)
                add(R, "quality")
                split("quality", "meters", V)
                split("meters", "tx", V)
                for extra in ("txmon", "txana"):
                    self.tabifyDockWidget(D["tx"], D[extra])
                    D[extra].show()
                    used.add(extra)
                D["tx"].raise_()
            else:
                # 세 열: [성상도 / NN 특징 / 타임라인] [아이 / LLR / 수신 메시지] [추이 / 동기 맵 / 품질]
                add(L, "const")
                split("const", "eye", H)
                split("const", "feat", V)
                split("feat", "timeline", V)
                split("eye", "llr", V)
                split("llr", "rx", V)
                add(R, "trend")
                split("trend", "sync", V)
                split("sync", "quality", V)
                sizes(["const", "eye", "trend"], [W * .29, W * .36, None], H)
                sizes(["const", "feat", "timeline"], [None, Hh * .38, 0], V)
                sizes(["eye", "llr", "rx"], [None, Hh * .3, Hh * .28], V)
                sizes(["trend", "sync", "quality"], [None, Hh * .3, Hh * .26], V)
                # 송신 해부 · 매크로는 오른쪽 열 탭 (추이 · 품질 뒤)
                for base, extra in (("trend", "txmon"), ("trend", "txana"), ("quality", "tx")):
                    if ok(base) and base in used:
                        self.tabifyDockWidget(D[base], D[extra])
                        D[extra].show()
                        used.add(extra)
                        D[base].raise_()
        else:   # 전체: 세 열 (실시간 · 메시지 | 분석 2x3 | 미터 · 송신)
            add(L, "spectrum")
            split("spectrum", "const", H)          # 가운데 열을 먼저 세로 전체로 만든다
            split("spectrum", "waterfall", V)
            split("waterfall", "timeline", V)
            split("timeline", "rx", V)
            split("const", "feat", V)
            split("feat", "sync", V)
            split("const", "eye", H)
            split("feat", "llr", H)
            split("sync", "trend", H)
            add(R, "vfo")                          # 18부: VFO 는 미터 바로 위
            split("vfo", "meters", V)
            split("meters", "quality", V)
            split("quality", "tx", V)
            sizes(["spectrum", "const", "vfo"], [W * .37, None, W * .26], H)
            sizes(["const", "eye"], [W * .2, W * .2], H)
            sizes(["spectrum", "waterfall", "timeline", "rx"], [Hh * .18, None, 0, Hh * .24], V)
            sizes(["const", "feat", "sync"], [Hh * .28, Hh * .28, None], V)
            sizes(["sync", "trend"], [W, W], H)          # 아래 줄 두 패널을 같은 폭으로
            sizes(["vfo", "meters", "quality", "tx"], [0, 0, None, 0], V)
            # 수신 · 송신은 수신 로그 · 매크로 탭, 송신 해부은 추이 탭
            for base, extra in (("rx", "qrx"), ("tx", "qtx"), ("trend", "txmon"), ("trend", "txana")):
                if ok(base) and base in used:
                    self.tabifyDockWidget(D[base], D[extra])
                    D[extra].show()
                    used.add(extra)
                    D[base].raise_()
        for k, d in D.items():
            if k not in used and ok(k) and k not in keep:
                # 레이아웃 안에 둔 채 숨긴다 — 빼 버리면 저장 상태에 안 남아 복원 때 다시 보인다
                self.addDockWidget(R, d)
                d.hide()
        for b in self.preset_btns.buttons() if hasattr(self, "preset_btns") else []:
            b.setChecked(b.property("preset") == name)
        self.show_status(tr("main.layout").format(tr(LAYOUT_KEY.get(name, name))))

    def detach_analysis(self):
        """분석 패널 묶음을 별도 창으로"""
        if self.detached:
            return
        if self.analysis_win is None:
            self.analysis_win = AnalysisWindow(self)
            self.analysis_win.setStyleSheet(T.load_qss())
        aw, D = self.analysis_win, self.docks
        for k in ANALYSIS_KEYS:
            self.removeDockWidget(D[k])
        L, H, V = Qt.LeftDockWidgetArea, Qt.Horizontal, Qt.Vertical
        aw.addDockWidget(L, D["const"])
        aw.splitDockWidget(D["const"], D["eye"], H)
        aw.splitDockWidget(D["const"], D["feat"], V)
        aw.splitDockWidget(D["eye"], D["llr"], V)
        aw.splitDockWidget(D["feat"], D["sync"], V)
        aw.splitDockWidget(D["llr"], D["trend"], V)
        for k in ANALYSIS_KEYS:
            D[k].setFloating(False)
            D[k].show()
        self.detached = True
        if hasattr(self, "btn_detach"):
            self.btn_detach.setChecked(True)
        aw.show()
        self.apply_preset(self.preset)

    def attach_analysis(self):
        if not self.detached:
            return
        aw, D = self.analysis_win, self.docks
        for k in ANALYSIS_KEYS:
            aw.removeDockWidget(D[k])
            self.addDockWidget(Qt.LeftDockWidgetArea, D[k])
        self.detached = False
        self.btn_detach.setChecked(False)
        aw.hide()
        self.apply_preset(self.preset)

    # ================================================================== 저장 / 복원
    def _save_layout(self):
        q = self.qs
        q.setValue("geometry", self.saveGeometry())
        q.setValue("state", self.saveState(LAYOUT_VER))
        q.setValue("preset", self.preset)
        q.setValue("detached", self.detached)
        if self.analysis_win is not None:
            q.setValue("ana_geometry", self.analysis_win.saveGeometry())
            q.setValue("ana_state", self.analysis_win.saveState(LAYOUT_VER))
        q.sync()

    def _restore_layout(self):
        q = self.qs
        st = q.value("state")
        if st is None:
            return False
        self.preset = q.value("preset", "운용")
        if str(q.value("detached", "false")).lower() == "true":
            self.detach_analysis()
            g = q.value("ana_geometry")
            if g is not None:
                self.analysis_win.restoreGeometry(g)
            s2 = q.value("ana_state")
            if s2 is not None:
                self.analysis_win.restoreState(s2, LAYOUT_VER)
        g = q.value("geometry")
        if g is not None:
            self.restoreGeometry(g)
        # 패널 배치는 restoreState 로 되살리지 않고, 프리셋 버튼과 같은 apply_preset 으로 적용한다.
        # (restoreState 는 창이 뜨기 전 · 스타일 적용 전에 잰 큰 최소 크기를 그대로 굳혀서, 처음 켠 창이
        #  줄어들지 않았다. 프리셋 버튼을 누르면 풀리던 것과 같은 원인.) 실제 적용은 창이 뜬 직후 한 번 더.
        self.apply_preset(self.preset if self.preset in ("운용", "분석", "전체") else "운용")
        return True

    # ================================================================== 도구 막대 · 상태 막대
    def _build_toolbar(self):
        tb = QToolBar(tr("main.tools"))
        tb.setObjectName("toolbar")
        tb.setMovable(False)
        self.addToolBar(Qt.TopToolBarArea, tb)
        self.toolbar = tb
        lab = QLabel(tr("main.layout2"))
        lab.setObjectName("toolLabel")
        tb.addWidget(lab)
        self.preset_btns = QButtonGroup(self)
        bl = []
        for name, tip in (("운용", "워터폴 · 수신 · 송신 중심 (전문가 패널 숨김)"),
                          ("분석", "성상도 · 아이 · 동기 맵 · LLR · 추이 · 수신 로그 · 송신 모니터 · 송신 해부 · 매크로"),
                          ("전체", "큰 화면이나 듀얼 모니터용, 전부 표시")):
            b = QToolButton()
            b.setText(tr(LAYOUT_KEY[name]))
            b.setProperty("preset", name)                  # 프리셋 이름 (화면 글자는 언어 따라)
            b.setCheckable(True)
            b.clicked.connect(lambda _=False, n=name: self.apply_preset(n))
            self.preset_btns.addButton(b)
            bl.append(b)
        tb.addWidget(segment(bl))
        tb.addSeparator()
        self.btn_detach = QToolButton()
        self.btn_detach.setText(tr("main.detach_analysis"))
        self.btn_detach.setCheckable(True)
        self.btn_detach.clicked.connect(lambda on: self.detach_analysis() if on else self.attach_analysis())
        view = QToolButton()
        view.setText(tr("main.panels"))
        view.setPopupMode(QToolButton.InstantPopup)
        menu = QMenu(view)
        for k, d in self.docks.items():
            if k in FLOAT_KEYS:
                act = menu.addAction(d.windowTitle())
                act.setCheckable(True)
                act.triggered.connect(lambda on, k_=k: self.show_float(k_) if on else self.docks[k_].hide())
                d.visibilityChanged.connect(lambda on, a_=act: a_.setChecked(on))
            else:
                menu.addAction(d.toggleViewAction())
        view.setMenu(menu)
        b = QToolButton()
        b.setText(tr("main.log_folder"))
        b.clicked.connect(lambda: os.startfile(os.path.dirname(self.session.path)))
        self.btn_settings = QToolButton()
        self.btn_settings.setText(tr("main.settings"))
        self.btn_settings.clicked.connect(self.open_settings)
        tb.addWidget(segment([self.btn_detach, view, b, self.btn_settings]))
        tb.addSeparator()
        lab = QLabel(tr("set.page.mode"))
        lab.setObjectName("toolLabel")
        tb.addWidget(lab)
        self.mode_btns = QButtonGroup(self)
        self.mode_btn = {}
        for name in ("TNG44", "TNG5", "TNG1"):
            p_ = next((x for x in registry.PROTOCOLS if x.name == name), None)
            b = QToolButton()
            b.setText(name)
            b.setCheckable(True)
            if name in registry.ENABLED:
                b.setToolTip(tr("main.g_cps_hz").format(registry.protocol(name).spec.cps_data, registry.protocol(name).spec.bw_hz))
                b.clicked.connect(lambda _=False, n=name: self.set_tx_mode(n))
            else:
                b.setEnabled(False)
                b.setToolTip(tr("main.not_used"))
            self.mode_btns.addButton(b)
            self.mode_btn[name] = b
        self.mode_btn[self.tx_mode].setChecked(True)
        tb.addWidget(segment(list(self.mode_btn.values())))
        self.center_hz = float(self.settings["center_hz"] or 1500.0)       # 16부: 송수신 중심
        self._rx_shift = None
        self._center_limits()
        self.sp_center.valueChanged.connect(self.set_center)
        self.waterfall.freqClicked.connect(self.set_center)
        self.vfo.dialChanged.connect(self._set_dial)
        self.vfo.pttPressed.connect(self.ptt_manual)
        self.bridge.rig_state.connect(self._on_rig_state)
        self._update_rf()
        QTimer.singleShot(0, self._rig_start)
        sp = QWidget()
        sp.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        tb.addWidget(sp)
        self.btn_help = QToolButton()
        self.btn_help.setText("Help")
        self.btn_help.clicked.connect(self.open_help)
        self.btn_about = QToolButton()
        self.btn_about.setText("About")
        self.btn_about.clicked.connect(self.open_about)
        tb.addWidget(segment([self.btn_help, self.btn_about]))

    def _build_statusbar(self):
        sb = self.statusBar()
        sb.setSizeGripEnabled(False)
        self.lb_msg = ElideLabel("")
        self.lb_msg.setObjectName("statusMsg")
        sb.addWidget(self.lb_msg, 1)
        self.lb_mode = QLabel("")
        self.lb_mode.setObjectName("statusInfo")
        self.lb_audio = QLabel("")
        self.lb_audio.setObjectName("statusInfo")
        self.lb_utc = QLabel("")
        self.lb_utc.setObjectName("clock")
        self.lb_local = QLabel("")
        self.lb_local.setObjectName("clock")
        for w in (self.lb_mode, self.lb_audio, self.lb_utc, self.lb_local):
            sb.addPermanentWidget(w)

    def _check_models(self):
        """앱 시작 때: 모델 파일 · TNG1 문자표 식별값이 registry 와 다르면 상태줄 경고 (다른 모델로는 교신 불가)"""
        bad = registry.model_check()
        if bad:
            self.show_status(tr("main.model_id_mismatch") + " · ".join(
                "{} (등록 {} / 파일 {})".format(m.name, m.expected[:8], (a or "없음")[:8]) for m, a in bad), True)

    def show_status(self, msg, error=False):
        if self.lb_msg.property("error") != error:
            self.lb_msg.setProperty("error", error)
            self.lb_msg.style().unpolish(self.lb_msg)
            self.lb_msg.style().polish(self.lb_msg)
        self.lb_msg.setText((tr("main.error_prefix") if error else "") + trf(msg))

    def status_text(self):
        return self.lb_msg.text()

    def set_busy(self, busy):
        for w in (self.btn_open, self.btn_save, self.btn_tx, self.qtx.btn_send):
            w.setEnabled(not busy)

    # ================================================================== 연결
    def _wire(self):
        b = self.bridge
        b.engine_ready.connect(self._on_engine_ready)
        b.decoded.connect(self._on_decoded)
        b.status.connect(self.show_status)
        b.rx_status.connect(self._on_rx_status)
        b.saved.connect(lambda p, d: self.show_status(tr("main.saved").format(p, fmt_clock(d))))
        b.rt_marker.connect(self._on_marker)
        b.rt_marker.connect(lambda m: self.latlog.event(m, "ui"))
        b.live.connect(self._on_live)
        self.qtx.sendRequested.connect(self._send_qso)
        self.qtx.stopRequested.connect(self._stop_tx)
        self.qtx.resendRequested.connect(self._resend_qso)
        self.qtx.set_resend(self.tx_mode == "TNG1", False)
        self.chk_segs.toggled.connect(self._segs_toggled)
        self.chk_macro_now.toggled.connect(lambda on: self.settings.__setitem__("macro_now", bool(on)))
        self.btn_open.clicked.connect(self._ask_open)
        self.btn_save.clicked.connect(self._ask_save)
        self.btn_tx.clicked.connect(self._toggle_tx)
        self.btn_csv.clicked.connect(self._ask_csv)
        self.tx_edit.returnPressed.connect(self._toggle_tx)
        self.vol.valueChanged.connect(lambda v: self.vol_label.setText("{}%".format(v)))
        self.vol.sliderReleased.connect(lambda: self.settings.__setitem__("volume", self.vol.value()))
        self.cb_in.currentIndexChanged.connect(lambda _: self._remember_device())
        self.cb_out.currentIndexChanged.connect(lambda _: self._remember_device())
        self.cb_api.currentIndexChanged.connect(self._fill_devices)
        self.cb_in.currentIndexChanged.connect(self._restart_input)
        self.chk_live.toggled.connect(self._on_live_toggled)
        self.chk_rx.toggled.connect(lambda _: self._ensure_rx())
        self.cb_cmap.currentTextChanged.connect(self._set_cmap)
        self.macros.macroText.connect(self._use_macro)
        self.macros.changed.connect(lambda ms: self.settings.__setitem__("macros", ms))
        self.ed_dx.editingFinished.connect(lambda: self.settings.__setitem__("dxcall", self.ed_dx.text().strip().upper()))
        self.table.packetSelected.connect(self._select_packet)
        self.timeline.frameClicked.connect(self._select_frame)
        self.view_toggle.changed.connect(lambda v: (self.p_const.set_view(v), self.p_eye.set_view(v)))

    def _set_wf_speed(self, name):
        self.waterfall.set_speed(name)
        self.settings["wf_speed"] = name

    def _set_zoom_speed(self, name):
        self.zoom_an.set_speed(name)
        self.settings["zoom_speed"] = name

    def _set_cmap(self, name):
        self.waterfall.set_cmap(name)
        self.p_zoom.set_cmap(name)
        self.settings["cmap"] = name

    def _use_macro(self, text):
        self.tx_edit.setText(expand_macro(text, self.ed_my.text().strip().upper(),
                                          self.ed_dx.text().strip().upper(), self._snr_hold[0]))
        self.tx_edit.setFocus()
        if self.chk_macro_now.isChecked() and not self._transmitting:
            self._toggle_tx()

    def _segs_toggled(self, on):
        self.settings["wf_segs"] = bool(on)
        self.waterfall.overlay.visible = bool(on)
        self.p_zoom.chk_segs.setChecked(bool(on))
        self.waterfall.draw_overlay()

    # ================================================================== 엔진
    def _load_engine(self):
        try:
            from engine import ModemEngine
            t0 = time.time()
            eng = ModemEngine(model_path=registry.protocol("TNG44").model_path)
            self.bridge.engine_ready.emit(eng, "{:.1f} s".format(time.time() - t0))
        except Exception as ex:
            self.bridge.engine_ready.emit(None, "{}: {}".format(type(ex).__name__, ex))

    def _on_engine_ready(self, eng, msg):
        if eng is None:
            self.show_status(tr("main.model_load_failed") + msg, True)
            return
        self.engine = eng
        self.set_busy(False)
        from analysis import PacketAnalyzer, LiveAnalyzer
        self.pkt_an = PacketAnalyzer(eng)
        self.pkt_an.ready.connect(self._on_packet_analysis)
        self.live_an = LiveAnalyzer(eng, self._live_source)
        self.live_an.ready.connect(self._on_live_analysis)
        self.p_sync.thr = eng.pc.detect_threshold
        if "TNG5" in registry.ENABLED and self.engine5 is None:
            try:
                from tng5_app import Tng5Engine
                self.engine5 = Tng5Engine(device=str(eng.device))
            except Exception as ex:
                self.show_status(tr("main.tng5_model_load_failed").format(type(ex).__name__, ex), True)
        self._update_mode_label()
        QTimer.singleShot(0, self._ensure_rx)
        self.show_status(tr("main.ready").format(msg, eng.device))
        for n in self._start_notes:
            self.show_status(n, tr("main.damaged") in n)

    def _update_mode_label(self):
        eng = self.engine
        if eng is None:
            return
        # 사양 문구는 registry 한 곳에서 (상태줄 = 핵심, 도움말 = 전체). 두 모드 같은 형식
        m = self.tx_mode if self.tx_mode in registry.ENABLED else "TNG44"
        self.lb_mode.setText("{} · {:.0f} Hz center".format(registry.spec_line(m), getattr(self, "center_hz", 1500.0)))


    # ================================================================== 모드 (단일 모드 운용: 송수신 같은 모드 하나)
    def _zoom_tones(self, mode):
        """대역 확대 워터폴 코스타스 톤 안내선: TNG44 7톤 62.5 Hz · TNG5 v2 Welch12 31.25 Hz (점선) + 중간 동기 · B 7톤 41.67 Hz (파선)"""
        if mode == "TNG1":
            import tng1
            c1 = tng1.CFGS["TNG1"]
            self.p_zoom.set_tones([([CENTER + v / c1.T for v in c1.slot], "dot")], key="TNG1")
            return
        if mode == "TNG5":
            import tng5
            w = [CENTER + (k - 5.5) * tng5.SP12_HZ for k in range(12)]
            m = [CENTER + (k - 3) * tng5.PC.tone_spacing_hz for k in range(7)]
            self.p_zoom.set_tones([(w, "dot"), (m, "dash")], key="TNG5")
        else:
            self.p_zoom.set_tones([([CENTER + (k - 3) * 62.5 for k in range(7)], "dot")], key="TNG44")

    def set_tx_mode(self, name):
        """상단바 모드 버튼 (단일 모드 운용: 이 모드 하나로만 송수신). 송신 중에는 잠금. 바꾸면 수신 상태 초기화.
        설정 '시작 시 모드' 규칙대로 마지막 모드 저장"""
        if name not in registry.ENABLED:
            return
        if self._transmitting:
            self.mode_btn[self.tx_mode].setChecked(True)
            self.show_status(tr("main.mode_locked_during_tx"), True)
            return
        changed = name != self.tx_mode
        self.tx_mode = name
        self.mode_btn[name].setChecked(True)
        self.settings["last_tx_mode"] = name
        self._center_limits()
        if changed:
            self._reset_rx_mode()
        self._zoom_tones(name)
        self._set_expert_mode(name)
        self._bad_ok = None
        self.qtx.set_resend(name == "TNG1", name == "TNG1" and bool(getattr(self, "_last_tx1", None)))
        self._update_mode_label()
        self.show_status(tr("main.mode").format(name))

    EXPERT1_TITLES = {"const": (tr("main.iq_constellation"), tr("main.symbol_confidence_tng1")), "eye": (tr("main.eye_diagram"), tr("main.sync_accumulation_tng1")),
                      "sync": (tr("main.costas_sync_map"), tr("main.12_sync_tones_tng1")), "dissect": (tr("main.packet_anatomy"), tr("main.packet_anatomy_tng1")),
                      "txana": (tr("main.tx_anatomy"), tr("main.tx_anatomy_tng1"))}

    def _set_expert_mode(self, mode):
        """전문가 패널 자리: TNG1 이면 TNG1 전용 (16-GFSK · 처프에 맞춘 것), 아니면 기존 패널 그대로"""
        on = mode == "TNG1"
        if not hasattr(self, "_stk") or on == self._expert1:
            return
        self._expert1 = on
        for k, st_ in self._stk.items():
            st_.setCurrentIndex(1 if on else 0)
        for k, (a_, b_) in self.EXPERT1_TITLES.items():
            if k in self.titles:
                lab = self.titles[k].findChild(QLabel, "dockTitleText")
                if lab is not None:
                    lab.setText(b_ if on else a_)

    def _tng1_tx_snapshot(self):
        tx = self._tx
        if tx is None or tx.get("mode") != "TNG1":
            return None
        import tng1
        c1 = tng1.CFGS["TNG1"]
        txa, t_now = tx["txa"], tx["player"].t
        syms = np.concatenate([b[0] for b in txa.blocks])
        cur = int(np.floor((t_now - txa.lead) / c1.T))
        n = int(32.0 / c1.T) + 1
        i0, i1 = max(0, cur - n + 1), min(len(syms), max(cur + 1, 1))
        idx = np.arange(i0, i1)
        if len(idx) == 0:
            return None
        pos = tng1.lay(c1)[0]
        return {"t": txa.lead + idx * c1.T - t_now, "T": c1.T, "sym": syms[idx], "cur": cur - i0,
                "sync": np.isin(idx % c1.nsym, pos)}

    def _tng1_expert_tick(self):
        """TNG1 전문가 창 갱신: UI 틱마다 한 가지씩 돌아가며 (격자 · 확신도 · 동기 누적 · 동기 12음), 각 패널 초당 약 4번.
        한 틱에 몰아서 하지 않아 화면 멈춤을 줄임 (보이는 패널만)"""
        if not self._expert1:
            return
        now = time.time()
        steps = ("grid", "conf", "fold", "sync")
        last = self.__dict__.setdefault("_t1_last", {k: 0.0 for k in steps})
        k = steps[self.__dict__.get("_t1_rr", 0) % len(steps)]
        self._t1_rr = self.__dict__.get("_t1_rr", 0) + 1
        if now - last.get(k, 0.0) < 0.25:                  # 이 패널은 0.25 s 안에 이미 그림
            return
        last[k] = now
        self._t1_tick = now
        vis = lambda d: self.docks[d].isVisible()
        rx1 = getattr(self._rx, "rx1", None) if self._rx is not None else None
        if k == "grid" and vis("tones"):
            self._t1_snap = rx1.grid_snapshot() if rx1 is not None else None
            self.p_tones.set_rx(self._t1_snap)
            self.p_tones.set_tx(self._tng1_tx_snapshot())
        elif k == "conf" and vis("const"):
            snap = getattr(self, "_t1_snap", None) if vis("tones") else (rx1.grid_snapshot() if rx1 is not None else None)
            self.p_conf1.set_data(snap)
        elif k == "fold" and vis("eye") and rx1 is not None:
            self.p_fold1.set_data(rx1.fold_snapshot())
        elif k == "sync" and vis("sync") and rx1 is not None:
            sy = rx1.sync_snapshot()
            if sy is not None:
                self.p_sync1.set_data(sy)

    def _tng1_block_clicked(self, d):
        """16음 격자에서 블록 클릭: 그 블록이 든 패킷 (없으면 수신 중 메시지) 해부"""
        a = d.get("abs")
        for pid in sorted(self.packets, reverse=True):
            o = self.packets[pid]
            if o.get("format") != "TNG1":
                continue
            bl = o.get("tng1_blocks") or []
            L_in = (bl[0].get("abs_in", 0) - bl[0].get("abs_in", 0)) if bl else 0
            if any(abs(b.get("abs_in", -1e18) - a) < 0.6 * 14.72 * (self._in_fs or 48000.0) for b in bl):
                self._select_packet(pid)
                self.p_dissect1.set_packet(bl, self._dissect1_title(o), select_abs=a)
                self.docks["dissect"].show()
                self.docks["dissect"].raise_()
                return
        rx1 = getattr(self._rx, "rx1", None) if self._rx is not None else None
        bl = rx1.live_blocks() if rx1 is not None else []
        if bl:
            self.p_dissect1.set_packet(bl, "수신 중 메시지 (CRC 통과 블록)", select_abs=a)
            self.docks["dissect"].show()
            self.docks["dissect"].raise_()
        else:
            self.show_status(tr("main.no_anatomy_data").format(d.get("label", "")), True)

    @staticmethod
    def _dissect1_title(o):
        utc = datetime.fromtimestamp(o["utc"], timezone.utc).strftime("%H:%M:%S") if o.get("utc") else ""
        return "{} · {} · {}".format(utc, (o.get("text") or o.get("partial") or "")[:28].replace("\n", " ↵ "),
                                    o.get("mode") or "")

    # ------------------------------------------------------------ 중심 주파수 (16부)
    def _center_limits(self):
        """모드 대역 절반을 고려한 선택 범위로 입력칸 제한 (넘으면 안쪽으로)"""
        import freqshift
        p = registry.protocol(self.tx_mode)
        lo, hi = freqshift.center_range(p.spec.bw_hz if p and p.spec else 500)
        self.sp_center.blockSignals(True)
        self.sp_center.setRange(int(lo), int(hi))
        self.sp_center.blockSignals(False)
        self.set_center(self.center_hz)

    def set_center(self, f):
        """송수신 중심 바꾸기: 송신 소리 옮김 (_start_tx) · 수신 입력을 1500 Hz 로 되돌림 (on_input_block) · 워터폴 대역선 · 표지 · 구간 · 확대 창"""
        global CENTER
        import freqshift
        import widgets as W
        if self._transmitting:
            self.sp_center.blockSignals(True)
            self.sp_center.setValue(int(round(self.center_hz)))
            self.sp_center.blockSignals(False)
            self.show_status(tr("main.offset_locked_during_tx"), True)
            return
        f = float(min(max(round(float(f) / 10.0) * 10.0, self.sp_center.minimum()), self.sp_center.maximum()))
        changed = abs(f - getattr(self, "center_hz", 1500.0)) > 1e-6
        self.center_hz = f
        self.settings["center_hz"] = f
        self.sp_center.blockSignals(True)
        self.sp_center.setValue(int(f))
        self.sp_center.blockSignals(False)
        CENTER = f
        W.set_center(f, [self.waterfall, self.spectrum.plot])
        self.p_zoom.set_center(f)
        self._zoom_tones(self.tx_mode)
        fs = self._in_fs
        self._rx_shift = freqshift.StreamShift(fs, f - freqshift.BASE_HZ) if (fs and abs(f - freqshift.BASE_HZ) > 1e-6) else None
        if changed:
            self._reset_rx_mode()                         # 옛 중심의 표지 · 구간 · 수신 상태 비움
            self._update_mode_label()
            self.show_status(tr("main.offset_0f_hz").format(f))
        if hasattr(self, "vfo"):
            self._update_rf()

    # ------------------------------------------------------------ 무전기 CAT · PTT · VFO (18부)
    def cat_ok(self):
        return self.rig is not None and bool(self.rig.state.get("connected"))

    def rf_hz(self):
        """실제 RF (CAT 연결 때만, 19부). 없으면 None"""
        import rig
        return rig.rf_hz(self.dial_hz, self.center_hz, self.rig_mode) if self.cat_ok() else None

    def rig_log(self):
        """수신 로그 · 세션 로그에 붙이는 다이얼 · RF (CAT 없으면 비움)"""
        if not self.cat_ok():
            return {}
        return {"dial_hz": int(self.dial_hz), "rf_hz": int(round(self.rf_hz())), "rig_mode": self.rig_mode}

    def _update_rf(self):
        self.vfo.show_rf(self.rf_hz())

    def _set_dial(self, hz):
        self.dial_hz = int(hz)
        self.settings["dial_hz"] = int(hz)
        if self.rig is not None and self.rig.state["connected"]:
            self.rig.set_freq(int(hz))
        self._update_rf()

    def _set_rig_mode(self, m):
        self.rig_mode = m
        self.settings["rig_mode"] = m
        if self.rig is not None and self.rig.state["connected"]:
            self.rig.set_mode(m)
        self._update_rf()

    def rig_cfg(self):
        import rig
        c = dict(rig.DEFAULTS)
        c.update(self.settings["rig"] or {})
        if c["cat"] in ("run", "connect"):
            c["cat_via"] = c["cat"]
        return c

    def _rig_start(self):
        """설정대로 CAT · PTT 시작 (이미 있으면 닫고 새로)"""
        import rig
        if self.rig is not None:
            self.rig.close()
            self.rig = None
        self.rig = rig.Rig(self.rig_cfg(), on_state=self.bridge.rig_state.emit)
        self.rig.start()
        self._on_rig_state(dict(self.rig.state))

    def _on_rig_state(self, st):
        r = self.rig
        if r is None:
            return
        c = r.cfg
        self.vfo.set_connected(bool(st.get("connected")) and c["cat"] != "none")
        if c["cat"] == "none":
            self.vfo.show_cat(tr("main.no_cat"), "none")
        elif st.get("connected"):
            self.vfo.show_cat(tr("main.cat_connected"), "ok")
            if st.get("freq") and self._fake_split is None:
                if int(st["freq"]) != self.dial_hz:
                    self.dial_hz = int(st["freq"])
                    self.settings.data["dial_hz"] = self.dial_hz
                self.vfo.show_dial(self.dial_hz)
            if st.get("mode") and st["mode"] != self.rig_mode:
                self.rig_mode = st["mode"]
                self.settings.data["rig_mode"] = self.rig_mode
                self.vfo.show_mode(self.rig_mode)
            self._update_rf()
        else:
            self.vfo.show_cat(tr("main.cat_lost"), "err")
        self.vfo.show_tx(self._transmitting, bool(st.get("ptt")))
        # 안전: PTT 실패 · CAT 끊김 (CAT PTT) 이면 송신 중단
        if self._transmitting and self._tx is not None:
            bad = st.get("ptt_error") or ""
            if not bad and c["ptt"] == "CAT" and not st.get("connected"):
                bad = "CAT 끊김"
            if bad:
                self._tx["player"].abort()
                self._end_tx("송신 중단: " + bad)
                self.show_status(tr("main.tx_halted").format(bad), True)
        elif st.get("ptt_error") and (self._ptt_manual or self._ptt_test):
            self.show_status(st["ptt_error"], True)

    def ptt_busy(self):
        """앱 송신을 막아야 하면 이유 (PTT 테스트 · VFO PTT 버튼)"""
        if self._ptt_test:
            return "PTT 테스트 중"
        if self._ptt_manual:
            return "PTT 버튼 누름"
        return ""

    def ptt_manual(self, on):
        """VFO 'PTT' 버튼: 누르는 동안 PTT 만 (소리 없음). 앱 송신 중에는 무시"""
        if self.rig is None:
            return
        if on:
            if self._transmitting or self._ptt_test:
                self.show_status(tr("main.tx2"), True)
                return
            err = self.rig.ptt(True)
            if err:
                self.show_status("PTT: " + err, True)
                return
            self._ptt_manual, self._ptt_t0 = True, time.time()
            self.vfo.show_tx(False, True)
            self.show_status(tr("main.ptt_on"))
        elif self._ptt_manual:
            self._ptt_manual = False
            self.rig.ptt(False)
            self.vfo.show_tx(False, False)
            self.show_status(tr("main.ptt_off"))

    def ptt_test(self, secs=3.0, done=None):
        """설정 'PTT 테스트': PTT 켜고 secs 뒤 자동 해제. 결과 문구를 done(ok, msg) 로. 이 동안 앱 송신 금지"""
        if self.rig is None or self._transmitting or self._ptt_manual or self._ptt_test:
            if done:
                done(False, "transmitting or PTT in use")
            return
        err = self.rig.ptt(True)
        if err:
            if done:
                done(False, err)
            return
        self._ptt_test, self._ptt_t0 = True, time.time()
        self.vfo.show_tx(False, True)
        seen = []

        def peek():
            if self._ptt_test and self.rig is not None and self.rig.state.get("ptt"):
                seen.append(1)
        for k in range(1, int(secs * 4)):
            QTimer.singleShot(250 * k, peek)

        def off():
            if not self._ptt_test:
                return
            self.rig.ptt(False)
            self._ptt_test = False
            self.vfo.show_tx(self._transmitting, False)
            m = self.rig.cfg["ptt"]
            if done:
                if m == "VOX":
                    done(True, "VOX: TNG PKT does not key the rig (audio keys VOX)")
                else:
                    ok = bool(seen)
                    done(ok, "{} PTT on {:.0f} s, released".format(m, secs) if ok else
                         "PTT not confirmed: " + (self.rig.state.get("ptt_error") or self.rig.state.get("error") or "no response"))
        QTimer.singleShot(int(secs * 1000), off)

    def _ptt_watch(self):
        """최대 송신 시간: 앱 송신 · PTT 버튼 · PTT 테스트 모두. 넘으면 소리 멈추고 PTT 해제"""
        mx = float(self.rig_cfg()["max_tx_s"])
        if self._transmitting and self._tx is not None and self._tx.get("t_ptt") is not None \
                and time.time() - self._tx["t_ptt"] > mx:
            self._tx["player"].abort()
            self._end_tx("최대 송신 시간 초과 {:.0f} s".format(mx))
            self.show_status(tr("main.max_tx_time_exceeded").format(mx), True)
        if (self._ptt_manual or self._ptt_test) and time.time() - getattr(self, "_ptt_t0", time.time()) > mx:
            self._ptt_manual = self._ptt_test = False
            if self.rig is not None:
                self.rig.ptt(False)
            self.vfo.show_tx(False, False)
            self.show_status(tr("main.max_tx_time_exceeded").format(mx), True)

    def rx_active(self):
        """수신할 모드 = 상단바에서 고른 모드 하나 (단일 모드 운용, 09-27)"""
        return [self.tx_mode] if self.tx_mode in registry.ENABLED else ["TNG44"]

    def _reset_rx_mode(self):
        """모드 전환: 수신 상태 초기화 (수신기 새로 · 수신 패널 · 워터폴 표지 · 구간 표시 · 감지 표시 비움)"""
        self._ov = None
        self._qrx_wait = None
        self._live_segs = []
        self._mark_hold = []
        self._det1 = None
        self._det1_ov = False
        self._live_mode = self.tx_mode
        self.waterfall.clear_markers()
        self.waterfall.set_overlay([])
        self.p_zoom.set_overlay([])
        self.table.remove_id(self.LIVE1_ID)
        self.qrx.discard_current()
        self.qrx.set_meta(state=tr("main.waiting"), snr=float("nan"), df=float("nan"), seg=tr("main.seg"))
        self.qrx.btn_stop.setEnabled(False)
        if self._rx is not None:                          # 옛 수신 스레드 모두 멈추고 (_stop_rx) 고른 모드 수신기 스레드만 새로 (15부)
            self._ensure_rx()

    def _live_source(self):
        rx = self._rx
        return None if rx is None or rx.paused else (rx.ring, rx.fs)

    # ================================================================== 오디오 장치
    def _fill_hostapis(self):
        self.cb_api.blockSignals(True)
        self.cb_api.clear()
        try:
            apis = sd.query_hostapis()
        except Exception as ex:
            self.show_status(tr("main.audio_api_query_failed").format(ex), True)
            self.cb_api.blockSignals(False)
            return
        default = 0
        want = self.settings["audio_api"] or "MME"
        for i, a in enumerate(apis):
            self.cb_api.addItem(a["name"], i)
            if a["name"] == want:
                default = self.cb_api.count() - 1
        self.cb_api.setCurrentIndex(default)
        self.cb_api.blockSignals(False)
        self._fill_devices()

    def _fill_devices(self):
        api = self.cb_api.currentData()
        try:
            devs = sd.query_devices()
            din, dout = sd.default.device
        except Exception as ex:
            self.show_status(tr("main.audio_device_query_failed").format(ex), True)
            return
        for cb in (self.cb_in, self.cb_out):
            cb.blockSignals(True)
            cb.clear()
        sel_in = sel_out = None
        for i, d in enumerate(devs):
            if d["hostapi"] != api:
                continue
            if d["max_input_channels"] > 0:
                self.cb_in.addItem(d["name"], i)
                if i == din:
                    sel_in = self.cb_in.count() - 1
            if d["max_output_channels"] > 0:
                self.cb_out.addItem(d["name"], i)
                if i == dout:
                    sel_out = self.cb_out.count() - 1
        # 저장된 장치 이름 우선, 없으면 시스템 기본 장치 + 상태줄 알림
        miss = []
        for cb, key, lab in ((self.cb_in, "audio_in", "입력"), (self.cb_out, "audio_out", "출력")):
            name = self.settings[key]
            if name:
                i = cb.findText(name)
                if i >= 0:
                    if cb is self.cb_in:
                        sel_in = i
                    else:
                        sel_out = i
                else:
                    miss.append("{} '{}'".format(lab, name))
        if sel_in is not None:
            self.cb_in.setCurrentIndex(sel_in)
        if sel_out is not None:
            self.cb_out.setCurrentIndex(sel_out)
        if miss:
            self.dev_notice = "저장 장치 없음, 기본 장치 사용: " + ", ".join(miss)
            self._start_notes = getattr(self, "_start_notes", []) + [self.dev_notice]
            self.show_status(self.dev_notice)
        for cb in (self.cb_in, self.cb_out):
            cb.blockSignals(False)
        self._restart_input()

    def select_device(self, which, name_part):
        """장치 이름 일부로 입력/출력 장치를 고른다 (시험·스크립트용). 찾으면 True"""
        cb = self.cb_in if which == "in" else self.cb_out
        for i in range(cb.count()):
            if name_part.lower() in cb.itemText(i).lower():
                cb.setCurrentIndex(i)
                return True
        return False

    def _device_rate(self, dev, output, prefer=48000):
        try:
            if output:
                sd.check_output_settings(device=dev, samplerate=prefer, channels=1)
            else:
                sd.check_input_settings(device=dev, samplerate=prefer, channels=1)
            return float(prefer)
        except Exception:
            return float(sd.query_devices(dev)["default_samplerate"])

    def _restart_input(self):
        """입력 스트림: 레벨 미터 · 워터폴 · 실시간 수신이 모두 여기서 샘플을 받는다."""
        self._stop_rx()
        if self._in_stream is not None:
            try:
                self._in_stream.stop()
                self._in_stream.close()
            except Exception:
                pass
            self._in_stream = None
        dev = self.cb_in.currentData()
        if dev is None or not self._open_input:
            return
        fs = self._device_rate(dev, output=False)

        def cb(indata, frames, t, status):
            try:
                self.latlog.dev_s = float(t.currentTime - t.inputBufferAdcTime)
            except Exception:
                pass
            self.on_input_block(indata[:, 0].copy())

        try:
            self._in_stream = sd.InputStream(device=dev, channels=1, samplerate=fs,
                                             blocksize=int(fs * 0.02), callback=cb)
            self._in_q.clear()
            self._in_fs = fs
            self._in_count = 0
            self._in_stream.start()
            self._audio_ok["in"] = True
            if self.chk_live.isChecked():
                self.waterfall.live_start(fs)
            self._ensure_rx()
        except Exception as ex:
            self._in_stream = None
            self._audio_ok["in"] = False
            self.show_status(tr("main.cannot_open_input_device").format(ex), True)

    def use_synthetic_input(self, fs=48000.0):
        """실제 장치 대신 on_input_block 으로 넣는 합성 입력을 쓴다 (자가검사용)"""
        self._stop_rx()
        if self._in_stream is not None:
            self._in_stream.stop()
            self._in_stream.close()
            self._in_stream = None
        self._in_fs, self._in_count = float(fs), 0
        self._in_q.clear()
        self._audio_ok["in"] = True
        self.chk_live.setChecked(True)
        self.waterfall.live_start(fs)
        self._ensure_rx()

    def on_input_block(self, x):
        """입력 한 덩어리 (오디오 콜백 스레드). 자가검사는 이 메서드로 합성 오디오를 넣는다."""
        if len(x) == 0:
            return
        self._in_level = max(float(np.max(np.abs(x))), self._in_level * 0.85)
        rx = self._rx
        if rx is not None:
            sh = getattr(self, "_rx_shift", None)
            rx.feed(sh.process(x) if sh is not None else x)           # 16부: 선택 중심 → 1500 Hz 로 되돌려 수신기에
        self._in_count += len(x)
        self._in_q.append((self._in_count, x))
        if self._in_fs:
            self.latlog.block(self._in_count, self._in_fs)

    def _drain_input(self):
        if not self._in_q:
            return
        parts, end = [], None
        while self._in_q:
            end, x = self._in_q.popleft()
            parts.append(x)
        if self.waterfall.live:
            self.waterfall.push_samples(np.concatenate(parts).astype(np.float64), abs_end=end)

    def _on_live_toggled(self, on):
        if on:
            if self._in_fs is None:
                self.show_status(tr("main.input_device_not_open"), True)
                return
            self._in_q.clear()
            self.waterfall.live_start(self._in_fs)
            self.waterfall.abs_now = self._in_count
        else:
            self.waterfall.live = False

    # ================================================================== 실시간 수신
    def _ensure_rx(self):
        self._stop_rx()
        import freqshift
        df_ = getattr(self, "center_hz", 1500.0) - freqshift.BASE_HZ
        self._rx_shift = freqshift.StreamShift(self._in_fs, df_) if (self._in_fs and abs(df_) > 1e-6) else None   # 입력 fs 가 바뀌면 새로
        if self.engine is None or self._in_fs is None or not self.chk_rx.isChecked():
            return
        from realtime_rx import RealtimeReceiver
        b = self.bridge

        def on_status(msg, err):
            self._log_rx("상태", msg)
            b.rx_status.emit(msg, err)

        def on_marker(m):
            self._log_rx("검출", "{} {} abs={} df={:+.2f}Hz rho2={:.3f}".format(m.get("mode", "TNG44"), m["kind"], m["abs"], m["df"], m.get("rho", 0.0)))   # 표시 지움 표지 (remove) 에는 rho 가 없음
            self.latlog.event(m, "emit")
            b.rt_marker.emit(m)

        def on_decoded(r, label):
            self.latlog.decoded("TNG5" if "TNG5" in label else "TNG44")
            self._log_rx("복조", "{} {} df={} snr={} | {}".format(
                "성공" if r.ok else "실패", label, r.df_hz, r.snr_db, r.text if r.ok else r.reason))
            b.decoded.emit(r, label)

        rx = RealtimeReceiver(self.engine, self._in_fs, on_status, on_marker, on_decoded, sync=self._rx_sync)
        rx.on_live = lambda ev: (self._log_rx("구간", "{} {} {}".format(ev.get("mode", "TNG44"), ev.get("type"), len(ev.get("segs") or []))),
                                 b.live.emit(ev))
        rx.on_heartbeat = lambda d: self._log_rx("동작", "지난 1분 탐색 {} 회, 최대 rho2 {:.3f}, 입력 {:.1f} dBFS".format(
            d["scans"], d["max_rho"], d["rms_dbfs"]))
        rx.ring.count = self._in_count
        rx5 = None
        if self.engine5 is not None:
            from tng5_app import Tng5Receiver
            rx5 = Tng5Receiver(self.engine5, self._in_fs, on_status, on_marker, on_decoded, sync=self._rx_sync)
            rx5.on_live = rx.on_live
            rx5.ring.count = self._in_count
            rx5._scan_from = self._in_count
        from tng5_app import MultiRx
        multi = MultiRx(rx, rx5, [m for m in self.rx_active() if m != "TNG1"])
        if "TNG1" in registry.ENABLED:                    # TNG1: 따로 수신 스레드 (NN 없음). 켜는 것은 고른 모드 하나 (set_active)
            from tng1_app import Tng1Receiver, Multi3
            rx1 = Tng1Receiver(self._in_fs, on_status, on_marker, on_decoded, sync=self._rx_sync)
            rx1.on_live = rx.on_live
            rx1.on_late = on_decoded                      # 늦은 결합 → _on_decoded 에서 결과 표에만
            rx1.ring.count = self._in_count
            multi = Multi3(multi, rx1, self.rx_active())
        multi.set_active(self.rx_active())
        if not rx.sync:
            multi.start()
        self._rx = multi

    def _stop_rx(self):
        rx, self._rx = self._rx, None
        if rx is not None:
            rx.stop()

    # ------------------------------------------------------------ CRC 통과 전 표시 (09-26 막기 → 09-27 표지 · 구간은 바로)
    # 원칙: 해독 글자 · 결과 행 · 상태 문구는 CRC 통과 뒤. 워터폴 동기 표지 · 구간 상자는 검출 순간 바로 '확정 전' (자홍 점선, 세 모드 공통) 으로
    # 그리고, 첫 CRC 통과 때 원래 색 (실선) 으로 바꾸며, CRC 통과 없이 끝나면 조용히 지운다 (_drop_tentative). 표시 전용 (수신 경로 무관)
    MARK_HOLD_S = 120.0

    def _crc_shown(self, mode=None):
        """수신 패널이 CRC 통과한 실시간 메시지를 보여 주는 중인가 (mode 주면 그 모드)"""
        o = self._ov
        return (o is not None and o.get("live") and not o.get("pre") and getattr(self, "_qrx_wait", None) is None
                and (mode is None or o.get("mode", "TNG44") == mode))

    def _on_rx_status(self, msg, err):
        if (err and "오류" in msg) or self._crc_shown():
            self.show_status(msg, err)                    # 수신 오류는 늘 보임, 나머지는 CRC 통과 메시지 수신 중에만

    def _mark_confirmed(self, mode):
        """이 모드 메시지가 CRC 를 통과: 확정 전 (자홍 점선) 으로 그려 둔 동기 표지를 원래 색으로"""
        now = time.time()
        self.__dict__.setdefault("_mark_ok", {})[mode] = now
        keep = []
        fs = self._in_fs or 48000.0
        for t, m in self.__dict__.get("_mark_hold", []):
            if m.get("mode", "TNG44") == mode and now - t < self.MARK_HOLD_S:
                if not m.get("id"):
                    self.waterfall.add_marker(dict(m, remove=True), fs)      # id 없는 표지 (TNG44): 지우고 다시
                self._draw_marker(m)
            elif now - t < self.MARK_HOLD_S:
                keep.append((t, m))
        self._mark_hold = keep

    def _drop_tentative(self, mode):
        """CRC 통과 없이 끝난 (가짜 시작 · 미확인) 이 모드의 확정 전 표지 · 구간 표시를 조용히 지움 (로그는 그대로)"""
        fs = self._in_fs or 48000.0
        hold = self.__dict__.get("_mark_hold", [])
        for t, m in hold:
            if m.get("mode", "TNG44") == mode:
                self.waterfall.add_marker(dict(m, remove=True), fs)
        self._mark_hold = [(t, m) for t, m in hold if m.get("mode", "TNG44") != mode]
        o = self._ov
        if o is not None and o.get("mode", "TNG44") == mode and (o.get("pre") or not o.get("shown")):
            self._ov = None
            self.waterfall.set_overlay([])
            self.p_zoom.set_overlay([])
            self._ov_tick(force=True)

    def _on_marker(self, m):
        mode = m.get("mode", "TNG44")
        hold = self.__dict__.setdefault("_mark_hold", [])
        if m.get("remove"):
            self._mark_hold = [(t, x) for t, x in hold if not (x.get("mode", "TNG44") == mode and x["kind"] == m["kind"] and
                                                              ((m.get("id") and x.get("id") == m.get("id")) or
                                                               (not m.get("id") and x["abs"] == m["abs"])))]
            self.waterfall.add_marker(m, self._in_fs or 48000.0)
            return
        recent = time.time() - self.__dict__.get("_mark_ok", {}).get(mode, -1e9) < 5.0     # 막 끝난 메시지의 B
        if not (self._crc_shown(mode) or recent):
            if m.get("id"):
                hold[:] = [(t, x) for t, x in hold if x.get("id") != m.get("id")]
            hold.append((time.time(), dict(m)))
            del hold[:-40]
            self.waterfall.add_marker(dict(m, tentative=True), self._in_fs or 48000.0)   # 바로 보임 (확정 전 = 자홍 점선)
            return                                        # 성상도 추적 · 동기 맵 스냅샷 · 주파수 계기는 CRC 통과 뒤 (_draw_marker)
        self._draw_marker(m)

    def _draw_marker(self, m):
        if not m.get("tentative") and m["kind"] == "A":
            self._marker_side(m)
            self._freq_hold = (m["df"], time.time(), "코스타스 A")      # 주파수 오차 계기: CRC 통과한 메시지의 A 만
            self.p_zoom.set_df(m["df"])
        self.waterfall.add_marker(m, self._in_fs or 48000.0)

    def _marker_side(self, m):
        """A 검출의 화면 밖 준비 (한 번만): 실시간 성상도 추적 · 검출 스냅샷 예약"""
        if m.get("_side"):
            return
        m["_side"] = True
        if self.live_an is not None:
            self.live_an.track(m["abs"], m["df"])
            # 검출 스냅샷은 복조가 끝난 뒤 계산한다 (검출 직후에 바로 계산하면 같은 순간 시작하는 복조와
            # CPU · GIL 을 다퉈 복조가 0.1초 늦어졌다 — 자가검사 U3 측정). 오디오는 링버퍼에 남아 있으니
            # 맵 내용은 검출 순간 그대로다. B 가 안 와서 복조가 없으면 4초 뒤에 계산.
            self._snap_pending = dict(m)
            QTimer.singleShot(4000, lambda a_=m["abs"]: self._flush_snapshot(a_))
        self.p_const.clear_live()

    def _log_rx(self, kind, msg):
        if not self.rxlog_path:
            return
        try:
            with open(self.rxlog_path, "a", encoding="utf-8") as fp:
                fp.write("{} [{}] {}\n".format(datetime.now().strftime("%H:%M:%S.%f")[:-3], kind, msg))
        except Exception:
            pass

    # ================================================================== 화면 갱신
    def _ui_tick(self):
        """초당 UI_FPS 회: 입력 → 워터폴, 스펙트럼, 미터, TX 표시"""
        self._drain_input()
        lv = self._in_level
        db = 20 * np.log10(lv) if lv > 1e-6 else None
        self.m_level.set(db)
        self._in_level *= 0.9
        self._update_spectrum()
        self._update_live_snr()
        if self.docks["zoom"].isVisible():
            rows = self.zoom_an.take()
            if rows:
                self.p_zoom.set_info(self.zoom_an.info())
                self.p_zoom.push(rows)
        if self._transmitting:
            self._tx_tick()
        self._ov_tick()
        self._tng1_detect_tick()
        self._tng1_expert_tick()

    def _tx_spectrum(self):
        if not self._transmitting or self._tx_audio is None:
            return None
        a, fs = self._tx_audio
        n = int(fs * 0.128)
        i = self._tx["player"].pos if self._tx is not None else int((time.time() - self._tx_start) * fs)
        a = self._tx["player"].buf if self._tx is not None else a
        seg = a[max(0, i - n // 2):max(0, i - n // 2) + n]
        if len(seg) < n:
            return None
        w = np.hanning(n)
        pw = np.abs(np.fft.rfft(seg * w)) ** 2 / (fs * np.sum(w ** 2))
        db = 10 * np.log10(pw + 1e-16) + WaterfallWidget.psd_to_dbfs(n, fs)       # 칸당 dBFS
        return np.interp(self.waterfall.fgrid, np.fft.rfftfreq(n, 1.0 / fs), db)

    def _update_spectrum(self):
        wf = self.waterfall
        rx = None
        if wf.live and wf.recent:
            rx = np.mean(np.stack(wf.recent[-3:]), axis=0)
        elif not wf.live and wf.file_spectrum is not None:
            rx = wf.file_spectrum
        if rx is not None:
            rx = rx + wf.dbfs_offset()                      # 워터폴 PSD → 칸당 dBFS (워터폴 그림은 그대로)
        self.spectrum.update_curves(wf.fgrid, rx, self._tx_spectrum())

    def _update_live_snr(self):
        now = time.time()
        v, t0 = self._snr_hold
        if v is not None and now - t0 < 30:
            self.m_snr.set(v)
        else:
            wf = self.waterfall
            est = None
            if wf.live and len(wf.recent) >= 10:
                P = np.mean(10 ** (np.stack(wf.recent) / 10.0), axis=0)
                f = wf.fgrid
                fv = self._freq_hold[0] if (self._freq_hold[0] is not None and now - self._freq_hold[1] < 10) else 0.0
                c = CENTER + fv
                nb = ((f > 400) & (f < c - 400)) | ((f > c + 400) & (f < 2600))
                sb = (f >= c - 250) & (f <= c + 250)
                n0 = np.median(P[nb])
                ps = np.sum(P[sb]) * (f[1] - f[0]) - n0 * 500
                est = 10 * np.log10(ps / (n0 * 2500)) if (n0 > 0 and ps > 0) else None
            self.m_snr.set(est if (est is not None and est > -15) else None)
        fv, ft, src = self._freq_hold
        live = fv is not None and now - ft < 60
        det = "TNG1" if self.tx_mode == "TNG1" else "sync"         # 단일 모드: TNG1 은 감지 기준
        self.m_freq.set(fv if live else None,
                        tr("meter.{}_found".format(det), t=fmt_clock(now - ft)) if live else tr("meter.{}_wait".format(det)))

    def _sec_tick(self):
        now = datetime.now(timezone.utc)
        self.lb_utc.setText(now.strftime("%H:%M:%S UTC"))
        self.lb_local.setText(datetime.now().strftime("%H:%M:%S") + tr("main.local_suffix"))
        td = self.settings["time_display"]
        self.lb_utc.setVisible(td in ("utc", "both"))
        self.lb_local.setVisible(td in ("local", "both"))

        def ind(ok):
            col = T.c("ok") if ok else (T.c("err") if ok is False else T.c("text3"))
            return '<span style="color:{}">●</span>'.format(col)

        self.lb_audio.setText(tr("main.in_nbsp_nbsp_nbsp").format(
            ind(self._audio_ok["in"]), short_device(self.cb_in.currentText(), 28),
            "{:.0f}k".format(self._in_fs / 1000) if self._in_fs else "",
            ind(self._audio_ok["out"]), short_device(self.cb_out.currentText(), 28)))
        self.lb_audio.setToolTip(tr("main.input_output").format(self.cb_in.currentText(), self.cb_out.currentText()))
        for t in self.titles.values():
            if t.kind == "packet":
                t.refresh()
        st = self.session.stats()
        self.s_grid.set("rx", str(st["rx"]))
        self.s_grid.set("rate", "-" if st["rate"] is None else "{:.0%}".format(st["rate"]),
                        None if st["rate"] is None else ("ok" if st["rate"] > 0.9 else ("fec" if st["rate"] > 0.6 else "err")))
        self.s_grid.set("snr", "-" if st["snr_avg"] is None else "{:+.1f} dB".format(st["snr_avg"]))
        self.s_grid.set("tx", str(st["tx"]))

    # ================================================================== WAV
    def _ask_save(self):
        if not self._need_text():
            return
        path, _ = QFileDialog.getSaveFileName(self, tr("main.save_as_wav"), "neuromod_tx.wav", "WAV (*.wav)")
        if path:
            self.save_wav_path(path)

    def save_wav_path(self, path):
        try:
            dur = self.engine.save_wav(self.tx_edit.text(), path)
            self.bridge.saved.emit(path, dur)
            return True
        except Exception as ex:
            self.show_status(tr("main.wav_save_failed").format(ex), True)
            return False

    def _ask_open(self):
        path, _ = QFileDialog.getOpenFileName(self, tr("main.open_wav"), "", "WAV (*.wav)")
        if path:
            self.open_wav_path(path)

    def open_wav_path(self, path):
        from engine import read_any_wav
        try:
            x, fs = read_any_wav(path)
        except Exception as ex:
            self.show_status(tr("main.cannot_read_wav").format(ex), True)
            return False
        self.chk_live.blockSignals(True)
        self.chk_live.setChecked(False)
        self.chk_live.blockSignals(False)
        self.waterfall.show_audio(x, fs)
        self.show_status(tr("main.decoding_0f_hz").format(os.path.basename(path), fmt_clock(len(x) / fs), fs))
        self.btn_open.setEnabled(False)
        threading.Thread(target=self._decode_worker, args=(x, fs, os.path.basename(path)), daemon=True).start()
        return True

    def _decode_worker(self, x, fs, label):
        """파일 복조: 모드 자동 판별. TNG5 동기(A 또는 B)가 있으면 TNG5 복조, TNG44 는 복원됐거나 TNG5 가 없을 때 표시"""
        from engine import DecodeResult
        r5 = None
        if self.engine5 is not None:
            try:
                r5 = self.engine5.decode_file(x, fs)          # v2, 안 되면 v1 파일
                if (r5.info or {}).get("format") != "TNG5" or "A 미검출" in (r5.mode or ""):
                    r5 = None
            except Exception as ex:
                r5 = DecodeResult(False, reason="TNG5 {}: {}".format(type(ex).__name__, ex), info={"format": "TNG5"})
        r1s = []
        if "TNG1" in registry.ENABLED:
            try:
                from tng1_app import decode_file as decode1
                r1s = decode1(x, fs)                           # TNG1 1.1 (rev 2), 안 되면 1.0 (rev 1)
            except Exception as ex:
                r1s = [(DecodeResult(False, reason="TNG1 {}: {}".format(type(ex).__name__, ex), info={"format": "TNG1"}), "TNG1")]
        try:
            r = self.engine.decode(x, fs)
        except Exception as ex:
            r = DecodeResult(False, reason="{}: {}".format(type(ex).__name__, ex))
        if (r5 is None and not r1s) or r.ok or (r.info or {}).get("segments_ok"):
            self.bridge.decoded.emit(r, label + " · TNG44")
        ok1 = any(r1.ok or (r1.info or {}).get("segments_ok") for r1, _ in r1s)
        if r5 is not None and not (ok1 and not r5.ok and not (r5.info or {}).get("segments_ok")):
            self.bridge.decoded.emit(r5, label + " · TNG5")          # TNG1 이 풀린 파일의 TNG5 실패 (CRC 없음) 는 표시 안 함
        for r1, lab1 in r1s:
            self.bridge.decoded.emit(r1, label + " · " + lab1)

    def _ask_csv(self):
        path, _ = QFileDialog.getSaveFileName(self, tr("main.export_csv"), "neuromod_session.csv", "CSV (*.csv)")
        if path:
            n = self.session.export_csv(path)
            self.show_status(tr("main.csv_export_rows").format(n, path))

    # ================================================================== 수신 결과
    # ------------------------------------------------------------------ 수신 결과 분류 (수신 로그 · 수신 패널 공통)
    @staticmethod
    def classify(r, label=""):
        """
        복조 결과 → (결과 문구, 색, 표시 텍스트, 구간 문자열). 부분 복원은 실패로 치지 않는다.
        끝낸 방법: 코스타스 B (기본) · 길이 필드 (label) · 신호 소실 (label) · 송신 중지 (길이 > B 로 정한 길이)
        """
        info = r.info or {}
        stream = info.get("format") in ("스트리밍", "TNG5", "TNG1")
        ok_n, used = info.get("segments_ok"), info.get("segments_used")
        decl = info.get("declared_segments")
        seg = ("{}/{}".format(ok_n, used) if stream and used else ("1/1" if r.ok else ("0/1" if info.get("llr") is not None else "")))
        if "수동 중지" in (label or ""):
            m = decl or used
            return "수동 중지 ({}/{} 구간)".format(ok_n or 0, m or 0), "ok" if r.ok else ("fec" if ok_n else "err"), \
                r.text if r.ok else (info.get("partial") or "({})".format(r.reason or "복원 없음")), \
                "{}/{}".format(ok_n or 0, m or 0)
        if "신호 소실" in (label or "") and not r.ok:          # 신호 소실 > 송신 중지 (B 없이는 중지 판정 불가)
            from stream_ui import collapse_tail
            m = decl or used
            body, n_tail = collapse_tail(info.get("partial") or "")
            txt = (body + "□ (이후 {}구간 수신 못 함)".format(n_tail)) if n_tail > 1 else (info.get("partial") or "")
            return "신호 소실 ({}/{} 구간 복원)".format(ok_n or 0, m), "fec" if ok_n else "err", \
                txt or "({})".format(r.reason or "복원 없음"), "{}/{}".format(ok_n or 0, m)
        if stream and info.get("stopped"):
            return "송신 중지 ({}/{} 구간)".format(ok_n, decl), ("ok" if ok_n == used else "fec"), \
                info.get("partial") or "", "{}/{}".format(ok_n, decl)
        if info.get("format") == "TNG1":
            nc = info.get("comb") or 1
            tag = " · {}회 결합".format(nc) if nc > 1 else ""
            if r.ok:
                return "완전 복원" + tag, "ok", r.text, seg
            if info.get("head_missing"):                     # 첫 블록 표시 (wire rev 2) 없는 블록부터 받음
                return "앞부분 미수신 ({}/{} 블록){}".format(ok_n, used, tag), "fec", info.get("partial") or "", seg
            if ok_n and ok_n == used:
                return "완전 복원 (끝 표식 없음 · {}){}".format("신호 소실" if "소실" in (label or "") else "중지", tag), "ok", \
                    info.get("partial") or "", seg
            return "부분 복원 {}/{}{}".format(ok_n, used, tag), "fec", info.get("partial") or "", seg
        if r.ok:
            return ("완전 복원 (길이 필드)" if "길이 필드" in (label or "") else "완전 복원"), "ok", r.text, seg
        if stream and ok_n and ok_n == used:
            return "완전 복원 (끝 글자 잘림)", "ok", info.get("partial") or "", seg
        if stream and ok_n:
            return "부분 복원 {}/{}".format(ok_n, used), "fec", info.get("partial") or "", seg
        mode = r.mode or ""
        why = r.reason or "복조 실패"
        if "B 없음" in mode:
            why = "B 누락 · " + why
        elif "미검출" in mode:
            why = "A 미검출 · " + why
        return "실패", "err", "({})".format(why), seg

    def _on_decoded(self, r, label):
        self.btn_open.setEnabled(self.engine is not None)
        if self.live_an is not None:
            self.live_an.untrack()
        self._flush_snapshot()                     # 복조가 끝났으니 미뤄 둔 검출 스냅샷을 이제 계산
        self._pid += 1
        pid = self._pid
        now = datetime.now(timezone.utc)
        snr = min(r.snr_db, 40.0) if np.isfinite(r.snr_db) else None
        df = r.df_hz if np.isfinite(r.df_hz) else None
        margin = (snr - FAIL_SNR_DB) if snr is not None else None
        if snr is not None:
            self._snr_hold = (snr, time.time())
        if df is not None:
            self._freq_hold = (df, time.time(), "복조")
        info = r.info or {}
        if info.get("late"):                               # TNG1 늦은 결합: 지나간 송신의 블록 → 결과 표 · 세션 로그만 (수신 패널 · 워터폴 그대로)
            now_ = datetime.now(timezone.utc)
            self.table.add({"id": None, "utc_hms": now_.strftime("%H:%M:%S"), "result": "TNG1 · 늦은 결합", "result_col": "fec",
                            "snr": None, "df": r.df_hz, "seg": "{}회 결합".format(info.get("comb")),
                            "text": (info.get("partial") or "").replace("\n", " ↵ "), "text_col": "text2",
                            "tip": "이미 지나간 송신의 블록이 뒤늦게 반복 합치기로 풀림 (CRC16 통과, 새 메시지로 열지 않음)"})
            self.session.add({"dir": "RX", "ok": False, "mode": r.mode, "mode_name": "TNG1", "text": "", "reason": "늦은 결합",
                              "src": label, "result": "늦은 결합", "format": "TNG1", "partial": info.get("partial")})
            return
        if info.get("format") == "TNG1" and "실시간" in (label or ""):
            self.table.remove_id(self.LIVE1_ID)            # 수신 중 행 → 최종 결과 행
        result, rcol, text, seg = self.classify(r, label)
        mode_name = info.get("format") if info.get("format") in ("TNG5", "TNG1") else "TNG44"
        if mode_name == "TNG5" and "길이 필드" in (label or ""):
            self._len5_pid = pid                           # 뒤에 오는 코스타스 B 확인 표시용
        if "역방향" in (label or "") and not r.ok and not (r.info or {}).get("segments_ok"):
            result, rcol = "B만 검출 (역방향 복원 없음)", "err"
        if "실시간" in (label or "") and getattr(self, "_qrx_wait", None) is not None and not r.ok and \
                not (r.info or {}).get("segments_ok"):
            result, rcol = "미확인 신호", "err"          # A 뒤 CRC16 통과 없음 (가짜 시작 가능) — 수신 패널에는 표시 안 함
        no_crc = "실시간" in (label or "") and not r.ok and not info.get("segments_ok")
        if no_crc:                                         # CRC 통과 없는 실시간 결과 (가짜 시작 · 교차 오검출 등): 화면에 안 냄, 세션 로그만
            self.session.add({"dir": "RX", "ok": False, "snr_db": snr, "df_hz": df, "mode": r.mode, "mode_name": mode_name,
                              "text": "", "reason": r.reason, "pid": pid, "src": label, "result": result + " (화면 표시 없음)",
                              "format": info.get("format"), "segments": seg})
            o_ = self._ov
            self._drop_tentative(mode_name)                # 확정 전 표지 · 구간 조용히 지움
            if o_ is not None and o_.get("live") and o_.get("mode", "TNG44") == mode_name and not self._crc_shown():
                self._ov_end()                             # 이 모드의 표시 전 (CRC 전) 진행 상태만 정리
                self._qrx_wait = None
                self.qrx.btn_stop.setEnabled(False)
                self.qrx.set_meta(state=tr("main.waiting"), snr=float("nan"), df=float("nan"), seg=tr("main.seg"))
            return
        self.m_margin.set(margin)
        self.table.add({"id": pid, "utc_hms": now.strftime("%H:%M:%S"), "result": "{} · {}".format(mode_name, result), "result_col": rcol,
                        "snr": snr, "df": df, "seg": seg, "drift": info.get("drift_ppm"),
                        "relock": info.get("relocks"), "text": text.replace("\n", " ↵ "),
                        "text_col": "text" if rcol != "err" else "err",
                        "tip": "{} · {} · {}\n{}".format(result, info.get("format") or "", r.mode, text)})
        self.session.add({"dir": "RX", "ok": r.ok, "snr_db": snr, "df_hz": df, "margin_db": margin,
                          "mode": r.mode, "mode_name": mode_name, "text": r.text if r.ok else "", "reason": r.reason, "pid": pid,
                          "src": label, "result": result, "format": info.get("format"),
                          "segments": seg, "drift_ppm": info.get("drift_ppm"), "relocks": info.get("relocks"),
                          "partial": info.get("partial") if not r.ok else None})
        # 수신 패널 · 구간 표시: 최종 판정으로 교체
        if "역방향" in (label or "") and not r.ok and not info.get("segments_ok"):
            pass                                          # B 만 잡혔고 복원도 없음 — 수신 패널에는 표시 안 함 (로그만)
        else:
            self._final_to_qso(r, result, rcol, snr, df, seg)
        if self.pkt_an is not None:
            self.pkt_an.submit({"kind": "rx", "result": r, "label": label, "id": pid,
                                "utc": now.timestamp(), "loopback": self.chk_loop.isChecked()})
        self.show_status("{}: {}".format(result, label), rcol == "err")

    def _stop_rx_panel(self):
        """수신 패널 '중지' (09-27 15부: 이름 바꿈 — 예전 같은 이름이 수신기 멈춤 _stop_rx 를 가려 장치 변경 때 옛 수신 스레드가 남음): 수신 스레드에 요청 → 받은 구간까지 복조하고 다시 코스타스 A 대기"""
        rx = getattr(self, "_rx", None)
        if rx is not None:
            rx.stop_req = True
        self.qrx.btn_stop.setEnabled(False)

    def _final_to_qso(self, r, result, rcol, snr, df, seg, standalone=False):
        info = r.info or {}
        self.qrx.btn_stop.setEnabled(False)
        m = (r.viz or {}).get("msg") or {}
        stream = info.get("format") in ("스트리밍", "TNG5", "TNG1")
        is5 = info.get("format") == "TNG5"
        is1 = info.get("format") == "TNG1"
        live = self.waterfall.live and self._ov is not None and self._ov.get("live") and not standalone
        if not live:                                     # WAV 열기: 새 메시지로 보여 준다
            self._qrx_wait = None
            self.qrx.new_message(df, mode="TNG5" if is5 else ("TNG1" if is1 else "TNG44"))
        elif getattr(self, "_qrx_wait", None) is not None:
            ok_any = r.ok or bool(info.get("segments_ok"))
            if not ok_any:                               # CRC 통과 없이 끝남 → 수신 로그에만 '미확인 신호'
                self._drop_tentative(info.get("format") if info.get("format") in ("TNG5", "TNG1") else "TNG44")
                self._ov_end()
                self._qrx_wait = None
                self.qrx.set_meta(state=tr("main.waiting"), snr=float("nan"), df=float("nan"), seg=tr("main.seg"))   # 미확인 신호는 로그만
                return
            self._qrx_show(1)
        if stream and info.get("segments"):
            parts = self._parts(info["segments"], final=True, version=info.get("version", 1),
                                mode="TNG5" if is5 else ("TNG1" if is1 else "TNG44"))
            if r.ok:
                parts = [(r.text, "ok")]
        elif r.ok:
            parts = [(r.text, "ok")]
        else:
            parts = [("({})".format(r.reason or "복조 실패"), "note")]
        if result.startswith("신호 소실") and parts and parts[-1][1] == "fail" and len(parts[-1][0]) > 1:
            n_tail = len(parts[-1][0])                         # 끝의 연속 실패 구간 = 끊긴 뒤 못 받은 부분
            parts = parts[:-1] + [("□", "fail"), (" 신호 소실 · 못 받은 구간 {}".format(n_tail), "note")]
        self.qrx.render(parts)
        tag = next((k for k in ("부분 복원", "신호 소실", "송신 중지", "수동 중지") if k in result), None)
        if tag:
            self.qrx.set_tag(tag)
        self.qrx.set_meta(state=tr("main.rx_done").format(result), snr=snr if snr is not None else float("nan"),
                          df=df if df is not None else float("nan"),
                          seg=tr("main.seg2").format(seg) if seg else tr("main.seg"), err=rcol == "err")
        # 구간 표시
        a = m.get("a")
        if a is None:
            return
        if is1:                                          # TNG1: 워터폴 블록 표시는 실시간 사건 (start · segments) 으로 이미 그림
            if live:
                self._ov["ended"] = True
                self._ov["final"] = result
                self._ov_tick(force=True)
            return
        n_data = info.get("n_data") or (info["llr"].shape[0] if info.get("llr") is not None else None)
        if live:
            o = self._ov
        elif self.waterfall.live:                        # 실시간인데 A 사건을 못 받은 경우: 위치를 모르니 표시 생략
            return
        else:                                            # 파일: 파일 샘플 번호 기준으로 새로 만든다
            fs = self.waterfall._file[0] if self.waterfall._file else 8000.0
            k = fs / self.engine.cfg.fs_base
            b = m.get("b")
            a0 = (a["start"] + (600 if is5 else 0)) * k                 # TNG5: 송신 시작 → 코스타스 A 시작 (톤 300 ms)
            o = {"a": a0, "fs": fs, "df": m.get("df", 0.0), "states": {},
                 "b": b["start"] * k if b else None, "live": False, "plan": None, "fmt": None,
                 "mode": "TNG5" if is5 else "TNG44"}
            self._ov = o
        o["fmt"] = "stream" if stream else "block"
        o["ended"] = True                                # 결과가 나오면 표시 확정 (더 늘어나지 않음)
        o["final"] = result
        o["final_ok"] = r.ok
        if is5 and n_data:
            from tng5_app import Plan5
            o["plan"] = Plan5(n_data, len(info.get("segments") or []) or None)
            o["states"] = {x["index"]: ("ok" if x["ok"] else "fail") for x in info.get("segments") or []}
            o["b_wait"] = bool(info.get("b_wait"))            # 길이 필드로 먼저 복원: B 는 인정되면 그린다
        elif stream and n_data:
            from stream_ui import StreamPlan
            o["plan"] = StreamPlan(n_data)
            o["states"] = {x["index"]: ("ok" if x["ok"] else "fail") for x in info.get("segments") or []}
        if not live:
            self.waterfall.set_overlay(self._ov_items(costas=True))
        else:
            self._ov_tick(force=True)                    # 확정된 구간 (계획 · 상태) 로 한 번 다시 그림

    # ------------------------------------------------------------------ 실시간 구간 (코스타스 B 전)
    def _parts(self, segs, final, version=None, mode=None):
        """구간 결과 → 수신 패널 조각 [(글, 종류)]. version: 2 = 첫 구간 앞 2바이트가 길이 필드"""
        if version is None:
            version = getattr(self, "_live_ver", 2)
        mode = mode or self._live_mode
        segs = sorted(segs, key=lambda x: x["index"])
        okmap = {x["index"]: x["ok"] for x in segs}
        if mode == "TNG1":                              # 블록마다 글자 (실패 블록 = □), 결합 블록도 같은 초록
            text, owner = [], []
            best = {}
            for x in segs:                                # 같은 번호가 두 번 오면 (앞부분 □ 뒤에 늦게 풀린 블록) 통과한 쪽
                if x["index"] not in best or (x["ok"] and not best[x["index"]]["ok"]):
                    best[x["index"]] = x
            segs = [best[k] for k in sorted(best)]
            okmap = {x["index"]: x["ok"] for x in segs}
            for x in segs:
                t_ = x.get("text") or ("" if x["ok"] else "□")
                text += list(t_)
                owner += [x["index"]] * len(t_)
            text = "".join(text)
        elif mode == "TNG5":
            from tng5_app import assemble
            text, owner = assemble(segs)
        else:
            from stream_ui import assemble_text
            text, owner = assemble_text(segs, version=version)
        parts, cur, kind = [], [], None
        for ch, o in zip(text, owner):
            k = "ok" if okmap.get(o) else "fail"
            if k != kind and cur:
                parts.append(("".join(cur), kind))
                cur = []
            kind = k
            cur.append(ch)
        if cur:
            parts.append(("".join(cur), kind))
        if not final:
            parts.append(("▍", "pending"))
        return parts

    def _qrx_show(self, n_ok):
        """아직 표시 전인 실시간 메시지: CRC16 통과 구간이 생기면 그때 구분선(A 검출 시각) 을 연다. 표시 중이면 True"""
        w = getattr(self, "_qrx_wait", None)
        if w is None:
            return True
        if n_ok <= 0:
            return False
        self._qrx_wait = None
        if self._ov is not None:
            self._ov["shown"] = True                      # 워터폴 구간 표시: 확정 전 (자홍 점선) → 상태 색 (_ov_tick)
            self._ov["pre"] = False                       # TNG5 길이 필드로 한꺼번에 끝난 경우도 확정
        self.qrx.new_message(w["df"], stamp=w["stamp"], mode=w.get("mode"))
        self._mark_confirmed(w.get("mode") or "TNG44")
        return True

    DET1_ID = "tng1-detect"
    DET1_HOLD_S = 20.0

    def _tng1_detect_tick(self):
        """TNG1 감지 표시 (10부, 표시 전용): CRC 전이라도 신호가 들어오는 자리를 워터폴에 점선 'TNG1 감지' 로.
        판단은 수신기 동기 엿보기의 부분 동기 z (Tng1Receiver.detect) 그대로. 첫 CRC 통과 (확정 상자) 또는 DET1_HOLD_S 동안 새 감지 없으면 조용히 지움"""
        rx1 = getattr(self._rx, "rx1", None) if self._rx is not None else None
        cur = self.__dict__.get("_det1")
        d = rx1.detect_snapshot() if (rx1 is not None and rx1.enabled) else None      # 단일 모드: TNG1 모드일 때만
        o = self._ov
        confirmed = o is not None and o.get("mode") == "TNG1" and o.get("shown") and not o.get("ended")
        fresh = d is not None and rx1 is not None and rx1.stream.end - d["at"] < self.DET1_HOLD_S * 1000 and not confirmed             and (o is None or not o.get("shown") or o.get("ended") or o.get("mode") != "TNG1" or d["abs_in"] > o.get("a", -1e18) + 0.5 * (o.get("block_in") or 0))
        if not fresh or not self.waterfall.live:
            if cur is not None:
                self.waterfall.add_marker({"kind": "감지", "mode": "TNG1", "abs": cur["abs_in"], "df": cur["f"], "id": self.DET1_ID,
                                           "remove": True}, self._in_fs or 48000.0)
                self._det1 = None
            return
        if cur is not None and cur["abs_in"] == d["abs_in"] and cur["f"] == d["f"]:
            return
        fs = self._in_fs or 48000.0
        if cur is not None:
            self.waterfall.add_marker({"kind": "감지", "mode": "TNG1", "abs": cur["abs_in"], "df": cur["f"], "id": self.DET1_ID,
                                       "remove": True}, fs)
        # 표지는 감지한 순간 자리 (블록 시작은 대개 워터폴 기록 8 s 밖), 블록 칸은 _det1_items 가 블록 시작부터
        self.waterfall.add_marker({"kind": "감지", "mode": "TNG1", "abs": int(max(d["abs_in"], self._in_count - fs)), "df": d["f"],
                                   "id": self.DET1_ID, "tentative": True, "label": "TNG1 감지", "dur": 1.0, "half": 80}, fs)
        self._det1 = dict(d)
        self._log_rx("감지", "TNG1 감지 {} · {:+.1f} Hz (표시만)".format(self._det1_how(d), d["f"]))

    LIVE1_ID = "live-tng1"

    def _tng1_live_view(self):
        """TNG1 수신 중: 블록 CRC 통과마다 패킷 타임라인 (프레임 · 구간 줄) 과 수신 로그 '수신 중' 행을 바로 갱신.
        예전에는 둘 다 메시지가 끝난 뒤 패킷 분석 결과로만 채워져, 수신 패널이 탭 뒤에 있는 배치 ('전체') 에서는
        블록이 통과해도 화면에 아무 변화가 없어 보였다 (09-26 실제 앱 · 사용자 레이아웃)"""
        o = self._ov
        if o is None or o.get("mode") != "TNG1" or not o.get("shown"):
            return
        st, info = o.get("states") or {}, o.get("binfo") or {}
        if not st:
            return
        fr = 1000.0 * 14.72
        k0 = min(0, min(st))
        k1 = max(st) + (0 if o.get("ended") else 1)                # 다음 블록 = 수신 중 (대기)
        frames, segs = [], []
        for k in range(k0, k1 + 1):
            s_ = st.get(k, "pending" if k > max(st) else "fail")
            n = info.get(k, 1)
            frames.append(("F{}".format(k - k0 + 1), (k - k0) * fr, fr, "fixed" if (s_ == "ok" and n > 1) else s_,
                           "{}×".format(n) if n > 1 else ""))
            segs.append(("S{}".format(k - k0 + 1), (k - k0) * fr, fr, s_, str(k - k0 + 1)))
        self.timeline.show_packet({"frames": frames, "segments": segs, "seg_frames": {i: [i] for i in range(len(segs))}})
        self.titles["timeline"].touch()
        n_ok = sum(1 for v in st.values() if v == "ok")
        txt = "".join(t for t, _ in self._parts(self._live_segs, final=False, mode="TNG1")).replace("\n", " ↵ ").rstrip("▍")
        self.table.remove_id(self.LIVE1_ID)
        self.table.add({"id": self.LIVE1_ID, "utc_hms": o.get("utc_hms", ""), "result": "TNG1 · 수신 중", "result_col": "text2",
                        "snr": None, "df": o.get("df"), "seg": "{} 블록 통과".format(n_ok), "text": txt, "text_col": "text",
                        "tip": "수신 중 (블록 CRC16 통과마다 갱신, 끝나면 최종 결과로 바뀜)"})

    def _tng5_open(self):
        """TNG5 실시간 메시지: 첫 CRC 통과 → 수신 패널 머리줄 · 워터폴 구간 · 동기 표지 표시 시작"""
        self._ov["pre"] = False
        self._mark_confirmed("TNG5")
        self.qrx.set_meta(state=tr("main.rx_tng5_confirm_first"), snr=float("nan"), df=self._ov["df"], seg=tr("main.seg_0_confirmed"))
        self.qrx.btn_stop.setEnabled(True)

    def _on_live(self, ev):
        t = ev.get("type")
        mode = ev.get("mode", "TNG44")
        if t != "start" and mode != self._live_mode and t not in ("b_back", "b_only", "b_confirm"):
            return                                      # 다른 모드의 진행 중 메시지 사건 (수신 패널은 한 메시지씩)
        if t == "start":
            o_old = self._ov
            if o_old is not None and o_old.get("live") and (o_old.get("pre") or not o_old.get("shown")):
                self._drop_tentative(o_old.get("mode", "TNG44"))  # 확인 못 한 앞 시작 (새 표지는 아래에서 다시 보류)
            self._live_mode = mode
            self._zoom_tones(mode)
            self._live_segs = []
            self._ov = {"a": ev["abs"], "fs": ev["fs"], "df": ev["df"], "states": {}, "b": None, "live": True,
                        "plan": None, "fmt": None, "mode": mode}
            # 구분선 · 글자는 첫 CRC16 통과 때 (_qrx_show). 가짜 시작(A 오검출)은 수신 패널에 남지 않는다
            self._qrx_wait = {"stamp": datetime.now(), "df": ev["df"], "mode": mode}
            if mode == "TNG1":                          # TNG1: 'start' 는 첫 블록 CRC 통과 (+ 표시 규칙) 뒤에만 온다
                self._ov["block_in"] = ev.get("block_in")
                self._ov["utc_hms"] = datetime.now(timezone.utc).strftime("%H:%M:%S")
                self._ov["binfo"] = {}
                self.qrx.set_meta(state=tr("main.rx_tng1_14_7"), snr=float("nan"), df=ev["df"], seg=tr("main.block_0"))
                self.qrx.btn_stop.setEnabled(True)
            elif mode == "TNG5":                          # TNG5: 확정 (첫 CRC · M1) 전에는 화면 표시 없음 → 'confirm' 사건 때
                self._ov["pre"] = True
            else:                                        # TNG44: 첫 CRC 통과 전에는 머리줄도 그대로 (검출 문구 없음)
                self.qrx.btn_stop.setEnabled(True)
        elif t in ("b_confirm", "b_timeout") and mode == "TNG5":   # 길이 필드로 끝낸 메시지의 가드 · B: 인정 / 미검출
            pid_ = getattr(self, "_len5_pid", None)
            if pid_ is not None:
                st_ = "ok" if t == "b_confirm" else "missing"
                self.table.add_result_note(pid_, tr("main.b_confirmed") if st_ == "ok" else tr("main.b_not_found"))
                self._b5 = getattr(self, "_b5", {})
                self._b5[pid_] = (st_, float(ev.get("off_s") or 0.0))
                self._apply_b5(pid_)
                o = self._ov
                if o is not None and o.get("mode") == "TNG5" and o.get("b_wait"):
                    o["b_wait"] = False
                    o["b_missing"] = st_ == "missing"
                    if st_ == "ok" and ev.get("abs") is not None:
                        o["b"] = ev["abs"]
                    o["_drawn_end"] = False
                    self._ov_tick(force=True)
        elif t in ("confirm", "m_fail") and mode == "TNG5":
            if self._ov is not None and t == "confirm" and self._ov.get("pre") and ev.get("by") == "crc":
                self._tng5_open()                        # 중간 동기 M1 확정은 CRC 가 아니라 로그만 (09-26: CRC 전 표시 금지)
            if self._ov is not None:
                if t == "m_fail":
                    self._ov["m1"], self._ov["m1_info"] = "fail", {"rho": ev.get("rho", 0.0), "need": ev.get("need", 0.0)}
                elif ev.get("by") == "mid":
                    self._ov["m1"], self._ov["m1_info"] = "ok", {"rho": ev.get("rho", 0.0), "need": ev.get("need", 0.0)}
                self._ov["confirm_by"] = ev.get("by") or self._ov.get("confirm_by")
        elif t == "format" and mode == "TNG5":
            if not (self._ov or {}).get("pre"):
                self.qrx.set_meta(state=tr("main.rx_tng5_length_field"))
        elif t == "format":
            if self._ov is not None:
                self._ov["fmt"] = ev["format"]
            self._live_ver = 2 if ev["format"] == "stream2" else 1
            if ev["format"] == "block":
                self.qrx.set_meta(state=tr("main.rx_single_block_format"))
            else:
                self.qrx.set_meta(state=tr("main.rx_streaming_format").format(" (길이 필드)" if self._live_ver == 2 else ""))
        elif t == "segments":
            self._live_segs += ev["segs"]
            if mode == "TNG5" and (self._ov or {}).get("pre") and any(x["ok"] for x in ev["segs"]):
                self._tng5_open()                        # M1 로 먼저 확정돼 'confirm (crc)' 사건이 없는 경우도 첫 CRC 통과에서 연다
            if self._ov is not None:
                for x in ev["segs"]:
                    self._ov["states"][x["index"]] = "ok" if x["ok"] else "fail"
                    if x.get("n_comb") and "binfo" in self._ov:
                        self._ov["binfo"][x["index"]] = x["n_comb"]
            n_ok = sum(1 for x in self._live_segs if x["ok"])
            if self._qrx_show(n_ok):
                self.qrx.render(self._parts(self._live_segs, final=False))
            if mode == "TNG1":
                self._tng1_live_view()
                nc = sum(1 for x in self._live_segs if (x.get("n_comb") or 1) > 1)
                self.qrx.set_meta(state=tr("main.rx_tng1_14_7"),
                                  seg=tr("main.block_ok").format(n_ok, " (결합 {})".format(nc) if nc else ""))
            elif not (self._ov or {}).get("pre"):
                self.qrx.set_meta(seg=tr("main.seg_confirmed_ok_fail").format(
                len(self._live_segs), n_ok, len(self._live_segs) - n_ok))
        elif t == "final_segments":                     # 길이 필드로 끝을 알고 정확한 길이로 다시 확정한 구간 전체
            self._ov_end()
            self._live_segs = list(ev["segs"])
            if self._ov is not None:
                self._ov["states"] = {x["index"]: ("ok" if x["ok"] else "fail") for x in ev["segs"]}
            n_ok = sum(1 for x in self._live_segs if x["ok"])
            if self._qrx_show(n_ok):
                self.qrx.render(self._parts(self._live_segs, final=True))
            self.qrx.set_meta(state=tr("main.decoding"),
                              seg=tr("main.seg_ok").format(n_ok, len(self._live_segs)))
        elif t == "cancel":                              # TNG5 미확정 폐기: 되돌림
            self._drop_tentative(mode)
            self._ov = None
            self._qrx_wait = None
            self._live_segs = []
            self.qrx.set_meta(state=tr("main.waiting"), snr=float("nan"), df=float("nan"), seg=tr("main.seg"))
            self.qrx.btn_stop.setEnabled(False)
        elif t == "end" and mode == "TNG1":
            self._ov_end()
        elif t == "lost":
            shown_ = self._crc_shown(mode)
            if not shown_:
                self._drop_tentative(mode)               # CRC 통과 전 소실: 확정 전 표시 조용히 지움
            self._ov_end()
            if shown_:                                   # CRC 통과 전 신호의 소실은 로그만
                self.qrx.set_meta(state=tr("main.signal_lost"), err=True)
            self.qrx.btn_stop.setEnabled(False)
        elif t == "manual_stop":
            self._ov_end()
            self.qrx.set_meta(state=tr("main.rx_stopped"))
            self.qrx.btn_stop.setEnabled(False)
        elif t == "fake":                               # 코스타스 A 오검출: 수신 패널에서 지우고 로그에만
            self._drop_tentative(mode)
            self.qrx.discard_current()
            self.qrx.set_meta(state=tr("main.waiting"), snr=float("nan"), df=float("nan"), seg=tr("main.seg"))   # 오검출 폐기는 로그만
            self.qrx.btn_stop.setEnabled(False)
            self._ov = None
            self._live_segs = []
            self._pid += 1
            note = "A 뒤 {:.1f} s · 가드 {:.2f} · |LLR| {:.2f} · ρ² {:.3f}".format(
                ev.get("t_s") or 0, ev.get("guard_q") or 0, ev.get("llr") or 0, ev.get("rho") or 0)
            self.table.add({"id": self._pid, "utc_hms": datetime.now(timezone.utc).strftime("%H:%M:%S"),
                            "result": "오검출 폐기", "result_col": "err", "df": ev.get("df"),
                            "text": "코스타스 A 오검출 · {}".format(note), "text_col": "err"})
            self.session.add({"dir": "RX", "ok": False, "result": "오검출 폐기", "reason": note})
        elif t == "b":
            self.qrx.btn_stop.setEnabled(False)
            if self._ov is not None:
                self._ov["b"] = ev["abs"]
            self._ov_end()
            self.qrx.set_meta(state=tr("main.decoding"))
        elif t == "b_only":                             # 첫 CRC 전 단계 → 화면 표시 없음, 세션 로그만
            self._drop_tentative(mode)
            self.session.add({"dir": "RX", "ok": False, "result": "B만 검출 (표시 안 함)", "mode": mode,
                              "reason": ev.get("reason") or "A 없음", "trace": ev.get("trace")})

    # ------------------------------------------------------------------ 워터폴 구간 표시
    def _ov_items5(self, costas):
        """TNG5 v2 워터폴 구간: 톤 · Welch12 · 가드 · 18바이트 구간 (20프레임마다 중간 동기 M) · 가드 · B×1"""
        import tng5
        from tng5_app import Plan5, N_SYNC, N_SYNC_B, NSEG_MAX, frames_for_segs
        o = self._ov
        fs = o["fs"]
        k = fs / self.engine.cfg.fs_base
        L = self.engine.cfg.frame_len
        fc = CENTER + float(o.get("df") or 0.0)
        a = o["a"]
        an = N_SYNC * k
        band = (fc - 235, fc + 235)
        out = [{"abs0": a - 0.3 * fs, "abs1": a, "kind": "tone", "label": "톤", "tip": "희생 톤 300 ms (TNG5)",
                "f0": fc - 70, "f1": fc + 70}]
        if costas:
            out.append({"abs0": a, "abs1": a + an, "kind": "costas_a", "label": "TNG5 A",
                        "tip": "TNG5 시작 동기 Welch 12차 코스타스 (0.77 s, +6 dB) · 주파수 오차 {:+.1f} Hz".format(o.get("df") or 0.0),
                        "f0": fc - 125, "f1": fc + 125})
        g0 = a + an
        f0 = g0 + L * k
        out.append({"abs0": g0, "abs1": f0, "kind": "guard", "label": "G", "tip": "가드 프레임 192 ms",
                    "f0": band[0], "f1": band[1]})
        now, b = o.get("now"), o.get("b")
        end = (b - L * k) if b is not None else now
        plan = o.get("plan")
        if plan is None:
            if getattr(self, "_plan5_max", None) is None:
                self._plan5_max = Plan5(frames_for_segs(NSEG_MAX), NSEG_MAX)
            plan = self._plan5_max
        for s_ in range(plan.n_seg):
            x0, x1 = plan.seg_nominal[s_]
            s0, s1 = f0 + float(tng5.frame_off(x0)) * k, f0 + float(tng5.frame_off(x1)) * k
            if o.get("plan") is None and (now is None or s0 > now or (end is not None and s0 > end)):
                break
            st = o["states"].get(s_, "fail" if o.get("ended") else "pending")      # 종료 뒤 못 받은 구간 = □
            fr = plan.seg_frames[s_]
            out.append({"abs0": s0, "abs1": min(s1, end) if (o.get("plan") is None and end is not None) else s1,
                        "kind": "seg", "state": st, "label": "S{}".format(s_ + 1),
                        "tip": "TNG5 구간 {}  ({}자)\n{}\n비트가 실린 프레임 F{}–F{} (인터리버 약 9.6 s)".format(
                            s_ + 1, 24 if s_ == 0 else 27,
                            {"ok": "CRC16 통과", "fail": "CRC16 실패", "pending": "대기 (미확정)"}[st],
                            int(fr.min()) + 1, int(fr.max()) + 1),
                        "f0": band[0], "f1": band[1]})
        # 중간 동기 M (20프레임 = 3.84 s 마다). 검출 표시가 따로 없으므로 일반 워터폴에도 그린다
        last = b if b is not None else now
        if o.get("plan") is not None:
            last = f0 + float(tng5.b_off(o["plan"].n) - L) * k if b is None else last
        m1 = o.get("m1")                                         # 첫 M 확인: None 대기 · "ok" · "fail"
        for kq in range(1, 80):
            m0 = f0 + float(tng5.mid_off(kq)) * k
            if last is None or m0 >= last - L * k:
                break
            st = {None: "확인 대기", "ok": "확인 통과 (수신 확정)", "fail": "확인 실패 (첫 CRC 로 확정)"}.get(m1, "")
            info_ = o.get("m1_info") or {}
            out.append({"abs0": m0, "abs1": m0 + tng5.NM * k, "kind": "costas_mx" if (kq == 1 and m1 == "fail") else "costas_m",
                        "label": "M{}".format(kq) + (" ✓" if kq == 1 and m1 == "ok" else " ✗" if kq == 1 and m1 == "fail" else ""),
                        "tip": "TNG5 중간 동기 M{}  0.34 s · 7차 코스타스 배열 {} (48 ms · 41.67 Hz)".format(kq, (kq - 1) % 4 + 1) +
                               ("\n첫 중간 동기: {}{}".format(st, " · ρ² {:.2f} (기준 {:.2f})".format(info_["rho"], info_["need"])
                                                               if "rho" in info_ else "") if kq == 1 else ""),
                        "f0": fc - 125, "f1": fc + 125})
        if b is not None:
            out.append({"abs0": b - L * k, "abs1": b, "kind": "guard", "label": "G", "tip": "가드 프레임 192 ms",
                        "f0": band[0], "f1": band[1]})
            if costas:
                out.append({"abs0": b, "abs1": b + N_SYNC_B * k, "kind": "costas_b", "label": "TNG5 B",
                            "tip": "TNG5 코스타스 B × 1 (0.34 s, 메시지 끝)", "f0": fc - 125, "f1": fc + 125})
        elif (o.get("b_wait") or o.get("b_missing")) and o.get("plan") is not None:
            # 길이 필드로 먼저 복원: 가드는 다른 구간처럼 그대로, B 는 인정되면 그 자리에 (없으면 예상 끝 + 2 s 뒤 'B 없음')
            be = f0 + float(tng5.b_off(o["plan"].n)) * k
            out.append({"abs0": be - L * k, "abs1": be, "kind": "guard", "label": "G", "tip": "가드 프레임 192 ms",
                        "f0": band[0], "f1": band[1]})
            if o.get("b_missing"):
                out.append({"abs0": be, "abs1": be + N_SYNC_B * k, "kind": "seg", "state": "fail", "label": "B 없음",
                            "tip": "TNG5 코스타스 B · 예상 끝 + 2 s 안에 미검출", "f0": fc - 125, "f1": fc + 125})
        return out

    def _ov_items(self, costas):
        o = self._ov
        if o is None:
            return []
        if o.get("mode") == "TNG5":
            return self._ov_items5(costas)
        if o.get("mode") == "TNG1":
            from tng1_app import blocks_overlay
            return blocks_overlay(o, CENTER)
        from stream_ui import StreamPlan
        fs = o["fs"]
        cfg = self.engine.cfg
        k = fs / cfg.fs_base
        L = cfg.frame_len
        fc = CENTER + float(o.get("df") or 0.0)
        a = o["a"]
        an = int(round(self.engine.pc.symbol_ms * 1e-3 * self.engine.pc.n_tones * cfg.fs_base)) * k
        band = (fc - 235, fc + 235)
        out = [{"abs0": a - self.engine.pc.tone_ms * 1e-3 * fs, "abs1": a, "kind": "tone", "label": "톤",
                "tip": "희생 톤 300 ms (송수신 전환 · AGC 안정)", "f0": fc - 70, "f1": fc + 70}]
        if costas:
            out.append({"abs0": a, "abs1": a + an, "kind": "costas_a", "label": "A",
                        "tip": "코스타스 A 224 ms (시작 동기 · 주파수 오차 {:+.1f} Hz)".format(o.get("df") or 0.0),
                        "f0": fc - 205, "f1": fc + 205})
        g0 = a + an
        f0 = g0 + L * k
        out.append({"abs0": g0, "abs1": f0, "kind": "guard", "label": "G", "tip": "가드 프레임 192 ms (형식 표지)",
                    "f0": band[0], "f1": band[1]})
        now = o.get("now")
        b = o.get("b")
        end = (b - L * k) if b is not None else now
        if o.get("fmt") == "block":
            if end is not None:
                st = "pending" if o.get("final") is None else ("ok" if o.get("final_ok") else "fail")
                out.append({"abs0": f0, "abs1": end, "kind": "block", "state": st, "label": "데이터 (단일 블록)",
                            "tip": "단일 블록 형식: 패킷 전체 CRC16 1개", "f0": band[0], "f1": band[1]})
        else:
            plan = o.get("plan")
            if plan is None:
                if self._plan_max is None:
                    n_max = int(self.engine.pc.max_message_s * cfg.fs_base / L) + 2
                    self._plan_max = StreamPlan(n_max)
                plan = self._plan_max
            for s_ in range(plan.n_seg):
                x0, x1 = plan.seg_nominal[s_]
                s0, s1 = f0 + x0 * L * k, f0 + x1 * L * k
                if o.get("plan") is None and (now is None or s0 > now or (end is not None and s0 > end)):
                    break
                st = o["states"].get(s_, "fail" if o.get("ended") else "pending")      # 종료 뒤 못 받은 구간 = □
                fr = plan.seg_frames[s_]
                out.append({"abs0": s0, "abs1": min(s1, end) if (o.get("plan") is None and end is not None) else s1,
                            "kind": "seg", "state": st, "label": "S{}".format(s_ + 1),
                            "tip": "구간 {}  바이트 {}–{}\n{}\n비트가 실린 프레임 F{}–F{} (인터리버 ±3.2 s)".format(
                                s_ + 1, 32 * s_, 32 * s_ + 31,
                                {"ok": "CRC16 통과", "fail": "CRC16 실패", "pending": "대기 (미확정)"}[st],
                                int(fr.min()) + 1, int(fr.max()) + 1),
                            "f0": band[0], "f1": band[1]})
        if b is not None:
            out.append({"abs0": b - L * k, "abs1": b, "kind": "guard", "label": "G", "tip": "가드 프레임 192 ms",
                        "f0": band[0], "f1": band[1]})
            if costas:
                out.append({"abs0": b, "abs1": b + an, "kind": "costas_b", "label": "B",
                            "tip": "코스타스 B 224 ms (메시지 끝 · 프레임 수 확정)", "f0": fc - 205, "f1": fc + 205})
        return out

    def _ov_end(self):
        """수신 종료 (B · 길이 필드 · 신호 소실 · 수동 중지 · CRC 없음): 워터폴 구간 표시를 이 시점에서 멈춘다.
        이후 못 받은 예상 구간은 실패 (□) 로 그린다 (수신 창 □ 처리와 같게). 새 A 는 새 표시, 가짜 시작 · 되돌림은 표시 삭제"""
        o = self._ov
        if o is not None and o.get("live") and not o.get("ended"):
            o["now"] = self._in_count
            o["ended"] = True
            self._ov_tick(force=True)

    def _ov_tick(self, force=False):
        """워터폴 · 확대 워터폴 구간 표시. 세 모드 모두 검출 (TNG44 · TNG5 A, TNG1 감지) 순간부터 그림:
        첫 CRC 전 = 확정 전 (tent, 자홍 점선), 첫 CRC 뒤 = 상태 색 (통과 · 실패 · 대기). CRC 없이 끝나면 _drop_tentative 가 지움"""
        o = self._ov
        det = self._det1_items()
        live_o = o is not None and o.get("live") and self.engine is not None
        if live_o:
            if o.get("ended"):
                if not force and o.get("_drawn_end") and not det and not self.__dict__.get("_det1_drawn"):
                    return                               # 끝난 표시는 다시 계산하지 않는다
                o["_drawn_end"] = True
            else:
                o["now"] = self._in_count
        elif not det and not self.__dict__.get("_det1_drawn"):
            return
        self._det1_drawn = bool(det)
        tent = live_o and (o.get("pre") or not o.get("shown"))

        def items(costas):
            out = self._ov_items(costas) if live_o else []
            return ([dict(it, tent=True) for it in out] if tent else out) + det
        if self.waterfall.live:
            self.waterfall.set_overlay(items(False))                       # 워터폴 A/B 는 기존 검출 표시 (add_marker)
        if self.docks["zoom"].isVisible():
            self.p_zoom.set_overlay(items(True))

    @staticmethod
    def _det1_how(d):
        """감지 근거 문구: 빠른 감지 (소리 모양 D, 15부) 또는 부분 동기 z (10부)"""
        if d.get("z") is None:
            return "소리 모양 D {:.2f}".format(d.get("fast") or 0.0)
        return "동기 z {:.1f} · 동기 칸 {}".format(d["z"], d.get("m"))

    def _det1_items(self):
        """TNG1 감지 (첫 CRC 전): 감지한 블록 칸 '수신 중' (확정 전, 받은 만큼). 확정된 TNG1 메시지 수신 중이면 없음"""
        d = self.__dict__.get("_det1")
        o = self._ov
        if d is None or (o is not None and o.get("mode") == "TNG1" and o.get("shown") and not o.get("ended")):
            return []
        rx1 = getattr(self._rx, "rx1", None) if self._rx is not None else None
        if rx1 is None or rx1.stream.end - d["at"] > 5000:        # 최근 5 s 안에 다시 감지된 것만 (끝난 신호를 다음 블록으로 늘이지 않게)
            return []
        fs = self._in_fs or 48000.0
        L = 14.72 * fs
        a0 = d["abs_in"] + max(0, int((self._in_count - d["abs_in"]) // L)) * L    # 지금 받는 블록 (감지 자리는 블록 주기 위상)
        a1 = max(a0 + 0.5 * fs, min(a0 + L, self._in_count))
        fc = CENTER + float(d["f"])
        return [{"abs0": a0, "abs1": a1, "kind": "seg", "state": "pending", "tent": True, "label": "TNG1 수신 중",
                 "tip": "TNG1 블록 수신 중 (감지 {}, 첫 CRC 전 · 확정 전 표시)\n블록 CRC16 통과 때 확정 칸으로 바뀌고, 확인 안 되면 조용히 지움".format(self._det1_how(d)),
                 "f0": fc - 80, "f1": fc + 80}]

    def _on_packet_analysis(self, out):
        kind = out.get("kind")
        if kind == "error":
            self.show_status(out["msg"], True)
            return
        if kind == "tx":
            self.p_const.set_tx(out.get("const"))
            self.p_eye.set_tx(out.get("eye"))
            return
        pid = out.get("id")
        self.packets[pid] = out
        for k in list(self.packets)[:-200]:
            del self.packets[k]
        if pid in getattr(self, "_b5", {}):                     # 분석보다 B 가 먼저 왔으면 여기서 반영
            self._apply_b5(pid)
        if out.get("kind") == "rx" and out.get("dissect") is not None:
            self.p_dissect.prepare(out["dissect"])            # 해부 창 열기 전에 오디오 채움 전체 보기 미리 그리기 (한 번 연 뒤부터)
            self._pre_apply_dissect(pid, out)
        if out.get("fec_fixed") is not None:
            self.table_set_fec(pid, out["fec_fixed"])
            self._patch_session(pid, out)
        fin = lambda v: v if (v is not None and np.isfinite(v)) else None
        self.trend.add({"t": out.get("utc") or time.time(), "snr": fin(out.get("snr")), "df": fin(out.get("df")),
                        "fec": out.get("fec_fixed"), "ber": out.get("ber"),
                        "ok": 1.0 if out["ok"] else (0.5 if out.get("segments_ok") else 0.0)})
        self.show_packet(pid)

    def _apply_b5(self, pid):
        """TNG5 길이 필드 결과 뒤 가드 · B 상태를 패킷 자료 (타임라인 · 해부) 에 반영하고, 보고 있는 화면이면 다시 그림"""
        st = getattr(self, "_b5", {}).get(pid)
        out = self.packets.get(pid)
        if st is None or out is None or out.get("b_state") == st[0]:
            return
        from tng5_app import timeline_b_state
        from dissect5 import b_state
        out["b_state"] = st[0]
        if out.get("timeline"):
            out["timeline"] = timeline_b_state(out["timeline"], st[0])
        if out.get("dissect") is not None:
            old = out["dissect"]
            out["dissect"] = b_state(old, st[0], st[1])
            if self.p_dissect.d is old and self.docks["dissect"].isVisible():
                self.p_dissect.show_packet(out["dissect"], self._dissect_title(out))
        if out.get("dissect5"):
            out["dissect5"]["has_b"] = st[0] == "ok"
        if self._shown_pid == pid:
            self.timeline.show_packet(out.get("timeline"), out.get("frame_err"), out.get("frame_llr"))
            self.titles["timeline"].touch()

    def _patch_session(self, pid, out):
        for r in reversed(self.session.rows):
            if r.get("pid") == pid:
                r["fec_fixed"], r["fec_total"], r["ber_hard"] = out.get("fec_fixed"), out.get("fec_total"), out.get("ber")
                break

    def table_set_fec(self, pid, v):
        self.table.set_fec(pid, v)

    def show_packet(self, pid):
        """패킷 단위 패널들을 이 패킷 기준으로"""
        out = self.packets.get(pid)
        if out is None:
            return
        self._shown_pid = pid
        if out.get("format") == "TNG1":
            self._set_expert_mode("TNG1")
            bl = out.get("tng1_blocks") or []
            self.p_dissect1.set_packet(bl, self._dissect1_title(out))
            if bl and bl[-1].get("E") is not None:
                from tng1_app import sync_scores
                self.p_sync1.set_data({"score": sync_scores(np.asarray(bl[-1]["E"])), "label": "B{} (선택 패킷)".format(bl[-1]["k"] + 1)})
        elif self._expert1:
            self._set_expert_mode(out.get("format") if out.get("format") == "TNG5" else "TNG44")
        self.p_const.set_packet(out)
        self.p_eye.set_packet(out)
        self.p_feat.set_packet(out)
        self.p_llr.set_packet(out.get("llr"))
        if out.get("sync5") is not None:                     # TNG5 v2: 시작 동기 맵 = 이 패킷 Welch12 둘레 (64 ms · 31.25 Hz) + 중간 동기 확인
            self.p_sync.set_snapshot(out["sync5"])
        self.timeline.show_packet(out.get("timeline"), out.get("frame_err"), out.get("frame_llr"))
        g = self.q_grid
        ber = out.get("ber")
        g.set("ber", "-" if ber is None else "{:.2%}".format(ber),
              None if ber is None else ("ok" if ber < 0.01 else ("fec" if ber < 0.05 else "err")))
        ff, ft = out.get("fec_fixed"), out.get("fec_total")
        g.set("fec", "-" if ff is None else "{} / {}".format(ff, ft), None if ff is None else ("fec" if ff else "ok"))
        s = out.get("snr")
        g.set("snr", "-" if s is None or not np.isfinite(s) else "{:+.1f} dB".format(min(s, 40)))
        g.set("evm", "-" if out.get("evm_eq") is None else "{:.1f} dB".format(out["evm_eq"]))
        g.set("rho", "-" if out.get("rho_a") is None else "{:.2f}".format(out["rho_a"]))
        g.set("drift", "-" if out.get("drift_hz") is None else "{:+.2f} Hz".format(out["drift_hz"]))
        for k in ("timeline", "quality", "trend", "const", "eye", "feat", "llr"):
            self.titles[k].touch()

    def _float_size(self, key):
        mw, mh = PANEL_MIN[key]
        return max(mw + 40, int(self.width() * (0.72 if key == "dissect" else 0.45))), \
            max(mh + 60, int(self.height() * (0.8 if key == "dissect" else 0.6)))

    def show_float(self, key):
        """프리셋 밖 패널을 떠 있는 창으로 연다 (이미 보이면 앞으로)"""
        d = self.docks[key]
        if not d.isVisible():
            if not d.isFloating():
                d.setFloating(True)
            w, h = self._float_size(key)
            d.resize(w, h)
            g = self.geometry()
            d.move(g.x() + (g.width() - w) // 2, g.y() + (g.height() - h) // 2)
            d.show()
        d.raise_()

    def _zoom_preset(self, name):
        self.zoom_an.set_preset(name)
        self.settings["zoom_preset"] = name

    @staticmethod
    def _dissect_title(out):
        utc = datetime.fromtimestamp(out["utc"], timezone.utc).strftime("%H:%M:%S") if out.get("utc") else ""
        return "{} UTC  {}".format(utc, (out.get("text") or out.get("partial") or out.get("reason") or "")[:60])

    def _pre_apply_dissect(self, pid, out):
        """새 패킷 결과를 숨은 해부 창에 미리 적용 (층 항목 · 트리 · 개요, 여러 틱으로 나눠서) → 열 때 바로"""
        d = self.docks["dissect"]
        if d.isVisible():
            return
        self.p_dissect.pre_apply(out["dissect"], self._dissect_title(out))    # 여러 이벤트 틱으로 나눠서
        self._dissect_pre = (pid, out["dissect"])

    def _select_packet(self, pid):
        if pid in self.packets:
            self.show_packet(pid)
            out = self.packets[pid]
            if out.get("kind") == "rx":
                pre = getattr(self, "_dissect_pre", None)
                if self.p_dissect.pending_d() is not None:
                    if pre is not None and pre[0] == pid and self.p_dissect.pending_d() is out.get("dissect"):
                        self.p_dissect.finish_pending()       # 나눠 하던 미리 적용을 즉시 마저 끝낸 뒤 표시
                    else:
                        self.p_dissect.cancel_pending()
                is5 = out.get("format") == "TNG5" and out.get("dissect") is None      # TNG5 는 8층 해부 (없을 때만 옛 표)
                self.p_dissect.setVisible(not is5)
                self.p_dissect5.setVisible(is5)
                if is5:
                    self.p_dissect5.show_packet(out.get("dissect5"), self._dissect_title(out))
                elif not (pre is not None and pre[0] == pid and pre[1] is out.get("dissect")
                          and self.p_dissect.d is pre[1]):               # 미리 적용한 패킷이 아니면 정상 경로
                    self.p_dissect.show_packet(out.get("dissect"), self._dissect_title(out))
                self._dissect_pre = None
                self.titles["dissect"].touch()
                self.show_float("dissect")

    def _select_frame(self, fi):
        for p in (self.p_const, self.p_eye, self.p_feat):
            p.set_frame(fi)

    def _flush_snapshot(self, only_abs=None):
        m = getattr(self, "_snap_pending", None)
        if m is None or (only_abs is not None and m["abs"] != only_abs) or self.live_an is None:
            return
        self._snap_pending = None
        self.live_an.snapshot(m)

    def _on_live_analysis(self, out):
        s = out.get("sync")
        if s is not None and self.docks["sync"].isVisible():
            self.p_sync.set_live(s)
        sn = out.get("snap")
        if sn is not None:
            if "error" in sn:
                self._log_rx("snap", sn["error"])
            else:
                self.p_sync.set_snapshot(sn)
                ck = sn["check"]
                self._log_rx("snap", "검출기 rho2 {:.3f} df {:+.2f} | 같은 위치 다시 잰 값 {:.3f} df {:+.2f} (시작 {:+d}) | "
                             "맵 검출칸 {:.3f} 맵 최대 {:.3f}".format(
                                 sn["det"]["rho"], sn["det"]["df"], ck["refine_rho"], ck["refine_df"],
                                 ck["refine_shift"], ck["map_at_det"], ck["map_max"]))
        pts = out.get("live_pts")
        if pts is not None:
            self.p_const.push_live(pts)

    # ================================================================== 송신
    def _need_text(self):
        if not self.tx_edit.text().strip():
            self.show_status(tr("main.message_is_empty"), True)
            return False
        if self.engine is None:
            self.show_status(tr("main.model_not_ready"), True)
            return False
        return True

    def _toggle_tx(self):
        """매크로 패널의 송신 / 중지 버튼 · Enter"""
        if self._transmitting:
            self._stop_tx()
            return
        if not self._need_text():
            return
        self._start_tx(self.tx_edit.text(), "매크로")

    def _send_qso(self, text):
        """송신 패널의 송신 버튼"""
        if self._transmitting:
            return
        if not text.strip():
            self.show_status(tr("main.message_is_empty2"), True)
            return
        if self.engine is None:
            self.show_status(tr("main.model_not_ready"), True)
            return
        self._start_tx(text, "송신")

    def _resend_qso(self):
        """TNG1 '다시 보내기': 방금 보낸 문장을 새 송신 (블록 경계) 으로 처음부터 → 받는 쪽 반복 합치기"""
        if self._transmitting or self.tx_mode != "TNG1":
            return
        text = getattr(self, "_last_tx1", None)
        if not text:
            self.show_status(tr("main.no_tng1_message_to"), True)
            return
        self._start_tx(text, "다시 보내기")

    def _start_tx(self, text, src):
        from stream_ui import TxAudio, TxPlayer, packet_timeline
        mode = self.tx_mode
        busy = self.ptt_busy()
        if busy:
            self.show_status(busy, True)
            return
        if mode == "TNG5":
            import tng5
            t5, bad = tng5.normalize(text)
            if bad and self._bad_ok != text:
                # 송신 전 경고: 집합 밖 글자 표시, 같은 텍스트로 한 번 더 누르면 그 글자를 빼고 송신
                self._bad_ok = text
                from tng1_app import bad_char_names
                msg = tr("main.invalid_chars_press_again").format(", ".join(bad_char_names(bad)))
                self.qtx.set_note(msg) if hasattr(self.qtx, "set_note") else None
                self.show_status(msg, True)
                return
            if not t5.strip():
                self.show_status(tr("main.no_chars_to_send"), True)
                return
        if mode == "TNG1":
            from tng1_app import normalize as n1, bad_char_names
            t1, bad = n1(text)
            if bad and self._bad_ok != text:
                self._bad_ok = text
                msg = tr("main.invalid_chars_press_again").format(", ".join(bad_char_names(bad)))
                self.qtx.set_note(msg) if hasattr(self.qtx, "set_note") else None
                self.show_status(msg, True)
                return
            if not t1.strip():
                self.show_status(tr("main.no_chars_to_send"), True)
                return
        dev = self.cb_out.currentData()
        if dev is None and not self.tx_dry:
            self.show_status(tr("main.no_output_device"), True)
            return
        fs = 48000.0 if self.tx_dry else self._device_rate(dev, output=True)
        vol = self.vol.value() / 100.0
        rc = self.rig_cfg()
        import freqshift
        # 분할 운용 (CAT 연결 때만): 소리는 1500 Hz 그대로, 무전기 송신 주파수를 다이얼 + (중심 − 1500) 으로 → RF 는 같음
        split = rc["split"] if (self.rig is not None and self.rig.state["connected"] and rc["split"] != "none") else "none"
        tx_center = freqshift.BASE_HZ if split != "none" else self.center_hz
        lead, lead_tone = 0.0, False
        if rc["ptt"] == "VOX":
            lead, lead_tone = float(rc["vox_lead_ms"]) / 1000.0, True        # VOX 앞 여유: 중심 톤 (희생 톤 앞에 덧붙임)
        else:
            lead = float(rc["tx_delay_ms"]) / 1000.0                          # PTT 켠 뒤 소리 시작까지 무음
        try:
            if mode == "TNG5":
                from tng5_app import Tx5Audio
                txa = Tx5Audio(self.engine5, text, fs)
            elif mode == "TNG1":
                from tng1_app import Tx1Audio
                txa = Tx1Audio(text, fs, fc=tx_center)            # TNG1: 선택 중심으로 바로 올림 (옮기기 필요 없음)
            else:
                txa = TxAudio(self.engine, text, fs)
            self._tx_df = 0.0 if mode == "TNG1" else tx_center - freqshift.BASE_HZ          # 16부: 소리만 선택 중심으로 옮김
            player = TxPlayer(freqshift.shift_audio(txa.audio, fs, self._tx_df) * vol, fs, dev, dry=self.tx_dry,
                              lead_s=lead, lead_tone=(tx_center, vol * 0.5) if lead_tone else None)
        except Exception as ex:
            self._audio_ok["out"] = False
            self.show_status(tr("main.playback_failed").format(ex), True)
            return
        # PTT 켜기 (VOX 는 제어 없음) → 무음 lead 동안 기다린 뒤 소리
        t_ptt = time.time()
        if self.rig is not None:
            if split == "rig":
                self.rig.set_split(True, self.dial_hz + int(round(self.center_hz - freqshift.BASE_HZ)))
            elif split == "fake":
                self._fake_split = self.dial_hz
                self.rig.set_freq(self.dial_hz + int(round(self.center_hz - freqshift.BASE_HZ)))
            err = self.rig.ptt(True)
            if err:
                player.close()
                self._split_restore()
                self.show_status(tr("main.ptt_failed") + err, True)
                return
        try:
            player.start()
            if not self.tx_dry:
                self._audio_ok["out"] = True
        except Exception as ex:
            if self.rig is not None:
                self.rig.ptt(False)                                  # 오디오 장치 오류: PTT 해제
            self._split_restore()
            self._audio_ok["out"] = False
            self.show_status(tr("main.playback_failed").format(ex), True)
            return
        self._tx = {"player": player, "txa": txa, "orig": txa, "text": text, "src": src, "vol": vol,
                    "cut": None, "stop_req": False, "t_end": None, "mode": mode, "t_ptt": t_ptt}
        self.vfo.show_tx(True)
        self._transmitting = True
        self._bad_ok = None
        for n_, b_ in self.mode_btn.items():
            b_.setEnabled(False)                                 # 송신 중 모드 전환 잠금
        if self._rx is not None and not self.chk_loop.isChecked():
            self._rx.pause(True)
        self._tx_start, self._tx_dur = time.time(), player.dur
        self._tx_audio = (player.buf, fs)
        if self.pkt_an is not None:
            self.pkt_an.submit({"kind": "tx", "text": text, "mode": mode})
        self.btn_tx.setText(tr("main.stop"))
        self.btn_tx.setProperty("transmitting", True)
        self.btn_tx.style().polish(self.btn_tx)
        self.btn_save.setEnabled(False)
        own = src == "송신"
        if mode == "TNG1":
            self._last_tx1 = txa.text                              # '다시 보내기' 용 (정규화한 문장)
        self.qtx.set_transmitting(True, (txa.text if mode in ("TNG5", "TNG1") else text) if own else "",
                                  txa.char_segments() if own else [])
        plan = txa.plan
        if mode == "TNG1":
            from tng1_app import timeline_tx
            tl = timeline_tx(txa)
            self.txana.clear("TNG1 송신")
            self.p_txd1.set_message(txa)
        else:
            tl = packet_timeline(plan.n, True, True, True, self.engine5.pc_tl if mode == "TNG5" else self.engine.pc,
                                 self.engine.cfg, [("wait", "")] * plan.n, plan, None)
            if mode == "TNG5":
                from tng5_app import timeline_v2
                tl = timeline_v2(tl)                                  # 중간 동기 M · B×1
            self.txana.set_message(txa, tl)
        now = datetime.now(timezone.utc)
        self._pid += 1
        first = text.strip().splitlines()[0] if text.strip() else ""
        self.table.add({"id": self._pid, "utc_hms": now.strftime("%H:%M:%S"),
                        "result": "{} · {}".format(mode, src if src in ("송신", "다시 보내기") else "송신 (매크로)"),
                        "result_col": "tx", "seg": "{}{}".format(plan.n_seg, "블록" if mode == "TNG1" else "구간"),
                        "text": first + (" …" if len(text.strip().splitlines()) > 1 else ""), "text_col": "tx",
                        "tip": text})
        self.session.add({"dir": "TX", "ok": True, "text": text, "duration_s": round(player.dur, 2), "ptt": rc["ptt"],
                          "src": src, "segments": plan.n_seg, "format": mode if mode in ("TNG5", "TNG1") else "스트리밍",
                          "mode_name": mode})
        self.show_status(tr("main.tx_seg_0f_hz").format(mode, 
            "가상 (장치 없음)" if self.tx_dry else self.cb_out.currentText(), fmt_clock(player.dur), plan.n_seg, fs,
            self.vol.value()))

    def tx_advance(self, n):
        """시험용 (tx_dry): 재생 위치를 n 샘플 옮기고 그 오디오를 돌려준다"""
        if self._tx is None:
            return np.zeros(0, dtype=np.float32)
        return self._tx["player"].advance(n)

    def _tx_tick(self):
        tx = self._tx
        if tx is None:
            return
        p, txa = tx["player"], tx["txa"]
        t = p.t
        done, started, st = txa.progress(t)
        cur = [str(k + 1) for k in np.nonzero(st == 1)[0][:4]]
        seg_text = tr("main.seg_done_sending").format(int(np.sum(st == 2)), len(st), ", ".join(cur) or "-")
        self.qtx.set_progress(t / max(p.dur, 1e-6), p.dur - t, st, seg_text, tx["cut"])
        if (self.docks["txmon"].isVisible() or self.docks["txana"].isVisible()) and tx.get("mode") != "TNG1":
            self.txana.update_tx(t, st, done, started, tx["cut"])
        if tx.get("mode") == "TNG1" and self.docks["txana"].isVisible():
            self.p_txd1.set_current(max(0, min(started, len(st)) - 1))
        self.tx_ind.set(True, t / max(p.dur, 1e-6), max(0.0, p.dur - t))
        if p.done or p.pos >= len(p.buf):
            if tx["t_end"] is None:
                tx["t_end"] = time.time()
            if self.tx_dry or time.time() - tx["t_end"] > 0.15:
                self._end_tx("송신 완료" if tx["cut"] is None else "송신 중지 완료 (앞 {}구간)".format(tx["cut"]))

    def _stop_tx(self):
        """첫 번째: 송신 중 구간까지 보내고 꼬리 · 코스타스 B 로 정상 종료. 두 번째: 즉시 중단"""
        tx = self._tx
        if tx is None:
            return
        if tx["cut"] is not None or tx["stop_req"]:
            tx["player"].abort()
            self._end_tx("송신 즉시 중단")
            return
        p = tx["player"]
        new, i_sw = tx["txa"].stopped(self.engine, p.t + (0.0 if self.tx_dry else 0.1))
        tx["stop_req"] = True
        if new is None:
            self.qtx.set_stopping(len(tx["orig"].P), len(tx["orig"].P))
            self.show_status(tr("main.near_end_cannot_stop"))
            return
        import freqshift
        if not p.splice(i_sw, freqshift.shift_audio(new.audio, p.fs if hasattr(p, "fs") else 48000.0, getattr(self, "_tx_df", 0.0)) * tx["vol"]):
            self.show_status(tr("main.near_end_cannot_stop"), True)
            return
        from stream_ui import packet_timeline
        from link7 import SEG_BYTES
        if tx.get("mode") in ("TNG5", "TNG1"):                   # TNG5 · TNG1: 구간 (블록) 단위 (글자 수로 표시)
            keep = min(len(new.text), new.chars_done(new.keep))
            tx["cut"] = new.keep
        else:
            keep = len(new.P)
            tx["cut"] = int(np.ceil(keep / SEG_BYTES))
        tx["txa"] = new
        tx["splice_err"] = new.splice_err
        self._tx_dur = p.dur
        self._tx_audio = (p.buf, p.fs)
        self.qtx.set_stopping(keep, len(tx["orig"].P))
        plan = new.plan
        if tx.get("mode") != "TNG1":
            tl_ = packet_timeline(plan.n, True, True, True, self.engine5.pc_tl if tx.get("mode") == "TNG5" else self.engine.pc,
                                  self.engine.cfg, [("wait", "")] * plan.n, plan, None)
            if tx.get("mode") == "TNG5":
                from tng5_app import timeline_v2
                tl_ = timeline_v2(tl_)
            self.txana.set_message(new, tl_)
        self._pid += 1
        self.table.add({"id": self._pid, "utc_hms": datetime.now(timezone.utc).strftime("%H:%M:%S"),
                        "result": "송신 중지", "result_col": "tx", "seg": "{}/{}구간".format(tx["cut"],
                                                                                        tx["orig"].plan.n_seg),
                        "text": "{} / {} 바이트".format(keep, len(tx["orig"].P)), "text_col": "tx"})
        self.session.add({"dir": "TX", "ok": True, "result": "송신 중지", "keep_bytes": keep,
                          "total_bytes": len(tx["orig"].P)})
        self.show_status(tr("main.tx_stopped_bytes_seg").format(
            keep, len(tx["orig"].P), tx["cut"], fmt_clock(p.dur - p.t)))

    def _split_restore(self):
        if self.rig is None:
            return
        rc = self.rig.cfg
        if self._fake_split is not None:
            self.rig.set_freq(self._fake_split)
            self._fake_split = None
        elif rc["split"] == "rig" and self.rig.state["connected"]:
            self.rig.set_split(False)

    def _end_tx(self, msg):
        tx, self._tx = self._tx, None
        if tx is not None:
            tx["player"].close()
        if self.rig is not None and not (self._ptt_manual or self._ptt_test):
            self.rig.ptt(False)                                      # 18부: 끝 · 중지 · 오류 모두 PTT 해제
            self._split_restore()
        self._transmitting = False
        self.vfo.show_tx(False, bool(self.rig and self.rig.state.get("ptt")))
        for n_, b_ in self.mode_btn.items():
            b_.setEnabled(n_ in registry.ENABLED)
        self.tx_ind.set(False, 1.0 if msg == "송신 완료" else 0.0, 0.0)
        if self._rx is not None and self._rx.paused:
            self._rx.pause(False)
        self.btn_tx.setText(tr("main.tx"))
        self.btn_tx.setProperty("transmitting", False)
        self.btn_tx.style().polish(self.btn_tx)
        self.btn_save.setEnabled(self.engine is not None)
        self.qtx.set_transmitting(False)
        self.qtx.set_resend(self.tx_mode == "TNG1", self.tx_mode == "TNG1" and bool(getattr(self, "_last_tx1", None)))
        self.qtx.reset_note()
        self.show_status(msg)

    # ================================================================== 종료
    # ================================================================== 설정 · About · Help
    def open_settings(self):
        from settings_dialog import SettingsDialog
        dlg = SettingsDialog(self)
        self._settings_dlg = dlg
        dlg.exec()

    def open_setup(self):
        """설정 '초기 설정 다시 하기' (언어 변경은 이 창에 바로, 앱은 재시작 후)"""
        from first_run import SetupWizard
        lang = __import__("strings_ko").LANG
        SetupWizard(self, self.settings, self).exec()
        __import__("strings_ko").set_lang(lang)            # 앱 문구는 시작 때 언어 그대로 (재시작 후 적용)

    def open_diag(self):
        from first_run import DiagDialog
        DiagDialog(self).exec()

    def open_about(self):
        from about_dialog import AboutDialog
        AboutDialog(self).exec()

    def open_help(self):
        from PySide6.QtWidgets import QMessageBox
        QMessageBox.information(self, "Help", "Coming soon")

    def audio_locked(self):
        """오디오 장치 변경을 막아야 하면 이유 문구, 아니면 빈 문자열"""
        if self._transmitting or self._tx is not None:
            return "송신 중"
        if self.qrx.btn_stop.isEnabled():
            return "메시지 수신 중"
        return ""

    def input_level_db(self):
        lv = self._in_level
        return 20 * np.log10(max(lv, 1e-6)) if self._in_stream is not None else None

    def _remember_device(self):
        if self.cb_in.count() and self.cb_out.count():
            self.settings.update({"audio_api": self.cb_api.currentText(),
                                  "audio_in": self.cb_in.currentText(), "audio_out": self.cb_out.currentText()})

    def apply_settings(self, d):
        """설정 창 값 반영 + 저장"""
        old = dict(self.settings.data)
        if "rig" in d:                                     # 18부: 부른 쪽이 나중에 고쳐도 저장 값 · 비교가 흔들리지 않게 복사
            d = dict(d)
            d["rig"] = dict(d["rig"])
        self.settings.update(d)
        self.ed_my.setText(d.get("mycall", self.ed_my.text()))
        if "volume" in d:
            self.vol.setValue(int(d["volume"]))
        if d.get("cmap") and d["cmap"] != old.get("cmap"):
            self.cb_cmap.setCurrentText(d["cmap"])
        if "qso_font" in d and int(d["qso_font"]) != int(self.qrx.size):
            self.qrx.size = int(d["qso_font"])
            self.qrx._apply_size()
        if "macros" in d:
            self.macros.macros = [dict(m) for m in d["macros"]]
            self.macros._build()
        if d.get("bands"):
            self.vfo.set_bands(d["bands"])
        if "rig" in d and (d["rig"] != (old.get("rig") or {}) or self.rig is None) and not self._transmitting:
            self._rig_start()                              # 18부: 무전기 설정 바뀜 → CAT · PTT 다시 시작 (PTT 는 닫을 때 해제)
        if "audio_api" in d and not self.audio_locked():
            if d["audio_api"] != self.cb_api.currentText():
                i = self.cb_api.findText(d["audio_api"])
                if i >= 0:
                    self.cb_api.setCurrentIndex(i)             # → 장치 목록 다시 채움 (저장된 이름으로 선택)
            for cb, key in ((self.cb_in, "audio_in"), (self.cb_out, "audio_out")):
                i = cb.findText(d.get(key) or "")
                if i >= 0 and i != cb.currentIndex():
                    cb.setCurrentIndex(i)
        if self._rx is not None and hasattr(self._rx, "set_active"):
            self._rx.set_active(self.rx_active())
        if d.get("log_dir", "") != (old.get("log_dir") or ""):
            self.show_status(tr("main.log_folder_changed_applies"))
        self._sec_tick()

    def closeEvent(self, ev):
        try:                                                      # 18부: PTT 해제가 가장 먼저 (아래에서 오류가 나도)
            if self._tx is not None:
                self._tx["player"].abort()
            if self.rig is not None:
                self.rig.close()
        except Exception:
            pass
        try:
            self._save_layout()
            self.settings["dxcall"] = self.ed_dx.text().strip().upper()
            self._stop_rx()
            if self._tx is not None:
                self._tx["player"].abort()
            for a in (self.pkt_an, self.live_an, self.zoom_an):
                if a is not None:
                    a.stop()
            self.p_dissect.shutdown()                             # 17부: 해부 창 미리 그리기 스레드 먼저 멈춤
            if self._in_stream is not None:
                self._in_stream.stop()
                self._in_stream.close()
            if self.analysis_win is not None:
                self.analysis_win.owner = None
                self.analysis_win.close()
        except Exception:
            pass
        super().closeEvent(ev)


def main():
    import argparse
    ap = argparse.ArgumentParser(description="TNG PKT 신경망 변조 디지털 모드")
    ap.add_argument("--rxlog", default=None, help="실시간 수신 사건을 이 파일에 기록")
    ap.add_argument("--quit-after", type=float, default=0.0, help="이 초 뒤 종료 (장시간 시험용)")
    ap.add_argument("--offscreen", action="store_true", help="창 없이 실행")
    ap.add_argument("--reset-layout", action="store_true", help="저장된 레이아웃을 무시하고 기본 배치")
    ap.add_argument("--no-setup", action="store_true", help="초기 설정 창 없이")
    args = ap.parse_args()
    if args.offscreen:
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        os.environ.setdefault("QT_QPA_FONTDIR", os.path.join(os.environ.get("WINDIR", "C:/Windows"), "Fonts"))
    app = QApplication.instance() or QApplication(sys.argv[:1])          # 초기 설정이 이미 만들었을 수 있음
    app.setApplicationName(APP_NAME)
    win = MainWindow(restore=not args.reset_layout)
    _hook = sys.excepthook

    def excepthook(t, v, tb):
        """18부: 처리 안 된 오류 → 송신 중이면 멈추고 PTT 해제 (오류 뒤 PTT 가 남지 않게)"""
        try:
            if win._tx is not None or win._ptt_manual or win._ptt_test:
                if win._tx is not None:
                    win._tx["player"].abort()
                    win._end_tx("오류로 송신 중단")
                win._ptt_manual = win._ptt_test = False
                if win.rig is not None:
                    win.rig.release_now()
                win.show_status(tr("main.tx_halted_by_error").format(t.__name__, v), True)
        except Exception:
            pass
        _hook(t, v, tb)
    sys.excepthook = excepthook
    win.rxlog_path = args.rxlog
    win.show()
    if args.quit_after > 0:
        QTimer.singleShot(int(args.quit_after * 1000), win.close)
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
