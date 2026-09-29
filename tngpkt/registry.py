"""
TNG 버전 · 프로토콜 등록 — 버전 정보를 모으는 단 하나의 장소.

버전 세 축
  1) 앱 본체 (UI 틀)                  APP_VERSION + APP_BUILD + BUILD_DATE + CHANNEL
  2) 프로토콜 (TNG44 · TNG5 · TNG1)    Protocol.version (0.MINOR.PATCH, Beta) + Protocol.wire_rev (교신 호환 번호, 버전과 별개)
  3) 프로토콜 전용 전문가 모듈          ExpertModule.version, 대응 프로토콜 · 버전 범위
모드 사양 (속도 · 대역 · 인터리버 · 부호율 · 문자 집합) 도 여기 한 곳. 화면 문구는 spec_line() · spec_help() 로만 만든다.

프로토콜 = 변조 / 복조 / 동기 / 모델 / 이름 / 버전 묶음. 앱은 PROTOCOLS 목록을 읽어 쓴다.
(엔진 코드는 그대로 두고 어느 모듈이 무엇을 맡는지만 선언한다.)
"""
import hashlib
import os
from dataclasses import dataclass, field

from tngpkt import app_paths

APP_NAME = "TNG PKT"
APP_VERSION = "0.11.2"          # 0.11.0: TNG1 모드 · TNG1 전문가 창 · 다시 보내기 (정식 출시 전 MAJOR 0)
APP_BUILD = 1010
BUILD_DATE = "2026-09-29"
CHANNEL = "Beta"                 # 앱 · 프로토콜 · 전문가 모듈 모두 Beta (정식 출시 전)
AUTHOR = "6L5TNG"


def app_version_text():
    """'0.10.0 (Build 1001) Beta' — 창 제목 · 상태줄 · About 공통"""
    return "{} (Build {}) {}".format(APP_VERSION, APP_BUILD, CHANNEL)


@dataclass
class Spec:
    """
    모드 사양 (화면 표기용, 계산값). 속도 두 가지:
      data  = 데이터 구간 기준 (긴 메시지 증분 Δ글자 / Δ송신 시간, TNG5 는 중간 동기 포함)
      whole = 동기 포함 전체 (24자 메시지, 톤 시작 ~ 끝 동기 끝)
    bps = 사용자 정보 비트 (TNG44: ASCII 1글자 8비트 · TNG5: base-40 3글자 16비트).
    """
    cps_data: float
    bps_data: float
    cps_whole: float
    bps_whole: float
    bw_hz: int                       # -60 dB 점유 대역
    interleaver_s: float
    code_rate: str
    charset: str
    charset_detail: str
    whole_ref: str = "24-char msg"


@dataclass
class Protocol:
    name: str
    version: str                     # 0.MINOR.PATCH (정식 출시 전 MAJOR 0, Beta). 호환 깨짐 = MINOR, 호환 유지 수정 = PATCH
    wire_rev: int                    # 교신 호환 번호: 공중 파형 · 프레임이 달라지면 +1 (같으면 서로 교신 가능). 0 = 없음
    title: str
    model: str                       # 프로그램 폴더 기준 상대 경로
    modulator: str                   # 담당 모듈 (선언)
    demodulator: str
    sync: str
    summary: str
    spec: Spec = None
    status: str = "Active"
    _hash: str = field(default=None, repr=False)

    @property
    def model_path(self):
        return app_paths.resource(self.model)

    @property
    def model_hash(self):
        """모델 파일 SHA-256 전체 (복사용). 모델이 없으면 ''"""
        if self._hash is None:
            self._hash = ""
            if self.model:
                try:
                    h = hashlib.sha256()
                    with open(self.model_path, "rb") as fp:
                        for b in iter(lambda: fp.read(1 << 20), b""):
                            h.update(b)
                    self._hash = h.hexdigest()
                except OSError:
                    pass
        return self._hash

    @property
    def model_id(self):
        """모델 식별값 표시: SHA-256 앞 8자리 (송수신이 같은 파일인지 확인용, 전체는 model_hash)"""
        return self.model_hash[:8] or "—"


@dataclass
class ExpertModule:
    name: str
    version: str
    protocol: str
    protocol_versions: str           # 대응 프로토콜 버전 범위 (예: ">=0.1.0,<0.2.0")
    module: str


PROTOCOLS = [
    Protocol(name="TNG44", version="0.1.0", wire_rev=1, title="TNG44", model="models/stage3.pt",
             modulator="modem5.text_to_baseband5 · link7 스트리밍 v2",
             demodulator="engine.ModemEngine · realtime_rx · stream_ui",
             sync="preamble.CostasDetector (톤 300ms + 코스타스 A / B)",
             summary="500 bps raw, R1/2 conv + CRC16 per 32 B segment, 1500 Hz center",
             spec=Spec(cps_data=29.5, bps_data=236, cps_whole=11.2, bps_whole=89, bw_hz=470, interleaver_s=3.2,
                       code_rate="R1/2", charset="UTF-8",
                       charset_detail="any UTF-8 text (cps counted for ASCII; Hangul = 3 bytes)")),
    Protocol(name="TNG5", version="0.4.0", wire_rev=4, title="TNG5", model="models/stage3_strong.pt",
             modulator="tng5.tx_waveform (charset1 57-char Huffman packed per segment, 18 B segments + CRC16 "
                       "(first 2 B = segment count), K7 R1/4 x4, interleaver 9.6 s)",
             demodulator="tng5.decode_llr",
             sync="tng5.detect v3 (tone + Welch 12-Costas 64 ms/31.25 Hz; mid M_k every 20 frames, 4 distinct 7-Costas; "
                  "end B12 x1 (12-Costas 32 ms/31.25 Hz); GFSK, +2.25 dB, whole-TX LPF; data envelope clip +3 dB + LPF x3 (PAPR 4.3 dB); start confirmed by M1 or first CRC)",
             summary="500 bps raw, K7 R1/4 x4, CRC16 per 18 B segment, 1500 Hz center. "
                     "Wire rev 1 (A x4 / B x4): WAV open only. Rev 2 · 3 (base-40 text): not decoded",
             spec=Spec(cps_data=4.8, bps_data=24.6, cps_whole=3.2, bps_whole=17.0, bw_hz=423, interleaver_s=9.6,
                       code_rate="R1/16", charset="57-char set",
                       charset_detail="A-Z 0-9 space line-break . , ? ' / - = ( ) ! : + \" @ & ; * # % "
                                      "(same charset1 as TNG1, avg 4.9 bits/char on QSO text)")),
    Protocol(name="TNG1", version="0.1.1", wire_rev=2, title="TNG1", model="",
             modulator="tng1.gfsk (16-GFSK 160 ms · 6.25 Hz, up-chirp 50 Hz/symbol, chirp return BT 8; no NN)",
             demodulator="tng1.decode_at (dechirp + FFT, symbol-trellis tail-biting conv K13 R1/4, multi-symbol W2 on CRC fail)",
             sync="tng1.fold_metric (12 scattered sync tones per 14.7 s block, folded over up to 4 blocks; candidates + CRC16; "
                  "no time slots, join any block)",
             summary="Streaming 14.7 s blocks (64 text bits + CRC16; first block of a message marked by CRC16 init), "
                     "57-char Huffman charset, 1500 Hz center, impulse blanker, repeat combining (no marker). "
                     "Wire rev 1 (0.1.0, no first-block mark): WAV open only",
             spec=Spec(cps_data=0.90, bps_data=4.35, cps_whole=0.82, bps_whole=4.08, bw_hz=356, interleaver_s=14.7,
                       code_rate="R1/4 K13", charset="57-char set",
                       charset_detail="A-Z 0-9 space line-break . , ? ' / - = ( ) ! : + \" @ & ; * # % "
                                      "(Huffman, avg 4.84 bits/char on QSO text)")),
]
ENABLED = ("TNG44", "TNG5", "TNG1")  # 앱에서 송수신에 쓰는 프로토콜

# 대응 범위 = 프로토콜 버전 (MAJOR.MINOR). TNG5 1.x (wire rev 1) 는 WAV 열기 전용 → v1 분석 경로. 2.x · 3.x 는 끝 B 만 다르고 같은 분석
EXPERT_MODULES = [
    ExpertModule("Constellation", "0.1.0", "TNG44", ">=0.1.0,<0.2.0", "widgets.ConstellationPanel"),
    ExpertModule("Eye diagram", "0.1.0", "TNG44", ">=0.1.0,<0.2.0", "widgets.EyePanel"),
    ExpertModule("NN features", "0.1.0", "TNG44", ">=0.1.0,<0.2.0", "widgets.FeaturePanel"),
    ExpertModule("LLR distribution", "0.1.0", "TNG44", ">=0.1.0,<0.2.0", "widgets.LLRPanel"),
    ExpertModule("Costas sync map", "0.1.0", "TNG44", ">=0.1.0,<0.2.0", "widgets.SyncMapPanel"),
    ExpertModule("Packet dissector", "0.1.0", "TNG44", ">=0.1.0,<0.2.0", "widgets_dissect.DissectPanel"),
    ExpertModule("TX encoder view", "0.1.0", "TNG44", ">=0.1.0,<0.2.0", "widgets_qso.TxAnalysisPanel"),
    ExpertModule("Packet dissector", "0.2.0", "TNG5", ">=0.2.0,<0.5.0", "widgets_dissect5.Dissect5Panel · dissect5"),
    ExpertModule("Constellation", "0.2.0", "TNG5", ">=0.2.0,<0.5.0", "widgets.ConstellationPanel (tng5_app.expert)"),
    ExpertModule("Eye diagram", "0.2.0", "TNG5", ">=0.2.0,<0.5.0", "widgets.EyePanel (tng5_app.expert)"),
    ExpertModule("NN features / phase trajectory", "0.2.0", "TNG5", ">=0.2.0,<0.5.0", "widgets.FeaturePanel (tng5_app.expert)"),
    ExpertModule("LLR distribution", "0.2.0", "TNG5", ">=0.2.0,<0.5.0", "widgets.LLRPanel (tng5_app.analyze)"),
    ExpertModule("Sync map (Welch12 + mid sync M)", "0.2.0", "TNG5", ">=0.2.0,<0.5.0", "widgets.SyncMapPanel (tng5_app.sync_map)"),
    ExpertModule("Packet timeline", "0.2.0", "TNG5", ">=0.2.0,<0.5.0", "widgets.TimelinePanel (tng5_app.timeline_v2)"),
    ExpertModule("TX encoder view", "0.2.0", "TNG5", ">=0.2.0,<0.5.0", "widgets_qso.TxAnalysisPanel (tng5_app.Tx5Audio)"),
    ExpertModule("Waterfall segments", "0.2.0", "TNG5", ">=0.2.0,<0.5.0", "neuromod_app._ov_items5"),
    ExpertModule("Legacy analysis (WAV open)", "0.1.0", "TNG5", ">=0.1.0,<0.2.0", "tng5_v1_app.analyze"),
    ExpertModule("Waterfall segments", "0.1.0", "TNG1", ">=0.1.0,<0.2.0", "neuromod_app._ov_items (tng1_app.blocks_overlay)"),
    ExpertModule("Packet timeline", "0.1.0", "TNG1", ">=0.1.0,<0.2.0", "widgets.TimelinePanel (tng1_app.analyze)"),
    ExpertModule("16-tone grid (RX / TX)", "0.1.0", "TNG1", ">=0.1.0,<0.2.0", "widgets_tng1.ToneGridPanel (tng1_app.grid_snapshot)"),
    ExpertModule("Symbol confidence", "0.1.0", "TNG1", ">=0.1.0,<0.2.0", "widgets_tng1.ConfPanel"),
    ExpertModule("Sync fold map", "0.1.0", "TNG1", ">=0.1.0,<0.2.0", "widgets_tng1.FoldPanel (tng1_app.fold_snapshot)"),
    ExpertModule("Sync tones (12)", "0.1.0", "TNG1", ">=0.1.0,<0.2.0", "widgets_tng1.SyncTonesPanel"),
    ExpertModule("Packet dissector", "0.1.1", "TNG1", ">=0.1.0,<0.2.0", "widgets_tng1.Dissect1Panel (CRC init: first block, rev 2)"),
    ExpertModule("TX encoder view", "0.1.1", "TNG1", ">=0.1.1,<0.2.0", "widgets_tng1.TxDissect1Panel"),
    ExpertModule("WAV open", "0.1.1", "TNG1", ">=0.1.0,<0.2.0", "tng1_app.decode_file (rev 2, then rev 1)"),
]

# 교신용 AI 모델 · 규격 표 (교신에 실제로 쓰이는 것만. 실험 · 학습용 모델은 넣지 않음).
# 송신 신경망이 바뀌면 MINOR + 그 모드 wire rev +1 · 수신 신경망만 개선하면 PATCH · TNG1 문자표가 바뀌면 MINOR + TNG1 wire rev +1
@dataclass
class AIModel:
    name: str
    version: str                     # 0.MINOR.PATCH (Beta), "—" = 해당 없음
    used_by: str
    role: str                        # "TX+RX" · "Charset" · "—"
    path: str = ""                   # 모델 파일 (프로그램 폴더 기준). 문자표는 "" (표 내용으로 해시)
    expected: str = ""               # 등록된 식별값 (SHA-256 전체). 파일 · 표가 이것과 다르면 경고
    _hash: str = field(default=None, repr=False)

    @property
    def actual(self):
        """지금 파일 (또는 문자표) 의 SHA-256 전체. 없으면 ''"""
        if self._hash is None:
            self._hash = ""
            try:
                if self.role == "Charset":
                    self._hash = charset_hash()
                elif self.path:
                    h = hashlib.sha256()
                    with open(app_paths.resource(self.path), "rb") as fp:
                        for b in iter(lambda: fp.read(1 << 20), b""):
                            h.update(b)
                    self._hash = h.hexdigest()
            except Exception:
                pass
        return self._hash

    @property
    def model_id(self):
        return self.expected[:8] if self.expected else "—"

    @property
    def ok(self):
        return not self.expected or self.actual == self.expected


def charset_hash():
    """TNG1 문자표 식별값: 57자 + EOT 의 (글자, 허프만 부호) 목록 SHA-256 (코드 주석이 바뀌어도 표가 같으면 같음)"""
    from tngpkt import charset1
    body = "".join("{:04x}:{};".format(ord(c), "".join(map(str, charset1.CODE[c]))) for c in charset1.SYMS)
    return hashlib.sha256(body.encode("ascii")).hexdigest()


AI_MODELS = [
    AIModel("TNG44 NN (stage3)", "0.1.0", "TNG44", "TX+RX", "models/stage3.pt",
            "e73e59a5e2d7dade6907140a6243eb87793b48dcea6570be543a5cf32533d30c"),
    AIModel("TNG5 NN (stage3_strong)", "0.1.0", "TNG5", "TX+RX", "models/stage3_strong.pt",
            "2cbd875d00c3408459d147a6dac28d8a09713c0592b2253b81e1115e59981c4c"),
    AIModel("No AI model (GFSK)", "—", "TNG1", "—"),
    AIModel("TNG1 charset (57-char Huffman)", "0.1.0", "TNG1", "Charset", "", "6f02debc92ee8b32fec02101e3bcf3010836b9c6bc092366e79fc4153b1e8808"),
]


def model_check():
    """등록 식별값과 다른 모델 · 문자표 목록 [(AIModel, 지금 값)] (앱 시작 때 상태줄 경고용)"""
    return [(m, m.actual) for m in AI_MODELS if m.expected and not m.ok]


# 외부 라이브러리 (배포 고지용): 배포 이름, 표시 이름, 라이선스
LIBRARIES = [
    ("PySide6", "PySide6 (Qt for Python)", "LGPL-3.0"),
    ("torch", "PyTorch", "BSD-3-Clause"),
    ("numpy", "NumPy", "BSD-3-Clause"),
    ("scipy", "SciPy", "BSD-3-Clause"),
    ("pyqtgraph", "pyqtgraph", "MIT"),
    ("sounddevice", "python-sounddevice (PortAudio: MIT)", "MIT"),
]


def protocol(name):
    return next(p for p in PROTOCOLS if p.name == name)


def _n(v):
    return "{:.1f}".format(v) if v < 100 else "{:.0f}".format(v)


def spec_line(name, full=False):
    """
    모드 사양 한 줄 (영어, 두 모드 같은 순서: 속도 · 대역 · 인터리버 · 부호율 · 문자 집합). SNR · dB 수치는 넣지 않는다.
    full=False: 상태줄용 핵심 (데이터 구간 속도 · 대역 · 부호율 · 문자 집합). 전체는 full=True · spec_help · About
    """
    p = protocol(name)
    s = p.spec
    if s is None:
        return "{} · {}".format(p.name, p.status)
    if not full:
        return "{} · {} cps · {} bps (data) · {} Hz · {} · {}".format(
            p.name, _n(s.cps_data), _n(s.bps_data), s.bw_hz, s.code_rate, s.charset)
    return "{} · {} cps · {} bps (data) · {} cps · {} bps (whole, {}) · {} Hz · Interleaver {:.1f} s · {} · {}".format(
        p.name, _n(s.cps_data), _n(s.bps_data), _n(s.cps_whole), _n(s.bps_whole), s.whole_ref,
        s.bw_hz, s.interleaver_s, s.code_rate, s.charset)


def spec_help(name):
    """'?' 도움말 · 툴팁용 여러 줄 (영어)"""
    p = protocol(name)
    s = p.spec
    if s is None:
        return "{}: {}".format(p.name, p.status)
    return "\n".join([
        "{} {} {} · Wire rev {} · Model {}".format(p.name, p.version, CHANNEL, p.wire_rev, p.model_id),
        "Speed (data section): {} cps · {} bps".format(_n(s.cps_data), _n(s.bps_data)),
        "Speed (whole TX incl. sync, {}): {} cps · {} bps".format(s.whole_ref, _n(s.cps_whole), _n(s.bps_whole)),
        "Bandwidth: {} Hz".format(s.bw_hz),
        "Interleaver: {:.1f} s".format(s.interleaver_s),
        "Code rate: {}".format(s.code_rate),
        "Character set: {} ({})".format(s.charset, s.charset_detail)])


def version_report():
    """버그 신고용 전체 버전 정보 (평문, About 'Copy Version Info')"""
    import platform
    import sys
    out = ["{} {}".format(APP_NAME, app_version_text()), "Build date: {}".format(BUILD_DATE),
           "Author: {}".format(AUTHOR), "OS: {} · Python {}".format(platform.platform(), sys.version.split()[0]),
           "", "[Protocols]"]
    for p in PROTOCOLS:
        out.append("{:<6} {:<4} wire rev {:<2} model {} · {}".format(
            p.name, p.version + " " + CHANNEL, p.wire_rev or "—", p.model_hash or "—", p.status))
    out += ["", "[AI models]"]
    for m in AI_MODELS:
        out.append("{:<34} {:<10} {:<6} {:<8} {}{}".format(
            m.name, m.version + (" " + CHANNEL if m.version != "—" else ""), m.used_by, m.role, m.expected or "—",
            "" if m.ok else "  (MISMATCH: file " + (m.actual or "missing") + ")"))
    out += ["", "[Expert modules]"]
    for m in EXPERT_MODULES:
        out.append("{:<34} {:<10} {} {}".format(m.name, m.version + " " + CHANNEL, m.protocol, m.protocol_versions))
    out += ["", "[Libraries]"]
    for title, v, lic in library_versions():
        out.append("{:<40} {:<14} {}".format(title, v, lic))
    return "\n".join(out)


def library_versions():
    from importlib.metadata import version, PackageNotFoundError
    out = []
    for dist, title, lic in LIBRARIES:
        try:
            v = version(dist)
        except PackageNotFoundError:
            v = "not installed"
        out.append((title, v, lic))
    return out


# 배포 1단계의 고정 입력 ONNX 변환 검증 산출물 (앱에서는 아직 사용하지 않음).
# 가변 복소 수신 배치 내보내기 오류로 현재 추론은 원래 CPU PyTorch 경로 유지.
ONNX_EXPORT_HASHES = {'stage3_tx.onnx': '0d4198b1a406c8b86c09125d1a91290ee44b08da3a3045a1614a5057f4151b9e', 'stage3_rx.onnx': '3e9777664bb0d01fa7bf4bf8b34bb524bab615e25342dcdf94228563cb18902c', 'stage3_strong_tx.onnx': '3deceadc79babb820a438f8ff1e384ad1bf0336af9851080694cf699c75ccb32', 'stage3_strong_rx.onnx': 'ddbeb5e8cede35fabf2e43e7ac4c407bc0ca0d1756b80d794f19e9862ff304ac'}
