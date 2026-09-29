"""
TNG PKT 설정 파일 — %APPDATA%\\TNG PKT\\settings.json

  · 기본값은 DEFAULTS 한 곳. 파일에 없는 항목은 기본값.
  · "version" 으로 형식 번호를 기록. 옛 형식은 MIGRATIONS 가 차례로 새 형식으로 바꾼다.
  · 파일이 깨졌으면 settings.broken-YYYYmmdd-HHMMSS.json 으로 백업하고 기본값으로 시작.
  · 저장은 임시 파일에 쓴 뒤 os.replace (저장 중 종료돼도 원본이 깨지지 않음).
  · 첫 실행: 옛 위치(프로그램 폴더 settings/neuromod.json, logs/, 레지스트리 레이아웃)를 새 위치로 복사.
"""
import copy
import json
import os
import shutil
import time

import app_paths

SETTINGS_VERSION = 4

DEFAULT_MACROS = [
    {"name": "CQ", "text": "CQ CQ CQ DE {MYCALL} {MYCALL} K"},
    {"name": "응답", "text": "{DXCALL} DE {MYCALL} TNX FER CALL UR SIG OK"},
    {"name": "리포트", "text": "{DXCALL} DE {MYCALL} UR SNR {SNR} DB"},
    {"name": "QSL", "text": "{DXCALL} DE {MYCALL} QSL TNX"},
    {"name": "73", "text": "{DXCALL} DE {MYCALL} 73 SK"},
]

DEFAULTS = {
    "version": SETTINGS_VERSION,
    # 일반
    "mycall": "",
    "dxcall": "",
    "time_display": "both",
    "language": "ko",
    "setup_done": False,                # 24부: 초기 설정 끝냄 (파일이 있으면 초기 설정 안 띄움, 기록용)
    "diag": None,                       # 24부: 마지막 진단 결과 (diag.py 결과 dict)                   # 화면 언어 ko / en (23부, 재시작 후 적용)             # utc / local / both (상태줄 시계)
    "log_dir": "",                      # 빈 값 = %APPDATA%\TNG PKT\logs
    # 오디오 (장치는 이름 + API 이름으로 저장)
    "audio_api": "MME",
    "audio_in": "",                     # 빈 값 = 시스템 기본 장치
    "audio_out": "",
    "volume": 80,
    # 모드 (단일 모드 운용: 상단바 '모드' 하나로 송수신)
    "tx_start_mode": "last",            # 시작 시 모드: last / TNG44
    "last_tx_mode": "TNG44",
    "center_hz": 1500.0,                # 송수신 오디오 중심 (16부, 모드 대역이 300~2700 Hz 안에 들게 제한)
    # 무전기 (18부): CAT · PTT (rig.DEFAULTS 키), 다이얼 · 무전기 모드 (CAT 없을 때 수동 값 = 로그 기록용), 밴드 목록
    "rig": None,
    "dial_hz": 14078000,
    "rig_mode": "USB",
    "bands": None,
    # 수신 / 표시
    "qso_font": 18,
    "cmap": "inferno",
    "wf_speed": "보통",
    "wf_segs": True,
    "spec_range": None,
    "sync_cmap": None,
    "sync_smooth": True,
    "zoom_preset": None,
    "zoom_speed": "보통",
    "zoom_span": 300,
    "zoom_range": None,
    # 매크로
    "macros": DEFAULT_MACROS,
    "macro_now": False,
}


def _v0_to_v1(d):
    """버전 번호 없는 옛 파일(settings/neuromod.json) → 1. 항목 이름이 같으므로 번호만 붙인다."""
    d["version"] = 1
    return d


def _v1_to_v2(d):
    """자동 전환 기능 보류: 상대 모드 따르기 · SNR 추천 항목 제거"""
    d.pop("follow_peer_mode", None)
    d.pop("mode_recommend", None)
    d["version"] = 2
    return d


def _v2_to_v3(d):
    """단일 모드 운용 (09-27): 수신 모드 (전체 동시 / 선택 모드만) 항목 제거"""
    d.pop("rx_modes", None)
    d.pop("rx_selected", None)
    d["version"] = 3
    return d


def _v3_to_v4(d):
    """무전기 CAT · PTT · VFO (18부): 무전기 설정 묶음 · 다이얼 · 밴드 목록 추가 (기본 = CAT 없음 · PTT VOX, 이전과 같은 동작)"""
    import rig
    r = dict(rig.DEFAULTS)
    r.update(d.get("rig") or {})
    d["rig"] = r
    d.setdefault("dial_hz", 14078000)
    d.setdefault("rig_mode", "USB")
    d.setdefault("bands", [list(b) for b in rig.DEFAULT_BANDS])
    d["version"] = 4
    return d


MIGRATIONS = {0: _v0_to_v1, 1: _v1_to_v2, 2: _v2_to_v3, 3: _v3_to_v4}             # 이전 번호 → 변환 함수


def migrate(d):
    v = int(d.get("version", 0) or 0)
    while v < SETTINGS_VERSION:
        d = MIGRATIONS[v](d)
        v = int(d["version"])
    return d


def atomic_write_json(path, data):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fp:
        json.dump(data, fp, ensure_ascii=False, indent=1)
        fp.flush()
        os.fsync(fp.fileno())
    os.replace(tmp, path)


class Settings:
    """session.Settings 와 같은 사용법 (s[key], s[key] = v → 즉시 저장)"""

    def __init__(self, path=None):
        self.path = path or app_paths.settings_path()
        self.notice = None                              # 시작 때 상태줄에 띄울 알림 (파일 손상 등)
        self.data = copy.deepcopy(DEFAULTS)
        loaded = None
        if os.path.exists(self.path):
            try:
                with open(self.path, encoding="utf-8") as fp:
                    loaded = json.load(fp)
                if not isinstance(loaded, dict):
                    raise ValueError("최상위가 객체가 아님")
                loaded = migrate(loaded)
            except Exception as ex:
                bak = self.path.replace(".json", time.strftime(".broken-%Y%m%d-%H%M%S.json"))
                try:
                    shutil.copy2(self.path, bak)
                except OSError:
                    pass
                self.notice = "설정 파일 손상 → 기본값으로 시작 (백업: {}) — {}".format(os.path.basename(bak), ex)
                loaded = None
        if loaded:
            for k, v in loaded.items():
                self.data[k] = v
        self.data["version"] = SETTINGS_VERSION
        import rig
        r = dict(rig.DEFAULTS)                          # 새 파일 · 빠진 항목 = 기본값
        r.update(self.data.get("rig") or {})
        self.data["rig"] = r
        if not self.data.get("bands"):
            self.data["bands"] = [list(b) for b in rig.DEFAULT_BANDS]
        self.save()

    def save(self):
        try:
            atomic_write_json(self.path, self.data)
        except OSError:
            pass

    def get(self, k, default=None):
        v = self.data.get(k)
        return default if v is None else v

    def __getitem__(self, k):
        return self.data.get(k, DEFAULTS.get(k))

    def __setitem__(self, k, v):
        self.data[k] = v
        self.save()

    def update(self, d):
        self.data.update(d)
        self.save()

    def log_dir(self):
        return self.data.get("log_dir") or app_paths.default_log_dir()


# ------------------------------------------------------------------ 첫 실행 이전
def migrate_old_locations():
    """옛 위치의 설정 · 로그 · 레이아웃을 새 위치로 복사한다 (새 파일이 없을 때 한 번). 원본은 남긴다. 반환: 알림 문구 또는 None"""
    done = []
    here = os.path.dirname(os.path.abspath(__file__))
    new_s = app_paths.settings_path()
    old_s = os.path.join(here, "settings", "neuromod.json")
    if not os.path.exists(new_s) and os.path.exists(old_s):
        try:
            shutil.copy2(old_s, new_s)
            done.append("설정")
        except OSError:
            pass
    old_logs = os.path.join(here, "logs")
    new_logs = app_paths.default_log_dir()
    if os.path.isdir(old_logs) and not os.path.isdir(new_logs):
        try:
            shutil.copytree(old_logs, new_logs)
            done.append("로그")
        except OSError:
            pass
    lay = app_paths.layout_path()
    if not os.path.exists(lay):
        try:
            from PySide6.QtCore import QSettings
            old = QSettings("NeuroMod", "NeuroMod")
            keys = old.allKeys()
            if keys:
                new = QSettings(lay, QSettings.IniFormat)
                for k in keys:
                    new.setValue(k, old.value(k))
                new.sync()
                done.append("레이아웃")
        except Exception:
            pass
    return ("옛 위치에서 가져옴: " + " · ".join(done)) if done else None
