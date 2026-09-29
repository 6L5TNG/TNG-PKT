"""
세션 기록과 사용자 설정.

  · SessionLog : 수신/송신을 세션마다 logs/session_YYYYmmdd_HHMMSSZ.jsonl 에 한 줄씩 남긴다.
                 export_csv() 로 표 형식 CSV 를 만든다.
  · Settings   : settings/neuromod.json — 매크로, 내 호출부호, 상대 호출부호, 워터폴 컬러맵 등.
                 사람이 직접 고쳐도 된다.
"""

import csv
import json
import os
from tngpkt import app_paths
import threading
from datetime import datetime, timezone

HERE = app_paths.resource_dir()
LOG_DIR = os.path.join(HERE, "logs")
SETTINGS_PATH = os.path.join(HERE, "settings", "neuromod.json")

DEFAULT_MACROS = [
    {"name": "CQ", "text": "CQ CQ CQ DE {MYCALL} {MYCALL} K"},
    {"name": "응답", "text": "{DXCALL} DE {MYCALL} TNX FER CALL UR SIG OK"},
    {"name": "리포트", "text": "{DXCALL} DE {MYCALL} UR SNR {SNR} DB"},
    {"name": "QSL", "text": "{DXCALL} DE {MYCALL} QSL TNX"},
    {"name": "73", "text": "{DXCALL} DE {MYCALL} 73 SK"},
]

CSV_FIELDS = ["utc", "dir", "ok", "snr_db", "df_hz", "margin_db", "fec_fixed", "fec_total",
              "ber_hard", "mode", "dial_hz", "rf_hz", "text"]


def utc_now():
    return datetime.now(timezone.utc)


class SessionLog:
    def __init__(self, folder=LOG_DIR):
        os.makedirs(folder, exist_ok=True)
        self.start = utc_now()
        self.path = os.path.join(folder, "session_{}.jsonl".format(self.start.strftime("%Y%m%d_%H%M%SZ")))
        self.rows = []
        self._lock = threading.Lock()
        self.extra = None                    # 18부: 모든 행에 붙일 값 (다이얼 · RF 주파수) 을 돌려주는 함수

    def add(self, row):
        row = dict(row)
        if self.extra is not None:
            try:
                for k, v in self.extra().items():
                    row.setdefault(k, v)
            except Exception:
                pass
        row.setdefault("utc", utc_now().strftime("%Y-%m-%dT%H:%M:%S.%fZ")[:-4] + "Z")
        with self._lock:
            self.rows.append(row)
            try:
                with open(self.path, "a", encoding="utf-8") as fp:
                    fp.write(json.dumps(row, ensure_ascii=False) + "\n")
            except OSError:
                pass
        return row

    def export_csv(self, path):
        with self._lock:
            rows = list(self.rows)
        with open(path, "w", newline="", encoding="utf-8-sig") as fp:     # 엑셀에서 한글이 깨지지 않게 BOM
            w = csv.DictWriter(fp, fieldnames=CSV_FIELDS, extrasaction="ignore")
            w.writeheader()
            for r in rows:
                w.writerow({k: r.get(k, "") for k in CSV_FIELDS})
        return len(rows)

    def stats(self):
        with self._lock:
            rx = [r for r in self.rows if r.get("dir") == "RX"]
            tx = [r for r in self.rows if r.get("dir") == "TX"]
        ok = [r for r in rx if r.get("ok")]
        snrs = [r["snr_db"] for r in rx if isinstance(r.get("snr_db"), (int, float))]
        return {"rx": len(rx), "ok": len(ok), "tx": len(tx),
                "rate": (len(ok) / len(rx)) if rx else None,
                "snr_avg": (sum(snrs) / len(snrs)) if snrs else None}


class Settings:
    def __init__(self, path=SETTINGS_PATH):
        self.path = path
        self.data = {"macros": [dict(m) for m in DEFAULT_MACROS], "mycall": "", "dxcall": "",
                     "cmap": "inferno"}
        try:
            with open(path, encoding="utf-8") as fp:
                self.data.update(json.load(fp))
        except (OSError, ValueError):
            self.save()

    def save(self):
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        with open(self.path, "w", encoding="utf-8") as fp:
            json.dump(self.data, fp, ensure_ascii=False, indent=1)

    def __getitem__(self, k):
        return self.data.get(k)

    def __setitem__(self, k, v):
        self.data[k] = v
        self.save()


def expand_macro(text, mycall="", dxcall="", snr=None):
    return (text.replace("{MYCALL}", mycall or "NOCALL").replace("{DXCALL}", dxcall or "")
            .replace("{SNR}", "{:+.0f}".format(snr) if snr is not None else "").strip())
