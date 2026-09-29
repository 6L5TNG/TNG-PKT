"""
무전기 CAT · PTT (18부) — Hamlib rigctld (TCP) + 시리얼 DTR/RTS PTT.

  · CAT: 앱이 rigctld 를 실행 · 관리 (cat="run") 또는 이미 실행 중인 rigctld 에 접속 (cat="connect"). "none" = CAT 없음
  · PTT: VOX (제어 안 함) · CAT (rigctld 'T') · DTR · RTS (PTT 포트. CAT 포트와 같으면 rigctld 가 그 선을 씀)
  · 모든 CAT 입출력은 작업 스레드 하나에서 (UI 멈춤 없음). PTT 켜기 · 끄기는 대기열 맨 앞
  · 안전: PTT 가 max_s + 2 s 넘게 켜져 있으면 작업 스레드가 스스로 끔 (UI 가 멈춰도), CAT 끊김 · 종료 · 프로세스 끝 (atexit) 때 끔

Qt 없이 쓴다 (시험 · 다른 도구). 상태는 on_state(dict) 콜백으로 (작업 스레드에서 부름 → 앱이 신호로 넘김).
"""
import atexit
import glob
import os
import shutil
import socket
import subprocess
import threading
import time
from collections import deque

# 화면 표기 ↔ Hamlib 모드 이름
MODES = [("USB", "USB"), ("LSB", "LSB"), ("DATA-U", "PKTUSB"), ("DATA-L", "PKTLSB"),
         ("FM", "FM"), ("DATA-FM", "PKTFM"), ("AM", "AM"), ("CW", "CW")]
UI2HL = dict(MODES)
HL2UI = {h: u for u, h in MODES}

# 기본 밴드 · 다이얼 (Hz). 설정 '무전기' 에서 편집
DEFAULT_BANDS = [["160m", 1838000], ["80m", 3578000], ["40m", 7078000], ["30m", 10130000], ["20m", 14078000],
                 ["17m", 18104000], ["15m", 21078000], ["12m", 24922000], ["10m", 28078000], ["6m", 50318000],
                 ["2m", 144174000], ["70cm", 432174000]]
BAND_EDGES = [("160m", 1.8e6, 2.0e6), ("80m", 3.5e6, 4.0e6), ("60m", 5.3e6, 5.45e6), ("40m", 7.0e6, 7.3e6),
              ("30m", 10.1e6, 10.15e6), ("20m", 14.0e6, 14.35e6), ("17m", 18.068e6, 18.168e6),
              ("15m", 21.0e6, 21.45e6), ("12m", 24.89e6, 24.99e6), ("10m", 28.0e6, 29.7e6), ("6m", 50e6, 54e6),
              ("2m", 144e6, 148e6), ("1.25m", 222e6, 225e6), ("70cm", 420e6, 450e6)]


def band_of(hz):
    for n, lo, hi in BAND_EDGES:
        if lo <= hz <= hi:
            return n
    return ""


def rf_hz(dial, center, mode):
    """실제 송신 RF = 다이얼 ± 오디오 중심 (LSB 계열은 아래로)"""
    if dial is None:
        return None
    m = (mode or "USB").upper()
    return dial - center if m in ("LSB", "DATA-L", "PKTLSB") else dial + center


def fmt_hz(hz):
    """14074000 → '14.074.000' (MHz.kHz.Hz)"""
    if hz is None:
        return "---.---.---"
    s = "{:d}".format(int(round(hz)))
    s = s.rjust(7, "0")
    return "{}.{}.{}".format(s[:-6], s[-6:-3], s[-3:])


def parse_hz(text):
    """'14.074' → 14074000, '14.074.000' → 14074000, '14074000' → 14074000, '7074k' → 7074000. 못 읽으면 None"""
    t = (text or "").strip().lower().replace(",", "").replace(" ", "")
    if not t:
        return None
    try:
        if t.endswith("k"):
            return int(round(float(t[:-1]) * 1e3))
        if t.endswith("m"):
            return int(round(float(t[:-1]) * 1e6))
        if t.count(".") >= 2:                          # 14.074.000 (MHz.kHz.Hz)
            a, b, c = (t.split(".") + ["", ""])[:3]
            return int(a or 0) * 1000000 + int((b or "0").ljust(3, "0")[:3]) * 1000 + int((c or "0").ljust(3, "0")[:3])
        v = float(t)
        return int(round(v * 1e6)) if v < 1000 else int(round(v))       # 1000 미만은 MHz 로
    except ValueError:
        return None


# ------------------------------------------------------------------ rigctld 찾기 · 기종 목록
def find_exe(name="rigctld", hint=""):
    if hint and os.path.isfile(hint):
        return hint
    p = shutil.which(name)
    if p:
        return p
    for pat in (r"C:\Program Files\hamlib*\bin\{}.exe", r"C:\Program Files (x86)\hamlib*\bin\{}.exe"):
        c = sorted(glob.glob(pat.format(name)))
        if c:
            return c[-1]
    return None


_models = None


def rig_models(hint=""):
    """[(번호, '제조사 기종')] — rigctl -l (한 번만 읽어 둠). 못 읽으면 가상 무전기만"""
    global _models
    if _models is not None:
        return _models
    out = []
    exe = find_exe("rigctl", os.path.join(os.path.dirname(hint), "rigctl.exe") if hint else "")
    if exe:
        try:
            r = subprocess.run([exe, "-l"], capture_output=True, text=True, timeout=10,
                               creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
            for line in r.stdout.splitlines()[1:]:
                parts = line.split()
                if not parts or not parts[0].isdigit():
                    continue
                # 칸 위치로 자름 (제조사 · 기종 이름에 공백 있음)
                mfg, model = line[8:31].strip(), line[31:55].strip()
                out.append((int(parts[0]), (mfg + " " + model).strip()))
        except Exception:
            out = []
    if not out:
        out = [(1, "Hamlib Dummy"), (2, "Hamlib NET rigctl")]
    _models = out
    return out


# ------------------------------------------------------------------ 기본 설정
DEFAULTS = {
    "cat": "none",              # none / run (rigctld 실행) / connect (실행 중인 rigctld)
    "cat_via": "run",           # 19부: Rig 고르면 쓸 방식 (run / connect) — Rig 'None' 이어도 기억
    "model": 1,
    "port": "",
    "baud": 9600,
    "data_bits": 0,             # 0 = Default (Hamlib 기종 기본값), 7 · 8
    "stop_bits": 0,             # 0 = Default, 1 · 2
    "handshake": "Default",     # Default / None / XONXOFF / Hardware
    "dtr": "Unset",             # Unset / ON / OFF
    "rts": "Unset",
    "host": "127.0.0.1",
    "tcp_port": 4532,
    "poll_ms": 1000,
    "mode_set": "none",         # none / USB / DATA
    "split": "none",            # none / rig / fake
    "ptt": "VOX",               # VOX / CAT / DTR / RTS
    "ptt_port": "",
    "tx_source": "data",        # CAT PTT 소리 입력: data (Rear/Data, T 3) / mic (Front/Mic, T 2). 무전기가 거부하면 T 1
    "tx_delay_ms": 100,
    "vox_lead_ms": 0,
    "max_tx_s": 300,
    "rigctld": "",
}


class RigError(Exception):
    pass


class Rig:
    """
    CAT + PTT 한 묶음. 설정 dict 는 DEFAULTS 키.
    상태 (on_state 로 알림): connected, freq, mode (화면 표기), ptt, error, cat
    """

    def __init__(self, cfg=None, on_state=None):
        self.cfg = dict(DEFAULTS)
        self.cfg.update(cfg or {})
        self.on_state = on_state
        self.state = {"cat": self.cfg["cat"], "connected": False, "freq": None, "mode": None, "ptt": False,
                      "error": "", "ptt_error": ""}
        self._q = deque()
        self._qlock = threading.Lock()
        self._wake = threading.Event()
        self._stop = threading.Event()
        self._sock = None
        self._buf = b""
        self._slock = threading.Lock()
        self._proc = None
        self._ser = None
        self._ptt_since = None
        self._thread = None
        _LIVE.add(self)

    # ---------------------------------------------------------------- 시작 · 끝
    def start(self):
        if self._thread is not None:
            return
        if self.cfg["ptt"] in ("DTR", "RTS") and not self._ptt_via_rigctld():
            self._open_serial_ptt()
        self._thread = threading.Thread(target=self._run, name="rig", daemon=True)
        self._thread.start()

    def close(self):
        """PTT 끄고 연결 · rigctld · 시리얼 닫기 (여러 번 불러도 됨)"""
        self.release_now()
        self._stop.set()
        self._wake.set()
        t = self._thread
        if t is not None and t is not threading.current_thread():
            t.join(timeout=2.0)
        self._thread = None
        self._disconnect()
        if self._ser is not None:
            try:
                self._ser.close()
            except Exception:
                pass
            self._ser = None
        _LIVE.discard(self)

    def cat_enabled(self):
        return self.cfg["cat"] in ("run", "connect")

    def _ptt_via_rigctld(self):
        """DTR/RTS 포트가 CAT 포트와 같으면 rigctld 가 그 포트를 잡고 있으므로 rigctld 로"""
        c = self.cfg
        return c["cat"] == "run" and c["ptt"] in ("DTR", "RTS") and (not c["ptt_port"] or c["ptt_port"] == c["port"])

    def _open_serial_ptt(self):
        port = self.cfg["ptt_port"] or self.cfg["port"]
        if not port:
            self._set(ptt_error="PTT 포트 없음")
            return
        try:
            import serial
            s = serial.Serial()
            s.port = port
            s.dtr = False                       # 열 때 선이 튀지 않게 먼저 끔
            s.rts = False
            s.open()
            s.dtr = False
            s.rts = False
            self._ser = s
            self._set(ptt_error="")
        except Exception as ex:
            self._ser = None
            self._set(ptt_error="PTT 포트 열기 실패: {}".format(ex))

    # ---------------------------------------------------------------- 상태
    def _set(self, **kw):
        ch = {k: v for k, v in kw.items() if self.state.get(k) != v}
        self.state.update(kw)
        if ch and self.on_state is not None:
            try:
                self.on_state(dict(self.state))
            except Exception:
                pass

    # ---------------------------------------------------------------- 요청 (어느 스레드에서나)
    def _push(self, item, front=False):
        with self._qlock:
            if front:
                self._q.appendleft(item)
            else:
                self._q.append(item)
        self._wake.set()

    def set_freq(self, hz):
        self._push(("F", int(hz)))

    def set_mode(self, ui_mode):
        self._push(("M", UI2HL.get(ui_mode, ui_mode)))

    def set_split(self, on, tx_hz=None):
        self._push(("S", bool(on), tx_hz))

    def read_now(self):
        self._push(("poll",))

    def ptt(self, on):
        """PTT 켜기 · 끄기. 반환: 오류 문구 ('' = 요청 성공 · VOX)"""
        m = self.cfg["ptt"]
        on = bool(on)
        if m == "VOX":
            self._ptt_since = time.time() if on else None
            self._set(ptt=on, ptt_error="")
            return ""
        if m in ("DTR", "RTS") and not self._ptt_via_rigctld():
            if self._ser is None:
                self._open_serial_ptt()
            if self._ser is None:
                return self.state["ptt_error"] or "PTT 포트 없음"
            try:
                if m == "DTR":
                    self._ser.dtr = on
                else:
                    self._ser.rts = on
                self._ptt_since = time.time() if on else None
                self._set(ptt=on, ptt_error="")
                return ""
            except Exception as ex:
                self._set(ptt_error="PTT 실패: {}".format(ex))
                return self.state["ptt_error"]
        # CAT (rigctld 'T')
        if not self.cat_enabled():
            return "CAT 없음 · PTT 방식 CAT 불가"
        if on and not self.state["connected"]:
            return "CAT 연결 안 됨"
        with self._qlock:                               # 반대 요청은 버림
            for it in list(self._q):
                if it[0] == "T":
                    self._q.remove(it)
        self._ptt_since = time.time() if on else None
        self._push(("T", on), front=True)
        return ""

    def release_now(self):
        """지금 이 스레드에서 PTT 를 끈다 (종료 · 비상). 짧은 제한 시간"""
        m = self.cfg["ptt"]
        self._ptt_since = None
        try:
            if self._ser is not None:
                self._ser.dtr = False
                self._ser.rts = False
        except Exception:
            pass
        if m == "CAT" or self._ptt_via_rigctld():
            if self._slock.acquire(timeout=1.5):
                try:
                    if self._sock is not None:
                        self._cmd_locked("T 0", timeout=1.0)
                except Exception:
                    pass
                finally:
                    self._slock.release()
        self._set(ptt=False)

    def ptt_on_for(self):
        return 0.0 if self._ptt_since is None else time.time() - self._ptt_since

    # ---------------------------------------------------------------- 작업 스레드
    def _run(self):
        next_poll, retry_at = 0.0, 0.0
        while not self._stop.is_set():
            now = time.time()
            # 안전: 최대 송신 시간 + 2 s 넘으면 스스로 끔 (UI 가 멈춰도)
            if self._ptt_since is not None and now - self._ptt_since > float(self.cfg["max_tx_s"]) + 2.0:
                self._do_ptt(False)
                self._ptt_since = None
                self._set(ptt=False, ptt_error="최대 송신 시간 초과 · PTT 자동 해제")
            if self.cat_enabled() and self._sock is None and now >= retry_at:
                if not self._connect():
                    retry_at = time.time() + 3.0
                    if self._ptt_since is not None and self.cfg["ptt"] == "CAT":
                        self._ptt_since = None
                        self._set(ptt=False)
                else:
                    next_poll = 0.0
            item = None
            with self._qlock:
                if self._q:
                    item = self._q.popleft()
            if item is not None and self._sock is not None:
                self._do(item)
                continue
            if self._sock is not None and time.time() >= next_poll:
                self._poll()
                next_poll = time.time() + max(0.2, float(self.cfg["poll_ms"]) / 1000.0)
            self._wake.wait(0.1)
            self._wake.clear()
        self._disconnect()

    def _do(self, item):
        k = item[0]
        try:
            if k == "F":
                self._cmd("F {}".format(item[1]))
                self._poll()
            elif k == "M":
                self._cmd("M {} 0".format(item[1]))
                self._poll()
            elif k == "T":
                self._do_ptt(item[1])
            elif k == "S":
                on, tx = item[1], item[2]
                if on and tx:
                    self._cmd("I {}".format(int(tx)))
                self._cmd("S {} VFOB".format(1 if on else 0))
            elif k == "poll":
                self._poll()
        except RigError as ex:
            self._set(error=str(ex))

    def _do_ptt(self, on):
        if self.cfg["ptt"] == "VOX":
            self._set(ptt=False)
            return
        if self.cfg["ptt"] in ("DTR", "RTS") and not self._ptt_via_rigctld():
            try:
                if self._ser is not None:
                    self._ser.dtr = False
                    self._ser.rts = False
            except Exception:
                pass
            self._ptt_since = None
            self._set(ptt=False)
            return
        try:
            if on and self.cfg["ptt"] == "CAT" and self.cfg.get("tx_source") in ("data", "mic"):
                try:                                          # Rear/Data = 3 · Front/Mic = 2, 거부하면 일반 PTT
                    self._cmd("T {}".format(3 if self.cfg["tx_source"] == "data" else 2))
                except RigError:
                    if self._sock is None:
                        raise
                    self._cmd("T 1")
            else:
                self._cmd("T {}".format(1 if on else 0))
            self._set(ptt=bool(on), ptt_error="")
        except RigError as ex:
            self._set(ptt_error="PTT 실패: {}".format(ex))
            if on:
                self._ptt_since = None
                self._set(ptt=False)

    def _poll(self):
        try:
            f = self._cmd("f")
            m = self._cmd("m")
            freq = int(float(f[0])) if f else None
            mode = HL2UI.get(m[0], m[0]) if m else None
            self._set(freq=freq, mode=mode, error="")
        except (RigError, ValueError) as ex:
            self._set(error=str(ex))

    # ---------------------------------------------------------------- TCP
    def _connect(self):
        c = self.cfg
        if c["cat"] == "run" and (self._proc is None or self._proc.poll() is not None):
            if not self._spawn():
                return False
        t_end = time.time() + (6.0 if c["cat"] == "run" else 1.5)
        last = None
        while time.time() < t_end and not self._stop.is_set():
            try:
                s = socket.create_connection(("127.0.0.1" if c["cat"] == "run" else (c["host"] or "127.0.0.1"), int(c["tcp_port"])), timeout=1.0)
                s.settimeout(2.0)
                with self._slock:
                    self._sock, self._buf = s, b""
                self._set(connected=True, error="")
                if c["mode_set"] in ("USB", "DATA"):
                    try:
                        self._cmd("M {} 0".format("USB" if c["mode_set"] == "USB" else "PKTUSB"))
                    except RigError:
                        pass
                if c["split"] == "none":
                    try:
                        self._cmd("S 0 VFOA")
                    except RigError:
                        pass
                return True
            except OSError as ex:
                last = ex
                if self._proc is not None and self._proc.poll() is not None:
                    break
                time.sleep(0.3)
        err = "rigctld 접속 실패 ({}:{})".format(c["host"], c["tcp_port"])
        if self._proc is not None and self._proc.poll() is not None:
            try:
                tail = (self._proc.stderr.read() or b"").decode("utf-8", "replace").strip().splitlines()[-1:]
            except Exception:
                tail = []
            err = "rigctld 종료됨 (코드 {}){}".format(self._proc.returncode, (" · " + tail[0]) if tail else "")
            self._proc = None
        elif last is not None:
            err += " · {}".format(last)
        self._set(connected=False, error=err)
        return False

    def spawn_args(self):
        c = self.cfg
        exe = find_exe("rigctld", c["rigctld"])
        if not exe:
            return None
        a = [exe, "-m", str(int(c["model"])), "-t", str(int(c["tcp_port"])), "-T", "127.0.0.1"]
        if c["port"]:
            a += ["-r", c["port"], "-s", str(int(c["baud"]))]
            conf = []                                         # Default = 넘기지 않음 (Hamlib 기종 기본값)
            if int(c["data_bits"]) in (7, 8):
                conf.append("data_bits={}".format(int(c["data_bits"])))
            if int(c["stop_bits"]) in (1, 2):
                conf.append("stop_bits={}".format(int(c["stop_bits"])))
            if c["handshake"] in ("None", "XONXOFF", "Hardware"):
                conf.append("serial_handshake={}".format(c["handshake"]))
            if c["dtr"] != "Unset":
                conf.append("dtr_state={}".format(c["dtr"]))
            if c["rts"] != "Unset":
                conf.append("rts_state={}".format(c["rts"]))
            if conf:
                a += ["--set-conf=" + ",".join(conf)]
        if c["ptt"] == "CAT":
            a += ["-P", "RIG"]
        elif self._ptt_via_rigctld():
            a += ["-P", c["ptt"], "-p", c["ptt_port"] or c["port"]]
        return a

    def _spawn(self):
        a = self.spawn_args()
        if a is None:
            self._set(connected=False, error="rigctld 없음 (Hamlib 설치 또는 경로 지정)")
            return False
        try:
            self._proc = subprocess.Popen(a, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
                                          creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
            return True
        except OSError as ex:
            self._set(connected=False, error="rigctld 실행 실패: {}".format(ex))
            return False

    def _disconnect(self):
        with self._slock:
            s, self._sock = self._sock, None
        if s is not None:
            try:
                s.close()
            except OSError:
                pass
        p, self._proc = self._proc, None
        if p is not None:
            try:
                p.terminate()
                p.wait(timeout=2.0)
            except Exception:
                try:
                    p.kill()
                except Exception:
                    pass
        if self.state["connected"]:
            self._set(connected=False)

    def _cmd(self, line, timeout=2.0):
        with self._slock:
            if self._sock is None:
                raise RigError("CAT 연결 안 됨")
            return self._cmd_locked(line, timeout)

    def _cmd_locked(self, line, timeout=2.0):
        """rigctld 한 명령. get 은 값 줄들, set 은 [] (RPRT 0). 오류 · 끊김 → RigError (끊김이면 연결 닫음)"""
        s = self._sock
        try:
            s.settimeout(timeout)
            s.sendall((line + "\n").encode("ascii"))
            get = line[:1].islower()
            want = {"f": 1, "m": 2, "t": 1}.get(line.split()[0], 1) if get else 0
            lines = []
            while True:
                while b"\n" not in self._buf:
                    d = s.recv(4096)
                    if not d:
                        raise OSError("rigctld 연결 끊김")
                    self._buf += d
                ln, self._buf = self._buf.split(b"\n", 1)
                ln = ln.decode("ascii", "replace").strip()
                if ln.startswith("RPRT"):
                    code = int(ln.split()[1]) if len(ln.split()) > 1 else 0
                    if code != 0:
                        raise RigError("'{}' 실패 (RPRT {})".format(line, code))
                    return lines
                lines.append(ln)
                if get and len(lines) >= want:
                    return lines
        except (OSError, socket.timeout) as ex:
            try:
                s.close()
            except OSError:
                pass
            self._sock = None
            if self.cfg["ptt"] == "CAT" or self._ptt_via_rigctld():
                # 끊긴 동안은 끌 수 없음 → 다시 연결되면 맨 먼저 T 0 (앱은 이 상태로 송신을 멈춤)
                was = self.state["ptt"] or self._ptt_since is not None
                self._ptt_since = None
                if was:
                    with self._qlock:
                        self._q = deque(it for it in self._q if it[0] != "T")
                        self._q.appendleft(("T", False))
                self._set(connected=False, ptt=False, error="CAT 끊김: {}".format(ex),
                          ptt_error="CAT 끊김 · 다시 연결되면 PTT 해제 명령" if was else "")
            else:
                self._set(connected=False, error="CAT 끊김: {}".format(ex))
            raise RigError("CAT 끊김: {}".format(ex))

    # ---------------------------------------------------------------- 한 번 시험 (설정 창 'CAT 테스트')
    def test_once(self, timeout=8.0):
        """연결 → 주파수 · 모드 읽기. 반환 (ok, 문구). 작업 스레드가 돌고 있으면 그 상태를 기다림"""
        self.start()
        self.read_now()
        t_end = time.time() + timeout
        while time.time() < t_end:
            st = self.state
            if st["connected"] and st["freq"] is not None:
                return True, "연결됨 · {} Hz · {}".format(fmt_hz(st["freq"]), st["mode"] or "?")
            if st["error"] and not st["connected"] and "접속 실패" not in st["error"] and "종료됨" in st["error"]:
                break
            time.sleep(0.1)
        return False, self.state["error"] or "응답 없음"


# 프로세스가 끝날 때도 PTT 해제 (예외로 앱이 죽는 경우 포함)
_LIVE = set()


@atexit.register
def _release_all():
    for r in list(_LIVE):
        try:
            r.close()
        except Exception:
            pass
