"""
진단 (24부): CPU (모드별 실시간 수신 처리 여유) · 오디오 (장치 열림 · 샘플레이트 · 끊김 · 클럭 오차). 전체 1분 이내.

    python diag.py --in "장치" --out "장치" --api MME [--json 결과.json] [--skip-audio] [--skip-cpu]

앱은 이 파일을 따로 프로세스로 돌린다 (창이 멈추지 않게). 진행은 stdout 에 'P <0~100> <단계>' 줄, 끝에 'R <json>' 한 줄.
판정 · 문구는 앱이 diag_view() 로 만든다 (언어 따라).
"""
import argparse
import json
import os
import sys
import time

from tngpkt import _io_utf8  # noqa: F401
import numpy as np

FS = 48000.0
MSG = {"TNG44": "CQ CQ DE N0CALL N0CALL K", "TNG5": "CQ CQ CQ DE N0CALL N0CALL K", "TNG1": "CQ DE N0CALL TNG1 K"}
FEED_S = {"TNG44": 16.0, "TNG5": 16.0, "TNG1": 16.0}        # 모드마다 넣는 소리 길이 (메시지 + 잡음)

# ---------------------------------------------------------------- 기준값
CPU_OK, CPU_WARN = 3.0, 1.5         # 처리 여유 (실제 시간 ÷ 처리 시간): ≥3 적합 · ≥1.5 주의 · 그 아래 부적합
PPM_OK, PPM_WARN = 200.0, 1000.0    # 클럭 오차 |ppm|: ≤200 적합 · ≤1000 주의 · 그 위 부적합
AUDIO_S = 8.0                       # 오디오 검사 길이
RUN_CAP_S = 4.0                     # 모드 하나 처리 시간 상한 (넘으면 넣은 만큼으로 계산)


def progress(p, what):
    print("P {} {}".format(int(p), what), flush=True)


# ---------------------------------------------------------------- 오디오
def _dev_index(name, api, kind):
    import sounddevice as sd
    apis = [a["name"] for a in sd.query_hostapis()]
    for i, d in enumerate(sd.query_devices()):
        if d["name"] == name and apis[d["hostapi"]] == api and d["max_{}_channels".format(kind)] > 0:
            return i
    return None


def audio_check(dev_in, dev_out, api):
    """입력 · 출력 동시에 AUDIO_S 동안 열어 끊김 · 콜백 시각으로 클럭 오차 추정 (PC 시계 기준)"""
    import sounddevice as sd
    out = {}
    for kind, name in (("input", dev_in), ("output", dev_out)):
        r = {"name": name or "", "open": False, "fs": None, "xrun": 0, "ppm": None, "ppm_err": None, "error": ""}
        out[kind] = r
        try:
            idx = _dev_index(name, api, kind) if name else (sd.default.device[0 if kind == "input" else 1])
            if idx is None or idx < 0:
                r["error"] = "no device"
                continue
            info = sd.query_devices(idx)
            r["name"] = info["name"]
            fs = FS
            try:
                (sd.check_input_settings if kind == "input" else sd.check_output_settings)(device=idx, samplerate=fs, channels=1)
            except Exception:
                fs = float(info["default_samplerate"])
            r["fs"] = fs
            r["_idx"] = idx
        except Exception as ex:
            r["error"] = str(ex)[:80]
    streams, marks = [], {"input": [], "output": []}
    t_ref = time.perf_counter()

    def cb_factory(kind):
        cnt = [0]

        def cb(*args):
            status = args[-1]
            frames = args[-3]
            if kind == "output":
                args[0].fill(0)
            cnt[0] += frames
            marks[kind].append((time.perf_counter() - t_ref, cnt[0]))
            if status:
                out[kind]["xrun"] += int(bool(status.input_overflow if kind == "input" else status.output_underflow))
        return cb
    for kind in ("input", "output"):
        r = out[kind]
        if "_idx" not in r:
            continue
        try:
            cls = sd.InputStream if kind == "input" else sd.OutputStream
            s = cls(device=r["_idx"], samplerate=r["fs"], channels=1, dtype="float32", blocksize=1024,
                    callback=cb_factory(kind))
            s.start()
            r["open"] = True
            streams.append(s)
        except Exception as ex:
            r["error"] = str(ex)[:80]
    t_end = time.time() + AUDIO_S
    while time.time() < t_end:
        time.sleep(0.25)
        progress(5 + 20 * (1 - (t_end - time.time()) / AUDIO_S), "audio")
    for s in streams:
        try:
            s.stop()
            s.close()
        except Exception:
            pass
    for kind in ("input", "output"):
        r = out[kind]
        r.pop("_idx", None)
        m = np.array(marks[kind][5:], float)                 # 처음 몇 콜백은 버림 (시작 흔들림)
        if r["open"] and len(m) > 20 and r["fs"]:
            t, n = m[:, 0], m[:, 1]
            # 콜백 시각은 흔들림이 크다 (MME 수십 ms). 0.5 s 칸마다 지연이 가장 작은 (= 가장 이른) 점만 골라 맞춤
            lag = t - n / r["fs"]
            k = np.floor((t - t[0]) / 0.5).astype(int)
            pick = [np.flatnonzero(k == j)[np.argmin(lag[k == j])] for j in np.unique(k)]
            t, n = t[pick], n[pick]
            A = np.vstack([t, np.ones_like(t)]).T
            coef = np.linalg.lstsq(A, n, rcond=None)[0]
            r["ppm"] = float((coef[0] / r["fs"] - 1) * 1e6)
            resid = n - A @ coef
            r["ppm_err"] = float(np.std(resid) / r["fs"] / max(t[-1] - t[0], 1e-3) * np.sqrt(12.0 / len(t)) * 1e6)
    return out


# ---------------------------------------------------------------- CPU
def _signal(mode, e44, e5):
    """송신기로 만든 메시지 + 잡음 (SNR 약 0 dB, 주파수 오차 +7 Hz) 을 FEED_S 길이로"""
    rng = np.random.default_rng(24)
    if mode == "TNG44":
        a = e44.encode(MSG[mode], FS)
    elif mode == "TNG5":
        from tngpkt.tng5_app import Tx5Audio
        a = Tx5Audio(e5, MSG[mode], FS).audio
    else:
        from tngpkt.tng1_app import Tx1Audio
        a = Tx1Audio(MSG[mode], FS).audio
    a = np.asarray(a, float)
    a = a / (np.max(np.abs(a)) + 1e-12) * 0.3
    n = int(FEED_S[mode] * FS)
    x = np.zeros(n)
    k = min(len(a), n - int(1.0 * FS))
    x[int(1.0 * FS):int(1.0 * FS) + k] = a[:k]
    x = x * np.cos(0) + rng.standard_normal(n) * 0.3 * np.sqrt(500.0 / 24000.0) * 0.7
    return x.astype(np.float32)


def _run_mode(m, e44, e5, x):
    """앱과 같은 수신기 묶음 (그 모드만 켬) 에 0.25 s 씩 넣고 벽시계 시간"""
    from tngpkt.realtime_rx import RealtimeReceiver
    from tngpkt.tng5_app import Tng5Receiver, MultiRx
    from tngpkt.tng1_app import Tng1Receiver, Multi3
    nop = lambda *a: None
    r44 = RealtimeReceiver(e44, FS, nop, nop, nop, sync=True)
    r5 = Tng5Receiver(e5, FS, nop, nop, nop, sync=True)
    r1 = Tng1Receiver(FS, nop, nop, nop, sync=True)
    rx = Multi3(MultiRx(r44, r5, [m]), r1, [m])
    step = int(0.25 * FS)
    w0 = time.perf_counter()
    j = 0
    for j in range(0, len(x), step):
        rx.feed(x[j:j + step])
        rx.step()
        if time.perf_counter() - w0 > RUN_CAP_S:              # 느린 PC · 바쁜 CPU: 넣은 만큼으로 여유 계산 (전체 1분 이내)
            break
    return time.perf_counter() - w0, min(len(x), j + step) / FS


def cpu_check(modes=("TNG44", "TNG5", "TNG1")):
    """여유 = 소리 길이 ÷ 처리 시간 (벽시계).
    지금 PC: 앱과 같은 장치 (GPU 있으면 GPU) · 스레드 그대로, 지금 부하 그대로.
    코어 1개: CPU 만 (GPU 안 씀) · 프로세스를 코어 하나에 묶고 torch 스레드 1 (저사양 PC 흉내)"""
    progress(28, "load")
    import torch
    import psutil
    from tngpkt.engine import ModemEngine
    from tngpkt.tng5_app import Tng5Engine
    from tngpkt import engine
    engine.app_threads()                                     # 앱과 같은 스레드 수
    if os.environ.get("TNG_TORCH_THREADS"):
        torch.set_num_threads(int(os.environ["TNG_TORCH_THREADS"]))
    e44 = ModemEngine()
    e5 = Tng5Engine(device=str(e44.device))
    out = {"threads": torch.get_num_threads(), "cores": os.cpu_count(), "device": str(e44.device)}
    xs = {}
    for i, m in enumerate(modes):                            # 30부: 판정은 코어 1개 기준만 (묶지 않은 측정은 CPU 가 바쁠 때 1.2~4.7배로 흔들려 뺌)
        progress(35 + 10 * i, m)
        xs[m] = _signal(m, e44, e5)
        out[m] = {"audio_s": len(xs[m]) / FS}
    progress(65, "1core")
    pr = psutil.Process()
    aff, nth = pr.cpu_affinity(), torch.get_num_threads()
    try:
        pr.cpu_affinity([aff[-1]])
        torch.set_num_threads(1)
        c44 = e44 if str(e44.device) == "cpu" else ModemEngine(device="cpu")
        c5 = e5 if str(e44.device) == "cpu" else Tng5Engine(device="cpu")
        for i, m in enumerate(modes):
            progress(70 + 9 * i, m + " 1core")
            w, dur = _run_mode(m, c44, c5, xs[m])
            out[m]["wall_1core_s"] = w
            out[m]["margin_1core"] = dur / max(w, 1e-6)
    finally:
        pr.cpu_affinity(aff)
        torch.set_num_threads(nth)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="dev_in", default="")
    ap.add_argument("--out", dest="dev_out", default="")
    ap.add_argument("--api", default="MME")
    ap.add_argument("--json", default="")
    ap.add_argument("--skip-audio", action="store_true")
    ap.add_argument("--skip-cpu", action="store_true")
    a = ap.parse_args()
    t0 = time.time()
    from tngpkt import registry
    res = {"time": time.strftime("%Y-%m-%d %H:%M"), "api": a.api,
           "app_version": registry.APP_VERSION, "app_build": registry.APP_BUILD}
    progress(2, "start")
    if not a.skip_audio:
        try:
            res["audio"] = audio_check(a.dev_in, a.dev_out, a.api)
        except Exception as ex:
            res["audio"] = {"error": str(ex)[:120]}
    if not a.skip_cpu:
        try:
            res["cpu"] = cpu_check()
        except Exception as ex:
            res["cpu"] = {"error": "{}: {}".format(type(ex).__name__, str(ex)[:120])}
    res["total_s"] = round(time.time() - t0, 1)
    progress(100, "done")
    if a.json:
        with open(a.json, "w", encoding="utf-8") as fp:
            json.dump(res, fp, ensure_ascii=False, indent=1)
    print("R " + json.dumps(res, ensure_ascii=False), flush=True)


# ---------------------------------------------------------------- 판정 (앱 쪽, 언어 따라)
def grade(v, ok, warn, higher_better=True):
    if v is None:
        return "bad"
    if higher_better:
        return "ok" if v >= ok else ("warn" if v >= warn else "bad")
    return "ok" if v <= ok else ("warn" if v <= warn else "bad")


def diag_view(res):
    """결과 dict → [(항목, 판정 ok/warn/bad, 이유 한 줄)] (문구는 tr 키)"""
    from tngpkt.strings_ko import tr
    rows = []
    cpu = (res or {}).get("cpu") or {}
    if cpu.get("error"):
        rows.append((tr("diag.cpu"), "bad", tr("diag.cpu_error").format(cpu["error"])))
    for m in ("TNG44", "TNG5", "TNG1"):
        c = cpu.get(m)
        if not c:
            continue
        g = grade(c["margin_1core"], CPU_OK, CPU_WARN)                  # 30부: 코어 1개 기준만
        rows.append((m, g, tr("diag.cpu_margin1").format(c["margin_1core"])))
    au = (res or {}).get("audio") or {}
    for kind, lab in (("input", tr("main.input")), ("output", tr("main.output"))):
        r = au.get(kind)
        if not r:
            continue
        if not r.get("open"):
            rows.append((lab, "bad", tr("diag.no_open").format(r.get("error") or r.get("name") or "-")))
            continue
        why = [tr("diag.fs").format(r["fs"])]
        g = "ok"
        if r.get("xrun"):
            g = "warn"
            why.append(tr("diag.xrun").format(r["xrun"]))
        if r.get("ppm") is not None:
            gp = grade(abs(r["ppm"]), PPM_OK, PPM_WARN, higher_better=False)
            g = max((g, gp), key=("ok", "warn", "bad").index)
            why.append(tr("diag.ppm").format(r["ppm"], r.get("ppm_err") or 0))
        if r["fs"] != FS:
            g = max((g, "warn"), key=("ok", "warn", "bad").index)
        rows.append((lab, g, " · ".join(why)))
    return rows


if __name__ == "__main__":
    main()
