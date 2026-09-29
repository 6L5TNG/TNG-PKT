"""
TNG5 패킷 해부 자료 — TNG44 해부 창 (widgets_dissect.DissectPanel, 8층 · 개요 · 페이지 · 층간 클릭) 형식으로.
분석 스레드에서 계산한다 (tng5_app.analyze). 엔진 함수는 읽기만 한다.

층 (가로축 = 송신 시작 (톤) 부터의 시간 [s])
  1 오디오 파형 · 2 기저대역 I/Q (주파수 보정 + 시작 동기 위상)
  3 구간: 톤 / 시작 동기 Welch12 / 가드 / 프레임 (20프레임마다 중간 동기 M_k) / 가드 / B×1
  4 수신 NN LLR (송신 순서, 중간 동기 틈 반영)
  5 디인터리빙 + 반복 4회 결합 (구간마다 664 비트 = R1/4 부호어, 사본 4개 LLR 합)
  6 비터비 정보 비트 (구간마다 18바이트 + CRC16 = 160 + 꼬리 6)
  7 18바이트 구간 CRC16 (첫 구간 앞 2바이트 = 길이 필드)
  8 바이트 (흰색화 되돌린 값) → 글자 (0.4.0 / rev 4: charset1 가변 길이, 구간마다 글자 단위)
v1 파일 (WAV 열기, info['format_version'] == '1') 은 중간 동기 없이 같은 층으로.
"""
import numpy as np

from tngpkt import tng5
from tngpkt.framing import crc16_ccitt
from tngpkt.modem import audio_to_baseband

L = 384
BPF = tng5.BITS_PER_FRAME
SC = tng5.SEG_CODED
NCW = SC // tng5.REP                      # 결합 부호어 664 (= (160 + 6) × 4)
NI = tng5.SEG_INFO + tng5.TAIL            # 166
IB = 168                                  # 구간당 정보 칸 (바이트 경계 맞춤: 160 + 꼬리 6 + 빈 2) = 21바이트
NB = IB // 8                              # 21


def b_state(d, state, off_s=0.0):
    """해부 자료의 끝 가드 · B 를 'ok' (인정, off_s = 예상 자리와의 차이) 또는 'missing' 으로 (새 dict)"""
    if d is None:
        return d
    sl = [x for x in d["segs"] if x[0] not in ("B 없음", "코스타스 B")]
    slot = d.get("b_slot")
    if slot is None:
        return d
    t0, t1 = slot
    sl.append(("코스타스 B", t0 + off_s, t1 + off_s, "costas") if state == "ok" else ("B 없음", t0, t1, "missing"))
    return dict(d, segs=sl, b_state=state)


def dissect5(res, cfg, e, llr, segs):
    from tngpkt.conv_sim import encode as conv_encode
    viz = res.viz or {}
    m = viz.get("msg") or {}
    info = res.info or {}
    if llr is None or "audio" not in viz or not m.get("a"):
        return None
    v1 = info.get("format_version") == "1"
    fsb, sps, S = float(cfg.fs_base), cfg.sps, cfg.symbols_per_frame
    n = int(llr.shape[0])
    nseg = len(segs)
    if nseg == 0:
        return None
    a_start = int(round(float(m["a"]["start"])))
    ntone = int(round(tng5.PC.tone_ms * 1e-3 * fsb))
    if v1:
        n_sync_a, data0_off = 4 * tng5.N7, 4 * tng5.N7 + ntone + L
        f_off = lambda f: np.asarray(f) * L
        b_rel = lambda nf: (nf + 1) * L
        n_b, mids = 4 * tng5.N7, []
    else:
        n_sync_a, data0_off = tng5.N12, e.data0
        f_off = tng5.frame_off
        b_rel = tng5.b_off
        n_b = tng5.NB if info.get("format_version") != "2" else tng5.N7
        mids = list(range(1, tng5.n_mids(n) + 1))
    org = a_start                                                        # 시간 0 = 톤 시작 (= 송신 시작)
    data0 = a_start + data0_off + int(info.get("h_off", 0))
    b = m.get("b")
    b_start = int(round(float(b["start"]))) if b else data0 + int(b_rel(n))
    end = b_start + n_b
    T = lambda i: (np.asarray(i, dtype=np.float64) - org) / fsb

    # 1 오디오
    x = np.asarray(viz["audio"], dtype=np.float64)
    fs = float(viz["fs"])
    k = fs / fsb
    i0, i1 = max(0, int(org * k)), min(len(x), int(end * k))
    blk = max(1, int(round(fs * 0.001)))
    nb_ = max(1, (i1 - i0) // blk)
    sg = x[i0:i0 + nb_ * blk].reshape(nb_, blk)
    audio = {"t": (i0 + (np.arange(nb_) + 0.5) * blk) / fs - org / fsb, "lo": sg.min(axis=1), "hi": sg.max(axis=1)}

    # 2 기저대역 (주파수 보정 + 시작 동기 상관 위상)
    bb = audio_to_baseband(x, fs, cfg)
    df = float(m.get("df") or 0.0)
    y = bb * np.exp(-2j * np.pi * df * np.arange(len(bb)) / fsb)
    acc = 0j
    for off, tmpl, kind in (e.A if not v1 else []):
        p0 = a_start + off
        sg_ = y[p0:p0 + len(tmpl)]
        if kind == "costas" and len(sg_) == len(tmpl):
            acc += np.sum(sg_ * np.conj(tmpl))
    if acc != 0:
        y = y * np.exp(-1j * np.angle(acc))
    lo, hi = max(org, 0), min(end, len(y))
    yb = y[lo:hi]
    d0, d1 = max(data0 - lo, 0), min(data0 + int(f_off(n)) - lo, len(yb))
    rms = np.sqrt(np.mean(np.abs(yb[d0:d1]) ** 2)) if d1 > d0 else 1.0
    base = {"t": T(np.arange(lo, hi)), "i": yb.real / max(rms, 1e-12), "q": yb.imag / max(rms, 1e-12)}

    # 3 구간 (프레임 · 중간 동기)
    lab_a = "코스타스 A×4" if v1 else "시작 동기 Welch12"
    sl = [("톤", org, org + ntone, "tone"), (lab_a, org + ntone, org + ntone + n_sync_a, "costas"),
          ("가드", data0 - L, data0, "guard")]
    frame_t = np.zeros(n)
    for f in range(n):
        s0 = data0 + int(f_off(f))
        if not v1 and f > 0 and f % tng5.MID_EVERY == 0:
            sl.append(("M{}".format(f // tng5.MID_EVERY), s0 - tng5.NM, s0, "costas"))
        sl.append(("프레임 {}".format(f + 1), s0, s0 + L, "frame"))
        frame_t[f] = float(T(s0))
    g0 = data0 + int(f_off(n - 1)) + L
    if not b and info.get("b_wait"):                                    # 길이 필드로 먼저 복원: B 는 인정되면 그 자리에 그린다
        sl += [("가드", g0, g0 + L, "guard")]
        b_slot = (float(T(b_start)), float(T(end)))
    else:
        b_slot = None
        sl += [("가드", g0, g0 + L, "guard"), (("코스타스 B" if b else "B 없음"), b_start, end, "costas" if b else "missing")]
    sl = [(nm, float(T(s0)), float(T(s1)), kd) for nm, s0, s1, kd in sl]

    # 4 LLR (송신 순서) — 비트 시각은 실제 프레임 위치 (중간 동기 틈 반영)
    ntx = n * BPF
    p = np.arange(ntx)
    f_of, b_of = p // BPF, p % BPF
    t_bit = frame_t[f_of] + ((b_of // 2) * sps + sps / 2.0) / fsb
    llr_tx = llr.reshape(-1).astype(np.float64)
    blocks = []                                                           # 연속 프레임 묶음 (그림 · 클릭)
    step = n if v1 else tng5.MID_EVERY
    for f in range(0, n, step):
        f2 = min(f + step, n)
        blocks.append((f, f2, float(frame_t[f]), float(frame_t[f2 - 1] + L / fsb)))

    # 5 디인터리빙 + 반복 결합: 결합 비트 c (구간 k, 0..663) ← 송신 위치 4개
    q = tng5.interleave_map(nseg * SC)
    c2tx = np.empty((nseg * NCW, tng5.REP), np.int64)
    for kk in range(nseg):
        qq = q[kk * SC:(kk + 1) * SC].reshape(tng5.REP, NCW)
        c2tx[kk * NCW:(kk + 1) * NCW] = qq.T
    llr_rep = llr_tx[c2tx]                                                # (ncoded, 4)
    llr_comb = llr_rep.sum(1)
    c2i = (np.arange(nseg * NCW) // NCW) * IB + (np.arange(nseg * NCW) % NCW) // tng5.RATE_N

    # 6 · 7 · 8 정보 비트 · 구간 CRC · 바이트
    info_bits = np.full(nseg * IB, -1, np.int64)
    byts = np.zeros(nseg * NB, np.int64)
    kinds = ["unk"] * (nseg * NB)
    rows, groups = [], []
    known_c = np.zeros(nseg * NCW, bool)
    rec = np.zeros(nseg * NCW, np.int64)
    nchars = info.get("chars")
    for r in sorted(segs, key=lambda r: r["index"]):
        kk = r["index"]
        if kk >= nseg:
            continue
        ok = bool(r["ok"])
        cw_frames = np.unique(q[kk * SC:(kk + 1) * SC] // BPF)
        row = {"k": kk, "ok": ok, "decided": True, "offset": None, "relock": False, "score": None,
               "bytes": (kk * NB, tng5.SEG_BYTES), "info": (kk * IB, kk * IB + NI), "frames": cw_frames,
               "crc_rx": None, "crc_calc": None, "text": "",
               "rep_llr": np.abs(llr_rep[kk * NCW:(kk + 1) * NCW]).mean(0),
               "len_field": None}
        data = r["data"]
        crc = crc16_ccitt(data)
        full = tng5.whiten(data) + crc.to_bytes(2, "big")                 # rev 4: 보내는 비트 = 흰색화 뒤
        b0 = kk * NB
        if ok:
            bits = np.unpackbits(np.frombuffer(full, np.uint8)).astype(np.int64)
            info_bits[kk * IB:kk * IB + 160] = bits
            info_bits[kk * IB + 160:kk * IB + NI] = 0
            byts[b0:b0 + 20] = np.frombuffer(data + crc.to_bytes(2, "big"), np.uint8)    # 8층: 흰색화 되돌린 값
            for i in range(tng5.SEG_BYTES):
                kinds[b0 + i] = "body"
            kinds[b0 + 18] = kinds[b0 + 19] = "crc"
            if kk == 0:
                kinds[b0] = kinds[b0 + 1] = "len"
                row["len_field"] = int.from_bytes(data[:2], "big")
            row["crc_rx"] = row["crc_calc"] = crc
            cw = conv_encode(np.concatenate([bits, np.zeros(tng5.TAIL, np.int64)]), tng5.RATE_N)
            rec[kk * NCW:(kk + 1) * NCW] = cw
            known_c[kk * NCW:(kk + 1) * NCW] = True
            body = data[2:] if kk == 0 else data
            row["text"] = tng5.unpack(body)
            # rev 4: 글자마다 charset1 부호 (가변 길이) — (바이트 위치, 글자, 부호 비트 수)
            from tngpkt import charset1 as CS
            bit = 0
            for ch in row["text"]:
                groups.append((b0 + (2 if kk == 0 else 0) + bit // 8, ch, len(CS.CODE[ch])))
                bit += len(CS.CODE[ch])
        kinds[b0 + 20] = "pad"                                           # 꼬리 6비트 칸 (바이트 없음)
        rows.append(row)
    hard = (llr_comb > 0).astype(np.int64)
    fixed_coded = np.nonzero(known_c & (hard != rec))[0]
    fixed_info = np.unique(c2i[fixed_coded])
    err_tx = np.zeros(ntx, bool)
    known_tx = np.zeros(ntx, bool)
    kc = np.nonzero(known_c)[0]
    if len(kc):
        tx_k = c2tx[kc]                                                   # (m, 4)
        bad = (llr_tx[tx_k] > 0).astype(np.int64) != rec[kc][:, None]
        err_tx[tx_k[bad]] = True
        known_tx[tx_k.reshape(-1)] = True
    frame_fixed = err_tx.reshape(n, BPF).sum(1)
    frame_known = known_tx.reshape(n, BPF).sum(1)
    n_ok = sum(r["ok"] for r in rows)
    text = res.text or info.get("partial") or ""
    cf = info.get("confirm") or {}
    a = m["a"]
    t_d0, t_d1 = float(frame_t[0]), float(frame_t[-1] + L / fsb)
    d = {
        "tng5": True, "v1": v1, "org": org, "b_slot": b_slot, "t_data": (t_d0, t_d1), "t_end": float(T(end)),
        "audio": audio, "base": base, "segs": sl, "n": n, "S": S, "t_bit": t_bit, "llr_tx": llr_tx,
        "frame_of_tx": f_of, "bit_of_tx": b_of, "frame_t": frame_t, "llr_blocks": blocks, "frame_s": L / fsb,
        "format": "스트리밍", "llr_coded": llr_comb, "tx_of_coded": c2tx, "c2i": c2i, "llr_rep": llr_rep,
        "info_bits": info_bits, "n_info": nseg * IB, "fixed_coded": fixed_coded, "fixed_info": fixed_info,
        "err_tx": err_tx, "frame_fixed": frame_fixed, "frame_known": frame_known,
        "bytes": byts.astype(np.uint8), "byte_kinds": kinds, "segments": rows, "text": text, "seg_nbytes": NB,
        "groups5": groups,
        "charset": "charset1",
        "crc": {"ok": n_ok == nseg, "rx": None, "calc": None, "reason": "구간 {}/{} 통과".format(n_ok, nseg)},
        "drift_ppm": None, "relocks": 0, "used": nseg, "known_coded": int(known_c.sum()),
        "mids": mids, "confirm": cf, "mid_checks": info.get("mid_checks"),
        "summary": {"ok": bool(res.ok), "snr": float(res.snr_db), "df": df, "rho_a": float(a.get("rho") or np.nan),
                    "rho_b": float(b.get("rho", np.nan)) if (b and b.get("rho") is not None) else None,
                    "mode": res.mode, "frames": n, "coded_bits": ntx, "info_bits": nseg * IB,
                    "mean_abs_llr": float(np.mean(np.abs(llr_tx))), "fixed": int(len(fixed_coded)),
                    "nbytes": nseg * NB, "plen": nchars, "segments_ok": n_ok, "segments_used": nseg,
                    "known_coded": int(known_c.sum())},
    }
    return d
