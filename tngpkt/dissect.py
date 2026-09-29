"""
패킷 해부 — 수신한 패킷 하나를 처리 단계별(층별)로 펼친다 (Wireshark 처럼). 분석 스레드에서 계산한다.

층 (가로축 = 톤 시작부터의 시간 [s], 모든 층 공통)
  1 오디오 파형 (수신 원본, 1ms 칸의 최소/최대)
  2 기저대역 I/Q (코스타스 주파수 오차 보정 + A 위상 기준)
  3 구간: 톤 / 코스타스 A / 가드 / 데이터 프레임 / 가드 / 코스타스 B
  4 수신 NN 출력 LLR (프레임 · 심볼마다 비트 2개, 심볼 중앙 시각에)
  5 디인터리빙 후 부호 비트 LLR        ┐ 인터리빙 뒤라 시간 순서가 없다 →
  6 비터비 출력 정보 비트 (고친 자리)   │ 데이터 구간 폭에 부호어 · 정보 비트 순서대로 펼친다.
  7 CRC (패킷 전체 CRC16 하나 +       │ 클릭하면 인터리버 치환을 따라
    프레임별 오류 비트 수)             │ 모든 층의 대응 위치를 찾는다
  8 바이트 [길이 2][본문][CRC 2][패딩] → 텍스트 ┘

엔진 함수는 읽기만 한다 (deinterleave · viterbi_decode · conv_encode · crc16 은 수신측과 같은 것).
링크 계층의 CRC 는 패킷 전체에 하나뿐이다 (프레임별 CRC 없음) — 프레임별로는 오류 비트 수를 보여 준다.
"""

import numpy as np

from tngpkt.modem import audio_to_baseband
from tngpkt.fec import deinterleave, interleaver, viterbi_decode, conv_encode
from tngpkt.framing import crc16_ccitt, bits_to_bytes
from tngpkt.link4 import info_bits_for, BITS_PER_FRAME, LEN_BYTES, CRC_BYTES
from tngpkt.viz import _phase_from_A


def dissect(res, cfg, pc, det):
    viz = getattr(res, "viz", None) or {}
    m = viz.get("msg") or {}
    info = m.get("info") or {}
    llr = info.get("llr")
    a = m.get("a")
    if llr is None or a is None or "frame0" not in info or "audio" not in viz:
        return None                                  # 옛 형식(코스타스 없음)은 구간 위치를 모른다
    fsb, L, sps, S = float(cfg.fs_base), cfg.frame_len, cfg.sps, cfg.symbols_per_frame
    n = int(llr.shape[0])
    a_start = int(a["start"])
    a_end = a_start + det.n
    data0 = a_end + int(info["frame0"]) - int(info.get("pad", 0))       # 첫 데이터 프레임 (기저대역 번호)
    tone_n = int(round(pc.tone_ms / 1000.0 * fsb))
    org = a_start - tone_n                                               # 시간 0 = 톤 시작
    b = m.get("b")
    b_start = int(b["start"]) if b else data0 + (n + 1) * L
    end = b_start + det.n
    T = lambda i: (np.asarray(i, dtype=np.float64) - org) / fsb          # 기저대역 번호 → 초

    # 1 오디오 (1ms 칸 최소/최대)
    x = np.asarray(viz["audio"], dtype=np.float64)
    fs = float(viz["fs"])
    k = fs / fsb
    i0, i1 = max(0, int(org * k)), min(len(x), int(end * k))
    blk = max(1, int(round(fs * 0.001)))
    nb = max(1, (i1 - i0) // blk)
    seg = x[i0:i0 + nb * blk].reshape(nb, blk)
    audio = {"t": (i0 + (np.arange(nb) + 0.5) * blk) / fs - org / fsb, "lo": seg.min(axis=1), "hi": seg.max(axis=1)}

    # 2 기저대역 I/Q (주파수 오차 보정 + A 위상)
    bb = audio_to_baseband(x, fs, cfg)
    df = float(m.get("df", 0.0))
    y = bb * np.exp(-2j * np.pi * df * np.arange(len(bb)) / fsb)
    ph = _phase_from_A(y, a_start, det)
    if ph is not None:
        y = y * np.exp(-1j * ph)
    lo, hi = max(org, 0), min(end, len(y))
    yb = y[lo:hi]
    d0, d1 = max(data0 - lo, 0), min(data0 + n * L - lo, len(yb))
    rms = np.sqrt(np.mean(np.abs(yb[d0:d1]) ** 2)) if d1 > d0 else 1.0
    base = {"t": T(np.arange(lo, hi)), "i": yb.real / max(rms, 1e-12), "q": yb.imag / max(rms, 1e-12)}

    # 3 구간
    segs = [("톤", org, a_start, "tone"), ("코스타스 A", a_start, a_end, "costas"),
            ("가드", data0 - L, data0, "guard")]
    segs += [("프레임 {}".format(f + 1), data0 + f * L, data0 + (f + 1) * L, "frame") for f in range(n)]
    segs += [("가드", data0 + n * L, data0 + (n + 1) * L, "guard"),
             ("코스타스 B" if b else "B 없음", b_start, end, "costas" if b else "missing")]
    segs = [(nm, float(T(s0)), float(T(s1)), kd) for nm, s0, s1, kd in segs]

    # 4 수신 NN LLR — 송신 순서 위치 p = 프레임*96 + 비트, 심볼 = 비트//2
    ntx = n * BITS_PER_FRAME
    p = np.arange(ntx)
    f_of, b_of = p // BITS_PER_FRAME, p % BITS_PER_FRAME
    t_bit = T(data0 + f_of * L + (b_of // 2) * sps + sps / 2.0)
    llr_tx = llr.reshape(-1).astype(np.float64)

    if info.get("format") == "스트리밍":                   # 7단계 새 형식: 구간별 CRC16 · 컨볼루션 인터리버
        from tngpkt.stream_ui import dissect_stream
        common = {"org": org, "t_data": (float(T(data0)), float(T(data0 + n * L))), "t_end": float(T(end)),
                  "audio": audio, "base": base, "segs": segs, "n": n, "S": S, "t_bit": t_bit, "llr_tx": llr_tx,
                  "frame_of_tx": f_of, "bit_of_tx": b_of,
                  "summary": {"ok": bool(res.ok), "snr": float(res.snr_db), "df": df,
                              "rho_a": float(a.get("rho", np.nan)),
                              "rho_b": float(b.get("rho", np.nan)) if b else None, "mode": res.mode,
                              "frames": n, "coded_bits": ntx, "mean_abs_llr": float(np.mean(np.abs(llr_tx)))}}
        return dissect_stream(res, cfg, pc, det, common)

    # 5 디인터리빙: 부호 비트 i 는 송신 위치 inv[i] 에 있다 (interleave: tx[j] = coded[perm[j]])
    perm = interleaver(ntx)
    tx_of_coded = np.empty(ntx, dtype=np.int64)
    tx_of_coded[perm] = np.arange(ntx)
    llr_coded = deinterleave(llr_tx)

    # 6 비터비 → 정보 비트, 다시 부호화해서 '고친' 부호 비트 자리
    n_info = info_bits_for(n)
    dec = np.asarray(viterbi_decode(llr_coded, n_info)).astype(np.int64)
    recoded = np.asarray(conv_encode(dec)).astype(np.int64)[:ntx]
    hard = (llr_coded > 0).astype(np.int64)
    fixed_coded = np.nonzero(hard != recoded)[0]
    fixed_info = np.unique(np.minimum(fixed_coded // 2, n_info - 1))
    err_tx = np.zeros(ntx, dtype=bool)
    err_tx[tx_of_coded[fixed_coded]] = True
    frame_fixed = err_tx.reshape(n, BITS_PER_FRAME).sum(axis=1)

    # 7 CRC (패킷 전체 하나)
    nbytes = n_info // 8
    data = bits_to_bytes(dec[:nbytes * 8])
    plen = int.from_bytes(data[:LEN_BYTES], "big")
    crc = {"ok": False, "rx": None, "calc": None, "plen": plen, "reason": ""}
    if plen + LEN_BYTES + CRC_BYTES <= len(data):
        body = data[:LEN_BYTES + plen]
        crc["rx"] = int.from_bytes(data[LEN_BYTES + plen:LEN_BYTES + plen + CRC_BYTES], "big")
        crc["calc"] = crc16_ccitt(body)
        crc["ok"] = crc["rx"] == crc["calc"]
        crc["reason"] = "일치" if crc["ok"] else "불일치"
    else:
        crc["reason"] = "길이 필드가 범위를 벗어남"

    # 8 바이트 종류
    kinds = []
    for i in range(nbytes):
        if i < LEN_BYTES:
            kinds.append("len")
        elif i < LEN_BYTES + plen:
            kinds.append("body")
        elif i < LEN_BYTES + plen + CRC_BYTES:
            kinds.append("crc")
        else:
            kinds.append("pad")
    try:
        text = data[LEN_BYTES:LEN_BYTES + plen].decode("utf-8") if crc["ok"] else ""
    except UnicodeDecodeError:
        text = ""

    return {
        "org": org, "t_data": (float(T(data0)), float(T(data0 + n * L))), "t_end": float(T(end)),
        "audio": audio, "base": base, "segs": segs,
        "n": n, "S": S, "t_bit": t_bit, "llr_tx": llr_tx, "frame_of_tx": f_of, "bit_of_tx": b_of,
        "llr_coded": llr_coded, "tx_of_coded": tx_of_coded,
        "info_bits": dec, "n_info": n_info, "fixed_coded": fixed_coded, "fixed_info": fixed_info,
        "frame_fixed": frame_fixed, "err_tx": err_tx,
        "format": "단일 블록",
        "bytes": np.frombuffer(data, dtype=np.uint8).copy(), "byte_kinds": kinds, "crc": crc, "text": text,
        "summary": {"ok": bool(res.ok), "snr": float(res.snr_db), "df": df,
                    "rho_a": float(a.get("rho", np.nan)), "rho_b": float(b.get("rho", np.nan)) if b else None,
                    "mode": res.mode, "frames": n, "coded_bits": ntx, "info_bits": n_info,
                    "mean_abs_llr": float(np.mean(np.abs(llr_tx))), "fixed": int(len(fixed_coded)),
                    "plen": plen, "nbytes": nbytes},
    }


# ---------------------------------------------------------------- 층 사이 대응 (클릭 강조용)
# TNG5 (dissect5): tx_of_coded 가 (부호 비트, 반복 사본 4) 2차원 · c2i = 부호 비트 → 정보 칸 (구간당 168칸 = 21바이트)
def _c2i(d, coded):
    coded = np.asarray(coded, dtype=np.int64)
    if "c2i" in d:
        return np.unique(d["c2i"][coded])
    return np.unique(np.minimum(coded // 2, d["n_info"] - 1))


def _i2c(d, info):
    if "c2i" in d:
        return np.nonzero(np.isin(d["c2i"], info))[0]
    coded = np.concatenate([2 * info, 2 * info + 1])
    return np.unique(coded[coded < len(d["tx_of_coded"])])


def _txs(d, coded):
    return np.unique(np.asarray(d["tx_of_coded"])[np.asarray(coded, dtype=np.int64)].reshape(-1))


def _bytes_of(d, info):
    """정보 비트 → 바이트 번호 (끝의 꼬리 · 남는 비트는 바이트가 없으니 뺀다)"""
    b = np.unique(np.asarray(info, dtype=np.int64) // 8)
    b = b[b < len(d["bytes"])]
    if "c2i" in d:                                   # TNG5: 꼬리 칸 (구간마다 21번째 바이트) 은 바이트 아님
        b = b[np.asarray([d["byte_kinds"][i] != "pad" for i in b], bool)] if len(b) else b
    return b


def links_from_tx(d, tx_positions):
    """송신 위치들 → {'tx','coded','info','byte'}"""
    tx = np.unique(np.asarray(tx_positions, dtype=np.int64))
    toc = np.asarray(d["tx_of_coded"])
    hit = np.isin(toc, tx)
    coded = np.nonzero(hit.any(axis=1) if hit.ndim == 2 else hit)[0]
    info = _c2i(d, coded)
    return {"tx": tx, "coded": coded, "info": info, "byte": _bytes_of(d, info)}


def links_from_coded(d, coded):
    coded = np.unique(np.asarray(coded, dtype=np.int64))
    info = _c2i(d, coded)
    return {"tx": _txs(d, coded), "coded": coded, "info": info, "byte": _bytes_of(d, info)}


def links_from_info(d, info):
    info = np.unique(np.asarray(info, dtype=np.int64))
    coded = _i2c(d, info)
    return {"tx": _txs(d, coded) if len(coded) else np.zeros(0, np.int64), "coded": np.unique(coded), "info": info,
            "byte": _bytes_of(d, info)}


def links_from_byte(d, byte):
    info = np.arange(int(byte) * 8, int(byte) * 8 + 8)
    info = info[info < d["n_info"]]
    r = links_from_info(d, info)
    r["byte"] = np.array([int(byte)])
    return r
