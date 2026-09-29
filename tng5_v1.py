"""
[v1 보관본 — WAV 열기에서 v1 파일 복조 전용 (2026-09-25 v2 적용 때 복사, 실시간 수신에는 안 씀)]
TNG5 프로토콜 v1 (강한 모드) — 엔진. TNG44 코드는 건드리지 않는다.

송신  [희생 톤 300ms][TNG5 코스타스 A × 4]  (+6 dB, 포락선 일정 → PEP 안 늘어남)
      [가드][데이터 프레임 …][가드]            (models/stage3_strong.pt, 프레임 96비트 = 192 ms)
      [TNG5 코스타스 B × 4]                   (+6 dB)
데이터
  · 문자 40자: A–Z 0–9 공백 / ? -   (소문자 → 대문자, 집합 밖 문자는 송신 전 경고 후 제외)
  · 3글자 → 16비트 (base-40, 40³ = 64000 < 65536). 남는 코드 64000–65535 는 예약 (추가 기호용)
  · 구간 = 18바이트 (9 묶음 = 27자) + CRC16. 첫 구간 앞 2바이트 = 글자 수 (길이 필드) → 첫 구간 24자
  · 부호: 구간마다 K=7 R1/4 (생성 0o117 0o127 0o155 0o171, 꼬리 6) → 4번 반복 = R1/16
    구간 부호어 (160+6)×16 = 2656 비트 = 13.8 프레임 = 5.31 s
  · 인터리버: ILV_SPAN 로 선택 (측정: tng5_meas.py → results/tng5_meas.log)
수신
  · 톤 + A×4 + 가드 묶음 템플릿 (조각별 동기 상관, 조각 상한 두고 비동기 합산, 문턱 = 시간당 오경보 1회 꼬리)
  · 봉우리 모양 검사: 코스타스 조각만으로 ±1·±2 심볼 이동 최대 / 봉우리 > 0.5 → 거부
  · 끝 B × 4 (+ 앞 가드). 길이 필드를 알면 예상 끝 ±0.5 s 밖 B 무시.
"""
import math
import numpy as np
import torch

from config import PreambleConfig
from preamble import cpfsk, _bandlimit_full, _samples
from framing import crc16_ccitt
from conv_sim import encode as conv_encode, viterbi
import dataclasses

NAME, VERSION = "TNG5", "1"
MODEL = "models/stage3_strong.pt"
CHARSET = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 /?-"
assert len(CHARSET) == 40
COSTAS_A = (0, 4, 6, 1, 2, 5, 3)
COSTAS_B = (2, 1, 4, 5, 3, 0, 6)
COSTAS_RESERVED_TNG1 = ((3, 6, 1, 2, 0, 5, 4), (4, 1, 5, 6, 0, 3, 2))
SEG_BYTES = 18
LEN_BYTES = 2
CRC_BYTES = 2
REP = 4
RATE_N = 4                         # R1/4
TAIL = 6
SEG_INFO = 8 * (SEG_BYTES + CRC_BYTES)
SEG_CODED = (SEG_INFO + TAIL) * RATE_N * REP          # 2656
BITS_PER_FRAME = 96
SYNC_REP = 4
BOOST_DB = 6.0
ILV_SPAN = 4800                    # 구간 안 무작위 + 지연 인터리버 최대 약 9.6 s (측정: tng5_meas.py ilv)
FS = 2000.0


# ------------------------------------------------------------------ 문자
def normalize(text):
    """→ (보낼 문자열, 집합 밖 문자 목록)"""
    t = text.upper().replace("\n", " ")
    bad = sorted({c for c in t if c not in CHARSET})
    return "".join(c for c in t if c in CHARSET), bad


def pack(text):
    t = text + " " * (-len(text) % 3)
    out = bytearray()
    for i in range(0, len(t), 3):
        v = CHARSET.index(t[i]) * 1600 + CHARSET.index(t[i + 1]) * 40 + CHARSET.index(t[i + 2])
        out += v.to_bytes(2, "big")
    return bytes(out)


def unpack(b, n=None):
    s = []
    for i in range(0, len(b) - 1, 2):
        v = int.from_bytes(b[i:i + 2], "big")
        if v >= 64000:
            s.append("□□□")
            continue
        s.append(CHARSET[v // 1600] + CHARSET[(v // 40) % 40] + CHARSET[v % 40])
    t = "".join(s)
    return t[:n] if n is not None else t


# ------------------------------------------------------------------ 구간 · 부호
def segments(text):
    """→ 구간 바이트 목록 (각 18바이트, 첫 구간 앞 2바이트 = 글자 수). 마지막 구간은 공백 묶음으로 채움"""
    raw = len(text).to_bytes(LEN_BYTES, "big") + pack(text)
    pad = pack("   ")
    while len(raw) % SEG_BYTES:
        raw += pad[:min(2, SEG_BYTES - len(raw) % SEG_BYTES)]
    return [raw[i:i + SEG_BYTES] for i in range(0, len(raw), SEG_BYTES)]


def seg_bits(sb):
    c = crc16_ccitt(sb)
    return np.unpackbits(np.frombuffer(sb + bytes([c >> 8, c & 0xFF]), np.uint8)).astype(np.int64)


def seg_codeword(sb):
    c = conv_encode(np.concatenate([seg_bits(sb), np.zeros(TAIL, np.int64)]), RATE_N)
    return np.tile(c, REP)


_PERM = np.random.default_rng(55).permutation(SEG_CODED)
_INV = np.argsort(_PERM)


def interleave_map(n, span=None):
    """부호 비트 p → 송신 위치 q[p] (길이 n)"""
    span = ILV_SPAN if span is None else span
    nseg = n // SEG_CODED
    if span == "seg":
        return np.concatenate([_PERM + k * SEG_CODED for k in range(nseg)])
    J = 16
    Mm = max(1, int(span) // (J * J))
    p = np.arange(n)
    tau = p + (p % J) * J * Mm
    order = np.lexsort((p, tau))
    q = np.empty(n, np.int64)
    q[order] = np.arange(n)
    # 구간 안 무작위도 함께 (반복 사본이 붙어 있지 않게)
    base = interleave_map(n, "seg")
    return q[base]


def encode_bits(text, span=None):
    segs = segments(text)
    c = np.concatenate([seg_codeword(s) for s in segs])
    q = interleave_map(len(c), span)
    tx = np.empty_like(c)
    tx[q] = c
    nF = int(math.ceil(len(tx) / BITS_PER_FRAME))
    tx = np.concatenate([tx, np.random.default_rng(len(tx)).integers(0, 2, nF * BITS_PER_FRAME - len(tx))])
    return tx.reshape(nF, BITS_PER_FRAME), {"segments": segs, "n_coded": len(c), "n_frames": nF}


def decode_llr(llr_tx, nseg, span=None):
    """송신 순서 LLR (B, ≥ nseg*SEG_CODED) → (bytes 목록 (B, nseg), ok (B, nseg)). llr 양수 = 1"""
    n = nseg * SEG_CODED
    q = interleave_map(n, span)
    L = llr_tx[:, q]                                    # 부호 순서
    B = L.shape[0]
    out_b = [[None] * nseg for _ in range(B)]
    ok = np.zeros((B, nseg), bool)
    for k in range(nseg):
        c = L[:, k * SEG_CODED:(k + 1) * SEG_CODED].reshape(B, REP, -1).sum(1)
        bits = viterbi(c, RATE_N, SEG_INFO + TAIL)[:, :SEG_INFO]
        by = np.packbits(bits.astype(np.uint8), axis=1)
        for b in range(B):
            data, crc = bytes(by[b, :SEG_BYTES]), int.from_bytes(bytes(by[b, SEG_BYTES:]), "big")
            out_b[b][k] = data
            ok[b, k] = crc16_ccitt(data) == crc
    return out_b, ok


def assemble_text(segs_b, ok):
    """구간 바이트 → (글자 수, 텍스트). 실패 구간은 □ 27개 (첫 구간 실패면 길이 모름)"""
    n = int.from_bytes(segs_b[0][:LEN_BYTES], "big") if ok[0] else None
    parts = []
    for k, (b, o) in enumerate(zip(segs_b, ok)):
        body = b[LEN_BYTES:] if k == 0 else b
        parts.append(unpack(body) if o else "□" * (len(body) // 2 * 3))
    t = "".join(parts)
    return n, (t[:n] if n is not None else t.rstrip())


# ------------------------------------------------------------------ 동기 파형
# 방안 3: TNG44 (32 ms / 62.5 Hz, 희생 톤 0 Hz) 와 겹치지 않게 48 ms / 41.67 Hz (변조지수 2 같음, ±125 Hz),
# 희생 톤 -104.2 Hz (TNG44 톤 격자 0 · ±62.5 · ±125 · ±187.5 Hz 어디에도 없음), 가드 비트도 TNG5 전용.
PC = dataclasses.replace(PreambleConfig(), symbol_ms=48.0, tone_spacing_hz=125.0 / 3)
TONE_HZ = -2.5 * 125.0 / 3
GUARD5 = np.random.default_rng(5005).integers(0, 2, BITS_PER_FRAME).astype(np.float32)


def _costas_block(seq, lead=0):
    """연속 위상 FSK (preamble.cpfsk 와 같은 평활), 앞 희생 톤은 TONE_HZ"""
    ns = _samples(PC.symbol_ms, FS)
    mid = (PC.n_tones - 1) / 2.0
    f = [np.full(lead, TONE_HZ)] if lead else []
    for k in list(seq) * SYNC_REP:
        f.append(np.full(ns, (k - mid) * PC.tone_spacing_hz))
    f = np.concatenate(f)
    m = _samples(PC.freq_smooth_ms, FS)
    w = np.hanning(m + 2)[1:-1]
    w /= w.sum()
    f = np.convolve(np.concatenate([np.full(m, f[0]), f, np.full(m, f[-1])]), w, mode="same")[m:-m]
    return np.exp(1j * 2 * np.pi * np.cumsum(f) / FS)


def build_tx(data_bb):
    """data_bb: [가드][데이터][가드] 기저대역 (평균전력 1) → (전체 기저대역, 정보)"""
    g = 10 ** (BOOST_DB / 20)
    ntone = _samples(PC.tone_ms, FS)
    pre_raw = _costas_block(COSTAS_A, lead=ntone)
    r = np.ones(len(pre_raw))
    r[:40] = 0.5 * (1 - np.cos(np.pi * np.arange(40) / 40))
    pre_f, hd = _bandlimit_full(pre_raw * r, PC, FS)
    post_raw = _costas_block(COSTAS_B)
    e = np.ones(len(post_raw))
    e[-20:] = 0.5 * (1 + np.cos(np.pi * np.arange(1, 21) / 20))
    post_f, hd2 = _bandlimit_full(post_raw * e, PC, FS)
    n_pre, n_post = len(pre_raw), len(post_raw)
    # 동기 경계: LPF 꼬리를 자르지 않고 이웃 데이터 (가드 프레임) 에 겹쳐 더한다 (v1 은 잘라서 -60 dB 994 Hz).
    # 바깥 끝 (송신 처음 · 끝) 꼬리만 버린다 — 램프가 있어 거의 0. 위치 (pre · data0 · b_start) 는 그대로
    x = np.zeros(hd + n_pre + len(data_bb) + n_post + hd2, complex)
    x[0:len(pre_f)] += pre_f * g
    x[hd + n_pre:hd + n_pre + len(data_bb)] += data_bb
    b0 = hd + n_pre + len(data_bb)
    x[b0 - hd2:b0 - hd2 + len(post_f)] += post_f * g
    x = x[hd:len(x) - hd2]
    return x, {"pre": n_pre, "data": len(data_bb), "post": n_post, "ntone": ntone,
               "data0": n_pre + 384, "b_start": n_pre + len(data_bb)}


def templates(data_guard_bb):
    """수신 템플릿 조각: A 묶음 (톤 + A×4 + 가드) 과 B 묶음 (가드 + B×4). 조각 위치는 A = 송신 시작, B = B 시작 기준"""
    ntone = _samples(PC.tone_ms, FS)
    pre, hd = _bandlimit_full(_costas_block(COSTAS_A, lead=ntone), PC, FS)
    pre = pre[hd:len(pre) - hd]
    post, hd2 = _bandlimit_full(_costas_block(COSTAS_B), PC, FS)
    post = post[hd2:len(post) - hd2]
    n7 = 7 * _samples(PC.symbol_ms, FS)
    gz = data_guard_bb[16:384 - 16]
    A = [(ntone + i * n7, pre[ntone + i * n7: ntone + (i + 1) * n7], "costas") for i in range(SYNC_REP)]
    A += [(40, pre[40:ntone - 8], "tone"), (len(pre) + 16, gz, "guard")]
    Bt = [(i * n7, post[i * n7:(i + 1) * n7], "costas") for i in range(SYNC_REP)]
    Bt += [(-384 + 16, gz, "guard")]
    return A, Bt


def tx_waveform(text, model, cfg, dev, span=None):
    """텍스트 → (기저대역, 정보). 집합 밖 문자는 빠진다 (호출 쪽이 normalize 로 먼저 경고)"""
    from modem5 import _frames_to_baseband
    t, bad = normalize(text)
    fr, info = encode_bits(t, span)
    data = _frames_to_baseband(fr.astype(np.float32), GUARD5, model, cfg, dev)
    x, i2 = build_tx(data)
    info.update(i2)
    info.update({"text": t, "bad": bad})
    return x, info, data


# ------------------------------------------------------------------ 검출 (torch: GPU 또는 CPU)
NFFT = 1024
STEP = 4
FREQ = np.fft.fftfreq(NFFT, 1 / FS)
BAND = np.where(np.abs(FREQ) <= 100.0)[0]
CELLS_PER_HOUR = 500 * 3600 * len(BAND)
RIDGE_MAX = 0.5
COSTAS_SHARE = 0.5                 # 인정 조건: 코스타스 조각 상한 합 ≥ 전체 문턱 × 이 값 (tng5_cross.py 측정)
DEDUP_S = 2.0                      # 한 송신 안 A 1건: 이 시간 안에서는 가장 센 봉우리만
RHO_FRAC = 0.10                    # SNR 일치 검사: 코스타스 평균 ρ² ≥ 이 값 × 기대 ρ² (s/(1+s), s = 창 전력 / 잡음 바닥 − 1)
THR_SCALE = 1.0                    # 방안 3 뒤 원래대로 (1.1 은 방안 3 전 임시)


def threshold(K):
    from scipy.stats import gamma
    return float(gamma.isf(1.0 / CELLS_PER_HOUR, K))


@torch.no_grad()
def metric(y, chunks, starts, dev, only=None):
    """y (B, Ly) complex torch, starts (T,) → (상한 합 (B,T,F), 상한 없는 합)"""
    B = y.shape[0]
    use = [c for c in chunks if only is None or c[2] in only]
    K = len(use)
    T_ = threshold(len(chunks))
    cap = 2 * T_ / len(chunks)
    cs = torch.cat([torch.zeros(B, 1, device=dev), torch.cumsum(y.abs() ** 2, 1)], 1)
    tot = raw = totc = None
    st = torch.as_tensor(starts, device=dev)
    bi = torch.as_tensor(BAND, device=dev)
    for off, tmpl, kind in use:
        n = len(tmpl)
        tt = torch.as_tensor(np.conj(tmpl).astype(np.complex64), device=dev)
        es = float(np.sum(np.abs(tmpl) ** 2))
        a = (st + off).clamp(n, y.shape[1] - 2 * n - 1)

        def we(p):
            return cs[:, p + n] - cs[:, p]
        den = torch.maximum(torch.maximum(we(a), we(a - n)), we(a + n)) * es
        out = []
        for i in range(0, len(a), 256):
            idx = a[i:i + 256, None] + torch.arange(n, device=dev)[None]
            Z = torch.fft.fft(y[:, idx] * tt[None, None], n=NFFT, dim=2)[..., bi]
            out.append(Z.real ** 2 + Z.imag ** 2)
        g = torch.cat(out, 1) / den[..., None].clamp(min=1e-20) * n
        raw = g if raw is None else raw + g
        g = g.clamp(max=cap)
        tot = g if tot is None else tot + g
        if kind == "costas":
            totc = g if totc is None else totc + g
    metric.costas = totc                 # 코스타스 조각만의 상한 합 (톤 · 가드만으로 문턱 넘는 것 방지)
    return tot, raw


@torch.no_grad()
def ridge(y, chunks, pos, fi, dev):
    """코스타스 조각만: ±1·±2 심볼 이동의 최대 (±160Hz) / 봉우리. y (1, Ly)"""
    sym = _samples(PC.symbol_ms, FS)
    st = np.array([pos, pos - 2 * sym, pos - sym, pos + sym, pos + 2 * sym])
    _, r = metric(y, chunks, st, dev, only=("costas",))
    r = r[0].cpu().numpy()
    peak = r[0, fi]
    f = FREQ[BAND]
    near = np.abs(f - f[fi]) <= 160
    return float(r[1:, near].max() / max(peak, 1e-9))


def noise_floor(y):
    """버퍼의 50 ms 블록 전력 하위 5% (잡음 바닥 추정). 실시간은 긴 링버퍼에서 부른다"""
    n = 100
    L = (y.shape[1] // n) * n
    p = (y[0, :L].abs() ** 2).reshape(-1, n).mean(1)
    # 잡음만 있는 100 샘플 블록 전력의 5% 분위 ≈ 평균의 0.84 (Gamma(100)/100) → 평균으로 보정
    return float(torch.quantile(p, 0.05)) / 0.84 if len(p) > 4 else float(p.mean())


@torch.no_grad()
def snr_consistent(y, chunks, p, fi, floor, dev):
    """참 동기라면 그 자리 신호 세기에 맞는 상관이 나와야 한다 (강한 다른 신호의 부분 상관 거르기)"""
    cs = [c for c in chunks if c[2] == "costas"]
    rho, pw = [], []
    for c in cs:
        _, r = metric(y, [c], np.array([p]), dev)
        rho.append(float(r[0, 0, fi]) / len(c[1]))
        a = int(p + c[0])
        pw.append(float((y[0, max(a, 0):a + len(c[1])].abs() ** 2).mean()))
    s_hat = max(np.mean(pw) / max(floor, 1e-20) - 1.0, 0.0)
    return float(np.mean(rho)) >= RHO_FRAC * s_hat / (1.0 + s_hat), float(np.mean(rho)), s_hat


@torch.no_grad()
def detect(y, chunks, dev, starts=None, ridge_check=True, thr_scale=1.0, floor=None):
    """y (1, Ly) → 검출 목록 [{'start','df','metric'}] (시간순, 1 s 안에서는 가장 센 것)"""
    K = len(chunks)
    T_ = threshold(K) * thr_scale * THR_SCALE
    span = max(c[0] + len(c[1]) for c in chunks)
    lo = -min(0, min(c[0] for c in chunks))
    if starts is None:
        starts = np.arange(lo, y.shape[1] - span, STEP)
    found = []
    for i in range(0, len(starts), 4096):
        s = starts[i:i + 4096]
        m, mr = metric(y, chunks, s, dev)
        mc = metric.costas[0]
        mb, fb = m[0].max(1)
        ok = (mb >= T_) & (mc.gather(1, fb[:, None])[:, 0] >= T_ * COSTAS_SHARE)
        for j in np.where(ok.cpu().numpy())[0]:
            found.append((float(mr[0, j].max()), int(s[j]), float(mb[j])))
    found.sort(reverse=True)
    taken, rejected = [], []
    if floor is None:
        floor = noise_floor(y)
    dd = int(DEDUP_S * FS)
    for rawv, p, v in found:
        if any(abs(p - q["start"]) < dd for q in taken) or any(abs(p - q) < 500 for q in rejected):
            continue
        m, mr = metric(y, chunks, np.array([p]), dev)
        fi = int(mr[0, 0].argmax())
        rd = ridge(y, chunks, p, fi, dev) if ridge_check else 0.0
        if rd > RIDGE_MAX:
            rejected.append(p)
            continue
        okc, rho_m, s_hat = snr_consistent(y, chunks, p, fi, floor, dev)
        if not okc:
            rejected.append(p)
            continue
        taken.append({"start": p, "df": float(FREQ[BAND][fi]), "metric": v, "raw": rawv, "ridge": rd})
    return sorted(taken, key=lambda d: d["start"])


# ------------------------------------------------------------------ 모드 판별 (TNG44 ↔ TNG5)
@torch.no_grad()
def rho_near(y, tmpl, center, df, dev, span=480, fwin=4.0):
    """center ± span 샘플 (4 샘플 간격), 주파수 df ± fwin 안 최대 ρ² (한 조각 정규화 상관, 이웃 창 에너지 분모)"""
    st = np.arange(center - span, center + span + 1, STEP)
    st = st[(st >= len(tmpl)) & (st < y.shape[1] - 2 * len(tmpl) - 1)]
    if len(st) == 0:
        return 0.0
    _, r = metric(y, [(0, tmpl, "costas")], st, dev)
    f = FREQ[BAND]
    sel = np.abs(f - df) <= fwin
    return float(r[0][:, sel].max()) / len(tmpl)


def costas44(eng):
    return eng.det.tA, eng.det.tB


def tng5_chunks():
    """TNG5 코스타스 A · B 한 벌(7심볼) 템플릿 (부스트 없는 모양)"""
    n7 = 7 * _samples(PC.symbol_ms, FS)
    a, hd = _bandlimit_full(cpfsk(list(COSTAS_A), PC, FS), PC, FS)
    b, hd2 = _bandlimit_full(cpfsk(list(COSTAS_B), PC, FS), PC, FS)       # cpfsk 는 PC(48 ms · 41.67 Hz) 를 따른다
    return a[hd:hd + n7], b[hd2:hd2 + n7]


def which_mode(y, pos, df, t44, t5, dev):
    """검출 위치(코스타스 한 벌 시작) 근처에서 두 모드 배열의 ρ² 를 비교 → ('TNG44'|'TNG5', ρ²44, ρ²5)"""
    r44 = max(rho_near(y, t, pos, df, dev, span=48) for t in t44)
    r5 = max(rho_near(y, t, pos, df, dev, span=480) for t in t5)
    return ("TNG5" if r5 > r44 else "TNG44"), r44, r5


def suppress44(det44, det5_A, det5_B, info=None, margin=600):
    """TNG5 동기 구간(톤 + A×4 + 가드 / 가드 + B×4)에 걸친 TNG44 검출 제거 (방안 2). det5_*: detect() 결과"""
    nA = _samples(PC.tone_ms, FS) + SYNC_REP * 7 * _samples(PC.symbol_ms, FS) + 384
    nB = SYNC_REP * 7 * _samples(PC.symbol_ms, FS)
    zones = [(d["start"] - margin, d["start"] + nA + margin) for d in det5_A]
    zones += [(d["start"] - 384 - margin, d["start"] + nB + margin) for d in det5_B]
    return [d for d in det44 if not any(a <= d["start"] <= b for a, b in zones)]


def resolve_ab(det_A, det_B):
    """A 템플릿이 B×4 에 걸친 검출 (또는 그 반대) 정리: 코스타스 영역이 겹치면 점수 큰 쪽만 남긴다"""
    ca = _samples(PC.tone_ms, FS)
    span = SYNC_REP * 7 * _samples(PC.symbol_ms, FS)
    A, B = list(det_A), list(det_B)
    for a in list(A):
        for b in list(B):
            if abs((a["start"] + ca) - b["start"]) < span:
                if a.get("raw", a["metric"]) >= b.get("raw", b["metric"]):
                    B.remove(b)
                else:
                    A.remove(a)
                    break
    return A, B
