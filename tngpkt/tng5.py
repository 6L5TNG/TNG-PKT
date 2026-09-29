"""
TNG5 프로토콜 v2 (강한 모드) — 엔진. TNG44 코드는 건드리지 않는다.

v2 (2026-09-25, NA_g) — 반복 없는 동기. 데이터 부분 (문자 · 구간 · 부호 · 인터리버 · 모델) 은 v1 과 같다.
송신  [희생 톤 300ms][Welch 12차 코스타스 1개 (64 ms · 31.25 Hz, 0.768 s)]
      [가드][데이터 20프레임][M1][데이터 20프레임][M2] … [가드]   M_k = 서로 다른 7차 코스타스 4종 ((k-1) mod 4), 48 ms · 41.67 Hz
      [끝 동기 B12 ×1 (12차 코스타스 32 ms · 31.25 Hz, 0.384 s) — rev 3 (09-26). rev 2 는 B7 (0.336 s, WAV 열기만)]
      모든 동기 GFSK (BT 0.5) · +6 dB, 송신 전체에 LPF 한 번
수신  시작 = Welch12 (4심볼 × 3조각) → 첫 중간 동기 M1 확인 (위치 ±8 샘플 · ±4 Hz · SNR 일치) 뒤 확정 (tng5_app)
v1 (A×4 · B×4) 은 tng5_v1.py 에 보관 (WAV 열기 전용)

--- 아래는 v1 설명 (데이터 부분은 v2 도 같음) ---

송신  [희생 톤 300ms][TNG5 코스타스 A × 4]  (+6 dB, 포락선 일정 → PEP 안 늘어남)
      [가드][데이터 프레임 …][가드]            (models/stage3_strong.pt, 프레임 96비트 = 192 ms)
      [TNG5 코스타스 B × 4]                   (+6 dB)
데이터
  · 문자 (0.4.0 / wire rev 4): TNG1 과 같은 charset1 57자 (줄바꿈 · 기호 포함, 정적 Huffman), 소문자 → 대문자, 집합 밖은 송신 전 경고 후 제외
  · 구간마다 글자 단위로 채움 (구간 경계 = 글자 경계), 마지막 구간 EOT, 첫 구간 2바이트 = 구간 수. 0.3.x (rev 3) 는 base-40 40자
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

from tngpkt.config import PreambleConfig
from tngpkt.preamble import cpfsk, _bandlimit_full, _samples
from tngpkt.framing import crc16_ccitt
from tngpkt.conv_sim import encode as conv_encode, viterbi
import dataclasses

NAME, VERSION = "TNG5", "3.1"
MODEL = "models/stage3_strong.pt"
from tngpkt import charset1 as CS
CHARSET = CS.CHARS                 # 0.4.0 / wire rev 4: TNG1 과 같은 charset1 (57자, 줄바꿈 · 기호 포함)
assert len(CHARSET) == 57
COSTAS_A = (0, 4, 6, 1, 2, 5, 3)
COSTAS_B = (9, 8, 6, 0, 5, 1, 11, 3, 4, 7, 2, 10)   # rev 3 끝 동기 B12 (32 ms · 31.25 Hz, 0.384 s). 비교: results/sync_cmp (B-M ρ² 0.17)
SYMB_MS, SPB_HZ = 32.0, 31.25
COSTAS_B_V2 = (2, 1, 4, 5, 3, 0, 6)              # rev 2 끝 동기 B7 (WAV 열기 전용)
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
SYNC_REP = 4                       # v1 (A×4 · B×4) — 참고용
WELCH12 = tuple(pow(2, i, 13) - 1 for i in range(1, 13))       # v2 시작 동기 (자기 부엽 1칸)
SYM12_MS, SP12_HZ = 64.0, 31.25
COSTAS_MID = ((1, 4, 3, 5, 2, 0, 6), (2, 0, 6, 5, 1, 3, 4), (2, 1, 6, 4, 0, 3, 5), (2, 3, 5, 0, 4, 1, 6))   # M_k = [(k-1) % 4]
MID_EVERY = 20                     # 데이터 프레임 20개 (3.84 s) 마다 중간 동기 (프레임 20k 앞, 20k < 프레임 수)
GFSK_BT = 0.5
BOOST_DB = 2.25                    # 3.1 (09-26): 6.0 → 2.25 — 데이터 PAPR 을 낮춘 뒤 동기 PEP 가 데이터 PEP 를 넘지 않게
PAPR_CLIP_DB, PAPR_ITERS = 3.0, 3  # 3.1: 데이터 구간 포락선 클리핑 (데이터 평균 전력 + 3 dB) → 송신 LPF, 3회 (results/papr, exp_papr.py L3i3). None = 끔
ILV_SPAN = 4800                    # 구간 안 무작위 + 지연 인터리버 최대 약 9.6 s (측정: tng5_meas.py ilv)
FS = 2000.0


# ------------------------------------------------------------------ 문자 (0.4.0 / wire rev 4: charset1 공유)
# 문자표 = TNG1 과 같은 charset1 모듈 (57자 + EOT, 정적 Huffman). 구간마다 들어가는 만큼 '글자 단위' 로 채우고 나머지는 1 로 채움
# → 구간 경계 = 글자 경계 (앞 구간을 놓쳐도 뒤 구간이 따로 풀림, 중간 합류 · □ 처리 그대로). 마지막 구간은 EOT (긴 채움이
# 1…1 최장 부호가 되지 않게). 첫 구간 앞 2바이트 = 구간 수 (가변 길이라 글자 수로는 구간 수를 모름). 비교: exp_tng5_charset.py
def normalize(text):
    """→ (보낼 문자열, 집합 밖 문자 목록). 소문자 → 대문자, \r\n → \n, 탭 → 공백 (charset1 그대로)"""
    return CS.normalize(text)


def seg_cap_bits(k):
    """구간 k 의 글자 칸 (비트): 첫 구간은 구간 수 필드 2바이트를 뺌"""
    return 8 * (SEG_BYTES - (LEN_BYTES if k == 0 else 0))


def split(text):
    """→ [(구간 글자 비트, 구간 글자열)] — 구간마다 들어가는 만큼 글자 (마지막 구간 EOT)"""
    out, i, k = [], 0, 0
    while True:
        b, n, _ = CS.pack_block(text[i:], seg_cap_bits(k), end=True)
        if n == 0 and i < len(text):
            raise ValueError("구간에 글자가 하나도 안 들어감")
        out.append((b, text[i:i + n]))
        i += n
        k += 1
        if i >= len(text):
            return out


def unpack(body):
    """구간 글자 칸 바이트 → 글자열 (EOT · 끝 채움에서 멈춤)"""
    return CS.unpack_block(np.unpackbits(np.frombuffer(bytes(body), np.uint8)))[0]


def seg_count_field(v):
    """첫 구간 앞 2바이트 값 → 구간 수"""
    return int(v)


# ------------------------------------------------------------------ 구간 · 부호
def segments(text):
    """→ 구간 바이트 목록 (각 18바이트, 첫 구간 앞 2바이트 = 구간 수)"""
    sp = split(text)
    out = []
    for k, (b, _) in enumerate(sp):
        by = np.packbits(np.asarray(b, np.uint8)).tobytes()
        out.append((len(sp).to_bytes(LEN_BYTES, "big") + by) if k == 0 else by)
    return out


def seg_texts(segs):
    """구간 바이트 목록 → 구간별 글자열 (송신 표시용)"""
    return [unpack(s[LEN_BYTES:] if k == 0 else s) for k, s in enumerate(segs)]


# rev 4: 구간 바이트 흰색화 (고정 의사 난수 XOR). 가변 길이 채움 (EOT 뒤 1…1) 이 긴 같은 비트열이 되면
# 같은 PEP 50% 지점이 0.5 dB 나빠졌음 (exp_papr rec 120회, 21부) → 보내는 비트를 고르게. CRC 는 흰색화 전 바이트 기준
WHITEN = np.random.default_rng(4040).integers(0, 256, SEG_BYTES).astype(np.uint8).tobytes()


def whiten(b):
    """흰색화 · 되돌리기 (같은 연산)"""
    return bytes(x ^ y for x, y in zip(bytes(b), WHITEN))


def seg_bits(sb):
    c = crc16_ccitt(sb)
    sb = whiten(sb)
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
            data, crc = whiten(by[b, :SEG_BYTES]), int.from_bytes(bytes(by[b, SEG_BYTES:]), "big")
            out_b[b][k] = data
            ok[b, k] = crc16_ccitt(data) == crc
    return out_b, ok


def assemble_text(segs_b, ok):
    """구간 바이트 → (구간 수 필드, 텍스트). 실패 구간은 □ 하나 (첫 구간 실패면 구간 수 모름)"""
    n = int.from_bytes(segs_b[0][:LEN_BYTES], "big") if ok[0] else None
    parts = []
    for k, (b, o) in enumerate(zip(segs_b, ok)):
        body = b[LEN_BYTES:] if k == 0 else b
        parts.append(unpack(body) if o else "□")
    return n, "".join(parts)


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


def _gfsk(seq, sym_ms=None, spacing=None, bt=GFSK_BT):
    """v2 동기: 연속 위상 + 가우시안 주파수 성형 (exp_sync_v3.gfsk 와 같음)"""
    sym_ms = PC.symbol_ms if sym_ms is None else sym_ms
    spacing = PC.tone_spacing_hz if spacing is None else spacing
    ns = _samples(sym_ms, FS)
    mid = (max(seq) + min(seq)) / 2.0
    f = np.concatenate([np.full(ns, (k - mid) * spacing) for k in seq])
    sig = np.sqrt(np.log(2)) / (2 * np.pi * bt) * ns
    m = int(3 * sig)
    w = np.exp(-0.5 * (np.arange(-m, m + 1) / sig) ** 2)
    w /= w.sum()
    f = np.convolve(np.concatenate([np.full(m, f[0]), f, np.full(m, f[-1])]), w, mode="same")[m:-m]
    return np.exp(1j * 2 * np.pi * np.cumsum(f) / FS)


def _lpf_only(x):
    y, hd = _bandlimit_full(x, PC, FS)
    return y[hd:len(y) - hd]


def _tone(n):
    return np.exp(1j * 2 * np.pi * TONE_HZ * np.arange(n) / FS)


N7 = 7 * _samples(PC.symbol_ms, FS)          # 7차 코스타스 한 벌 (336 ms = 672 샘플)
N12 = 12 * _samples(SYM12_MS, FS)            # Welch12 (768 ms = 1536 샘플)
NM = N7                                        # 중간 동기 길이
NB = 12 * _samples(SYMB_MS, FS)                # rev 3 끝 동기 B12 (384 ms = 768 샘플)
FL = BITS_PER_FRAME * 4                        # 프레임 384 샘플 (192 ms)


def mid_seq(k):
    return COSTAS_MID[(k - 1) % 4]


def mids_before(f):
    """프레임 f (0 = 첫 데이터 프레임) 앞에 들어간 중간 동기 수"""
    return np.maximum(np.asarray(f), 0) // MID_EVERY


def stream_to_tx(j):
    """데이터 스트림 번호 j (첫 데이터 프레임 시작 = 0, 중간 동기 뺀 기준) → 송신 번호 (같은 기준, 중간 동기 포함)"""
    j = np.asarray(j)
    return j + NM * (np.maximum(j, 0) // (MID_EVERY * FL))


def frame_off(f):
    """프레임 f 시작의 송신 위치 (첫 데이터 프레임 기준, 샘플)"""
    return np.asarray(f) * FL + NM * mids_before(f)


def mid_off(k):
    """중간 동기 k (1, 2, …) 시작의 송신 위치 (첫 데이터 프레임 기준)"""
    return k * MID_EVERY * FL + NM * (k - 1)


def n_mids(nF):
    return max(0, (int(nF) - 1) // MID_EVERY)


def b_off(nF):
    """코스타스 B 시작 (첫 데이터 프레임 기준): 프레임 nF 개 + 끝 가드 + 그 사이 중간 동기"""
    return (nF + 1) * FL + NM * n_mids(nF)


def nF_from_b(delta):
    """B 시작 − 첫 데이터 프레임 (샘플) → 프레임 수"""
    best = None
    for n in range(1, 2000):
        e = abs(b_off(n) - delta)
        if best is None or e < best[0]:
            best = (e, n)
        if b_off(n) > delta + FL:
            break
    return best[1]


def _head():
    ntone = _samples(PC.tone_ms, FS)
    return np.concatenate([_tone(ntone), _gfsk(WELCH12, SYM12_MS, SP12_HZ)]), ntone


def build_tx(data_bb):
    """data_bb: [가드][데이터 nF][가드] 기저대역 (평균전력 1) → (전체 기저대역, 정보).
    v2: 톤 + Welch12 | 데이터 (20프레임마다 M_k) | B×1, 동기 GFSK +6 dB, 송신 전체 LPF 1회"""
    g = 10 ** (BOOST_DB / 20)
    head, ntone = _head()
    pre = head * g
    pre[:40] *= 0.5 * (1 - np.cos(np.pi * np.arange(40) / 40))
    nF = len(data_bb) // FL - 2
    parts, prev, mids = [pre], 0, []
    pos = len(pre)
    for k in range(1, 10 ** 6):
        f = k * MID_EVERY
        if f >= nF:
            break
        c = FL + f * FL                                    # 스트림 번호 (앞 가드 포함)
        parts.append(data_bb[prev:c])
        pos += c - prev
        mids.append((k, pos))
        parts.append(_gfsk(mid_seq(k)) * g)
        pos += NM
        prev = c
    parts.append(data_bb[prev:])
    pos += len(data_bb) - prev
    b_start = pos
    post = _gfsk(COSTAS_B, SYMB_MS, SPB_HZ) * g
    post[-20:] *= 0.5 * (1 + np.cos(np.pi * np.arange(1, 21) / 20))
    parts.append(post)
    x = _lpf_only(np.concatenate(parts))
    if PAPR_CLIP_DB is not None and PAPR_ITERS:
        x = _papr_clip(x, len(pre), b_start, mids)
    return x, {"pre": len(pre), "data": b_start - len(pre), "post": len(post), "ntone": ntone,
               "data0": len(pre) + FL, "b_start": b_start, "mids": mids, "n_frames_tx": nF}


def _papr_clip(x, d_lo, d_hi, mids):
    """3.1: 데이터 구간 (중간 동기 제외) 포락선을 데이터 평균 전력 + PAPR_CLIP_DB 에서 자르고 송신 LPF, PAPR_ITERS 회.
    같은 PEP 에서 데이터 평균 출력 +2.9 dB, 신호 대 왜곡 18.9 dB, -60 dB 대역 418 Hz (측정: exp_papr.py). 수신기 변경 없음"""
    m = np.zeros(len(x), bool)
    m[d_lo:d_hi] = True
    for _, p in mids:
        m[p:p + NM] = False
    A = np.sqrt(np.mean(np.abs(x[m]) ** 2)) * 10 ** (PAPR_CLIP_DB / 20)
    for _ in range(PAPR_ITERS):
        e = np.abs(x)
        s = m & (e > A)
        x = x.copy()
        x[s] *= A / e[s]
        x = _lpf_only(x)
    return x


def templates(data_guard_bb):
    """수신 템플릿 조각: A = Welch12 (4심볼씩 3조각, 위치 = 송신 시작 기준), B = 가드 + B×1 (위치 = B 시작 기준)"""
    ntone = _samples(PC.tone_ms, FS)
    w = _lpf_only(_gfsk(WELCH12, SYM12_MS, SP12_HZ))
    n4 = 4 * _samples(SYM12_MS, FS)
    A = [(ntone + i * n4, w[i * n4:(i + 1) * n4], "costas") for i in range(3)]
    gz = data_guard_bb[16:FL - 16]
    Bt = [(0, _lpf_only(_gfsk(COSTAS_B, SYMB_MS, SPB_HZ)), "costas"), (-FL + 16, gz, "guard")]
    return A, Bt


def templates_b_v2(data_guard_bb):
    """rev 2 파일의 B7 + 가드 (WAV 열기 전용)"""
    gz = data_guard_bb[16:FL - 16]
    return [(0, _lpf_only(_gfsk(COSTAS_B_V2)), "costas"), (-FL + 16, gz, "guard")]


def mid_templates():
    """중간 동기 4종 템플릿 (순서 = COSTAS_MID, M_k 는 [(k-1) % 4])"""
    return [_lpf_only(_gfsk(m)) for m in COSTAS_MID]


def tx_waveform(text, model, cfg, dev, span=None):
    """텍스트 → (기저대역, 정보). 집합 밖 문자는 빠진다 (호출 쪽이 normalize 로 먼저 경고)"""
    from tngpkt.modem5 import _frames_to_baseband
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


GUARD_MIN = 0.05                   # B×1 + 가드: 센 B 에서 가드 ρ² ≥ 코스타스 ρ² × 이 값 (v2b 측정: 참 0.18~0.21, 데이터 속 가짜 0.03~0.09 일부)
STRUCT_MIN = 0.15


@torch.no_grad()
def _b_guard_ok(y, chunks, p, fi, dev):
    """B 묶음 (가드가 앞) 에서만: 센 코스타스인데 앞 가드가 안 맞으면 데이터 속 가짜 B"""
    if not any(c[2] == "guard" and c[0] < 0 for c in chunks):
        return True
    rc, rg = [], []
    for c in chunks:
        _, r = metric(y, [c], np.array([p]), dev)
        (rc if c[2] == "costas" else rg).append(float(r[0, 0, fi]) / len(c[1]))
    if not rc or max(rc) < STRUCT_MIN:
        return True
    return not (rg and max(rg) < GUARD_MIN * max(rc))


@torch.no_grad()
def detect(y, chunks, dev, starts=None, ridge_check=True, thr_scale=1.0, floor=None, dedup_s=None):
    """y (1, Ly) → 검출 목록 [{'start','df','metric'}] (시간순, DEDUP_S 안에서는 가장 센 것. dedup_s 로 좁히면 가까운 봉우리도 남김)"""
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
    dd = int((DEDUP_S if dedup_s is None else dedup_s) * FS)
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
        if not okc or not _b_guard_ok(y, chunks, p, fi, dev):
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
    """TNG5 v2 동기 템플릿 (Welch12 전체 · B 한 벌, 부스트 없는 모양) — 모드 간 공동 판정용"""
    return _lpf_only(_gfsk(WELCH12, SYM12_MS, SP12_HZ)), _lpf_only(_gfsk(COSTAS_B, SYMB_MS, SPB_HZ))


def which_mode(y, pos, df, t44, t5, dev):
    """검출 위치(코스타스 한 벌 시작) 근처에서 두 모드 배열의 ρ² 를 비교 → ('TNG44'|'TNG5', ρ²44, ρ²5)"""
    r44 = max(rho_near(y, t, pos, df, dev, span=48) for t in t44)
    r5 = max(rho_near(y, t, pos, df, dev, span=480) for t in t5)
    return ("TNG5" if r5 > r44 else "TNG44"), r44, r5


def suppress44(det44, det5_A, det5_B, info=None, margin=600):
    """TNG5 동기 구간(톤 + A×4 + 가드 / 가드 + B×4)에 걸친 TNG44 검출 제거 (방안 2). det5_*: detect() 결과"""
    nA = _samples(PC.tone_ms, FS) + N12 + 384
    nB = NB
    zones = [(d["start"] - margin, d["start"] + nA + margin) for d in det5_A]
    zones += [(d["start"] - 384 - margin, d["start"] + nB + margin) for d in det5_B]
    return [d for d in det44 if not any(a <= d["start"] <= b for a, b in zones)]


def resolve_ab(det_A, det_B):
    """A 템플릿이 B×4 에 걸친 검출 (또는 그 반대) 정리: 코스타스 영역이 겹치면 점수 큰 쪽만 남긴다"""
    ca = _samples(PC.tone_ms, FS)
    span = N12
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
