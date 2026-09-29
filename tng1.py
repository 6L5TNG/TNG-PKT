"""
TNG1 (강한+) 엔진 — 실험판. TNG44 · TNG5 코드와 무관, 앱 연결 없음.

복소 기저대역 FS = 1000 Hz (오디오 중심 1500 Hz). 송신 PEP = 1 (GFSK 는 포락선 일정 → 평균 = PEP).
블록 (반복, 무제한 스트리밍)
  [심볼 NSYM = NS 동기 + ND 데이터], 동기 위치 · 톤은 모든 블록이 같음 → 블록 주기로 접어 누적, 늦게 켜도 다음 블록부터 합류
  데이터: 문자 (charset1 가변) KI 비트 + CRC16 → 테일바이팅 R1/log2M 컨볼루션 (가지 출력 = 심볼 1개, 심볼 거리 d = K) → 블록 안 심볼 인터리버
수신 (오프라인 버퍼 방식, 실험용)
  1) 스펙트로그램 (심볼 길이 창, 홉 1/4 심볼, 반 빈 간격) → 동기 톤 대비 지표를 K 블록 접어 누적
  2) 후보 상위 NCAND (블록 주기 위상 · 주파수) → 블록마다 미세 탐색 (±1/2 홉, ±3 Hz) → 복조 → 비터비 → CRC16
  3) 표시 규칙 (CRC 전 표시 금지): 같은 격자에서 CRC 블록 2개, 또는 CRC 1개 + 접은 동기 지표 ≥ THR1
SNR (이 모듈 · 실험 공통): PEP ÷ (2500 Hz 대역 잡음), FT8 공표값과 같은 방식.
  snr_ref (TNG5 표기) 환산: SNR_PEP = SNR_TNG5표기 + 8.48 dB  (snr_ref.PEP_ANCHOR_TNG5)
"""
import math
import dataclasses
import numpy as np
import charset1

FS = 1000.0
CRC_BITS = 16


@dataclasses.dataclass(frozen=True)
class Cfg:
    name: str
    M: int            # 톤 수
    Ns: int           # 심볼 길이 (샘플 @1 kHz) → 톤 간격 1000/Ns Hz
    NS: int           # 블록당 동기 심볼
    KI: int           # 블록당 문자 비트 (CRC 제외). 데이터 심볼 ND = KI + CRC16 (심볼당 정보 1비트)
    Kc: int = 7       # 컨볼루션 구속장 (심볼 거리 d = Kc)
    lst: int = 1      # 리스트 비터비 크기 (1 = 보통 비터비)
    iters: int = 0    # CRC 실패 시 국소 SNR 가중 재복호 횟수
    bpsym: int = 1    # 심볼당 정보 비트 (1: R1/n 격자, Kc = 구속장 · 2: R2/n 격자, Kc = 상태 비트 m)
    bt: float = 2.0   # GFSK BT (심볼 시간 기준)
    seed: int = 11
    # --- 소리 변형 (09-26 2부). 기본값 = 채택안 그대로
    slots: tuple = None      # 톤 m 의 주파수 (1/T 단위, 모두 같은 소수부). None = m - M/2 + 0.5 (간격 1/T)
    chirp: float = 0.0       # 심볼마다 위로 미끄러지는 폭 [Hz] (모든 톤 같은 기울기 → 역처프 뒤 직교)
    sync_pos: tuple = None   # 동기 위치 · 톤 직접 지정 (None = 무작위 흩뿌림)
    sync_tone: tuple = None
    chirp_smooth: bool = False   # 처프 되돌림도 GFSK 로 부드럽게
    chirp_bt: float = 0.0        # 처프 되돌림만 BT chirp_bt 가우스로 (0 = 그대로)
    symnorm: bool = False    # 순간 잡음 대비: 심볼별 에너지 정규화
    first_flag: bool = False # wire rev 2: 첫 블록 CRC 시작값 표시 (TNG1 1.1)
    msd: int = 0             # 음 연결 복조: CRC 실패 시 1차 결정으로 ±msd 심볼 창 결맞음 상관 → 재복호 (0 = 끔)

    @property
    def slot(self):
        return np.asarray(self.slots if self.slots is not None else np.arange(self.M) - self.M / 2 + 0.5, float)

    @property
    def T(self):
        return self.Ns / FS

    @property
    def ND(self):
        return (self.KI + CRC_BITS) // self.bpsym

    @property
    def nsym(self):
        return self.NS + self.ND

    @property
    def L(self):
        return self.nsym * self.Ns

    @property
    def bps(self):
        return int(math.log2(self.M))

    @property
    def ncoded(self):
        return self.ND * self.bps

    @property
    def k(self):
        return self.KI + CRC_BITS

    @property
    def rate(self):
        return self.k / self.ncoded

    @property
    def block_s(self):
        return self.L / FS

    @property
    def cps(self):
        return self.KI / 4.84 / self.block_s          # 교신 예문 평균 4.84 비트/자 (charset1 측정)

    @property
    def bw(self):
        return (self.slot.max() - self.slot.min() + 1) / self.T + self.chirp

    def describe(self):
        return ("%s: %d-GFSK %.0f ms (%.2f Hz), 블록 %d+%d 심볼 = %.1f s, 문자 %d + CRC16, R=%.3f, 동기 %.1f%%, 약 %.2f 자/s, 톤 폭 %.0f Hz" %
                (self.name, self.M, self.T * 1e3, 1 / self.T, self.NS, self.ND, self.block_s, self.KI, self.rate,
                 100 * self.NS / self.nsym, self.cps, self.bw))


CFGS = {
    # 1 자/s 부근 (160 ms, 6.25 Hz)
    "A64": Cfg("A64", 64, 160, 12, 64),
    "A32": Cfg("A32", 32, 160, 12, 64),
    "A16": Cfg("A16", 16, 160, 12, 64),
    "A16L": Cfg("A16L", 16, 176, 12, 64),
    # TNG1 1.0 (wire rev 1, 09-26 확정): A16-K13 + chirp50s (음마다 50 Hz 위로 미끄러짐, 되돌림 BT 8), 음 연결 복조 W=2
    # 1.1 (wire rev 2, 09-26): 첫 블록 표시 (CRC16 시작값). 공중 형식은 첫 블록 CRC 만 다름
    "TNG1": Cfg("TNG1", 16, 160, 12, 64, Kc=13, chirp=50.0, chirp_bt=8.0, msd=2, first_flag=True),
    "TNG1r1": Cfg("TNG1r1", 16, 160, 12, 64, Kc=13, chirp=50.0, chirp_bt=8.0, msd=2),   # 1.0 / wire rev 1 (WAV 열기용)
    "TNG1base": Cfg("TNG1base", 16, 160, 12, 64, Kc=13),   # 비교용: 처프 없는 A16-K13 (1부 채택안, FT8 과 비슷한 소리)
    # 동기 줄이기 측정용 (블록 길이 14.7 s 그대로, 줄인 동기만큼 심볼을 길게): 적용 안 함
    "SR8": Cfg("SR8", 16, 168, 8, 64, Kc=13),
    "SR4": Cfg("SR4", 16, 175, 4, 64, Kc=13),     # 176 ms · 5.68 Hz · 91 Hz → 0.82 자/s (하한 0.8 근처)
    # 2 자/s (80 ms, 12.5 Hz)
    "B32": Cfg("B32", 32, 80, 14, 96),
    "B16": Cfg("B16", 16, 80, 14, 96),
    # 3 자/s (50 ms, 20 Hz)
    "C16": Cfg("C16", 16, 50, 14, 96),
    # 2비트/심볼 (R2/n 격자, Kc = 상태 비트 m)
    "A32r2": Cfg("A32r2", 32, 256, 10, 64, Kc=10, bpsym=2),     # 1 자/s: 256 ms · 3.9 Hz · 125 Hz
    "A64r2": Cfg("A64r2", 64, 256, 10, 64, Kc=10, bpsym=2),     # 1 자/s: 250 Hz
    "B32r2": Cfg("B32r2", 32, 160, 10, 96, Kc=10, bpsym=2),     # 2 자/s: 160 ms · 6.25 Hz · 200 Hz
    "B64r2": Cfg("B64r2", 64, 160, 10, 96, Kc=10, bpsym=2),     # 2 자/s: 400 Hz
    "C32r2": Cfg("C32r2", 32, 100, 10, 96, Kc=10, bpsym=2),     # 3 자/s: 100 ms · 10 Hz · 320 Hz
}


# ------------------------------------------------------------------ 구조 (동기 위치 · 톤, 천공, 인터리버)
def layout(c):
    r = np.random.default_rng(c.seed)
    # 동기: 블록 안에 고르게 (간격 nsym/NS) + 작은 흔들림, 톤은 서로 다르고 넓게
    base = (np.arange(c.NS) + 0.5) * c.nsym / c.NS
    pos = np.clip(np.round(base + r.uniform(-0.3, 0.3, c.NS) * c.nsym / c.NS / 2).astype(int), 0, c.nsym - 1)
    pos = np.unique(pos)
    assert len(pos) == c.NS
    tone = r.permutation(c.M)[:c.NS]
    if c.sync_pos is not None:
        pos, tone = np.asarray(c.sync_pos), np.asarray(c.sync_tone)
        assert len(pos) == c.NS and len(np.unique(pos)) == c.NS
    data_pos = np.setdiff1d(np.arange(c.nsym), pos)
    keep = None
    ilv = r.permutation(c.ND)                               # 심볼 인터리버 (블록 안)
    return pos, tone, data_pos, keep, ilv


_LAY = {}


def lay(c):
    if c.name not in _LAY:
        _LAY[c.name] = layout(c)
    return _LAY[c.name]


# ------------------------------------------------------------------ CRC16 (CCITT, 비트열)
CRC_INIT = 0xFFFF
CRC_INIT_FIRST = 0x5A5A      # wire rev 2: 메시지 첫 블록은 CRC16 시작값을 바꿔 표시 (문자 비트 · 심볼 추가 없음, 거짓 통과 ×2)


class Bits(np.ndarray):
    """복호한 정보 비트 + first (메시지 첫 블록인지, wire rev 2 CRC 시작값으로 판정)"""
    first = False


def _bits(u, first):
    b = np.asarray(u).view(Bits)
    b.first = bool(first)
    return b


def crc16_bits(bits, init=CRC_INIT):
    reg = init
    for b in bits:
        top = ((reg >> 15) & 1) ^ int(b)
        reg = (reg << 1) & 0xFFFF
        if top:
            reg ^= 0x1021
    return [(reg >> (15 - i)) & 1 for i in range(16)]


# ------------------------------------------------------------------ 테일바이팅 컨볼루션 (가지 출력 n 비트 = 심볼 1개, M = 2^n)
# 심볼 자유 거리 d = K 인 생성다항식 (exp_tng1_gensearch.py 무작위 탐색 → results/tng1_build/gensearch.log)
GENS = {(7, 4): (0o137, 0o121, 0o157, 0o175), (7, 5): (0o177, 0o123, 0o131, 0o163, 0o153),
        (7, 6): (0o123, 0o127, 0o113, 0o161, 0o173, 0o165),
        (9, 4): (0o657, 0o441, 0o633, 0o727), (9, 5): (0o773, 0o521, 0o503, 0o661, 0o727),
        (9, 6): (0o737, 0o535, 0o513, 0o421, 0o763, 0o423),
        (11, 4): (0o2317, 0o2655, 0o3731, 0o2163), (11, 5): (0o2155, 0o3121, 0o2617, 0o2431, 0o2625),
        (11, 6): (0o3775, 0o3433, 0o2517, 0o2323, 0o2305, 0o3473),
        (13, 4): (0o13635, 0o12055, 0o15727, 0o17453), (13, 5): (0o17657, 0o15033, 0o12407, 0o15021, 0o11113),
        (15, 4): (0o57165, 0o50261, 0o67533, 0o76255), (15, 5): (0o63311, 0o65541, 0o40567, 0o72445, 0o64115)}


def _par(v):
    return bin(v).count("1") & 1


class Trellis:
    def __init__(self, K, n):
        self.K, self.n, self.S = K, n, 1 << (K - 1)
        g = GENS[(K, n)]
        lab = np.zeros((self.S, 2), int)
        for s in range(self.S):
            for u in range(2):
                reg = (u << (K - 1)) | s
                v = 0
                for gg in g:
                    v = (v << 1) | _par(reg & gg)
                lab[s, u] = v
        self.lab = lab
        ns = np.arange(self.S)
        self.U = ns >> (K - 2)                                # 다음 상태 ns 의 입력 비트
        self.P0 = (ns << 1) & (self.S - 1)                    # 이전 상태 두 개
        self.P1 = self.P0 | 1
        self.l0 = lab[self.P0, self.U]
        self.l1 = lab[self.P1, self.U]

    def encode(self, bits):
        K = self.K
        s = 0
        for u in bits[-(K - 1):]:                             # 테일바이팅: 마지막 K-1 비트로 시작 상태
            s = ((u << (K - 1)) | s) >> 1
        out = np.zeros(len(bits), int)
        for i, u in enumerate(bits):
            out[i] = self.lab[s, u]
            s = ((u << (K - 1)) | s) >> 1
        return out

    def viterbi(self, Lm, wrap=None, known=None):
        """Lm: (B, k, M) 심볼 로그 우도. 순환 확장 테일바이팅 비터비 → (B, k) 비트.
        known: (k,) -1 = 모름, 0/1 = 사전 지식 비트 (그 입력이 아닌 가지 금지)"""
        B, k, M = Lm.shape
        W = wrap or min(k, 6 * self.K)
        x = np.concatenate([Lm[:, k - W:], Lm, Lm[:, :W]], 1).astype(np.float32)
        n = x.shape[1]
        kn = None if known is None else np.concatenate([known[k - W:], known, known[:W]])
        pm = np.zeros((B, self.S), np.float32)
        dec = np.zeros((n, B, self.S), bool)
        P0, P1, l0, l1 = self.P0, self.P1, self.l0, self.l1
        for t in range(n):
            xt = x[:, t]
            a = pm[:, P0] + xt[:, l0]
            b = pm[:, P1] + xt[:, l1]
            dec[t] = b > a
            pm = np.maximum(a, b)
            if kn is not None and kn[t] >= 0:
                pm = np.where(self.U[None] == kn[t], pm, -1e9)
            pm -= pm.max(1, keepdims=True)
        s = pm.argmax(1)
        out = np.zeros((B, n), np.int8)
        ar = np.arange(B)
        for t in range(n - 1, -1, -1):
            out[:, t] = s >> (self.K - 2)
            s = np.where(dec[t, ar, s], P1[s], P0[s])
        return out[:, W:W + k]

    def viterbi_list(self, Lm, L, wrap=None):
        """병렬 리스트 비터비 (상태마다 상위 L 경로). Lm: (k, M) → 서로 다른 후보 비트열 list (좋은 순, 최대 L)"""
        k, M = Lm.shape
        W = wrap or min(k, 6 * self.K)
        x = np.concatenate([Lm[k - W:], Lm, Lm[:W]], 0).astype(np.float64)
        n = len(x)
        S = self.S
        pm = np.full((S, L), -1e30)
        pm[:, 0] = 0.0
        ptr = np.zeros((n, S, L), np.int16)
        P0, P1, l0, l1 = self.P0, self.P1, self.l0, self.l1
        for t in range(n):
            c = np.concatenate([pm[P0] + x[t][l0][:, None], pm[P1] + x[t][l1][:, None]], 1)   # (S, 2L)
            o = np.argsort(-c, 1)[:, :L]
            ptr[t] = o
            pm = np.take_along_axis(c, o, 1)
            pm -= pm.max()
        flat = np.argsort(-pm.reshape(-1))[:4 * L]
        outs, seen = [], set()
        for f in flat:
            s, r = divmod(int(f), L)
            if pm[s, r] < -1e29:
                break
            bits = np.zeros(n, np.int8)
            for t in range(n - 1, -1, -1):
                bits[t] = s >> (self.K - 2)
                b, r = divmod(int(ptr[t, s, r]), L)
                s = P1[s] if b else P0[s]
            u = bits[W:W + k]
            key = u.tobytes()
            if key not in seen:
                seen.add(key)
                outs.append(u)
            if len(outs) >= L:
                break
        return outs


# 2비트/심볼: 레지스터 [u(2) | 상태 m], 가지 4개. 심볼 거리 d = m/2 + 1 (exp_tng1_gensearch2.py → gensearch2*.log)
GENS2 = {(6, 5): (0o75, 0o116, 0o33, 0o271, 0o146), (6, 6): (0o45, 0o34, 0o70, 0o156, 0o230, 0o252),
         (8, 5): (0o1365, 0o1076, 0o1137, 0o1447, 0o1444), (8, 6): (0o1022, 0o1347, 0o236, 0o1061, 0o430, 0o642),
         (10, 5): (0o2253, 0o6432, 0o6036, 0o2550, 0o3041), (10, 6): (0o2463, 0o2361, 0o7476, 0o6241, 0o7605, 0o6463)}


class Trellis2:
    """정보 2비트/가지. Kc 자리에 상태 비트 수 m 을 쓴다 (Cfg.Kc = m)"""

    def __init__(self, m, n):
        self.m, self.n, self.S = m, n, 1 << m
        g = GENS2[(m, n)]
        lab = np.zeros((self.S, 4), int)
        for s in range(self.S):
            for u in range(4):
                reg = (u << m) | s
                v = 0
                for gg in g:
                    v = (v << 1) | _par(reg & gg)
                lab[s, u] = v
        self.lab = lab
        ns = np.arange(self.S)
        self.U = ns >> (m - 2)
        self.P = [((ns << 2) & (self.S - 1)) | q for q in range(4)]
        self.L = [lab[P, self.U] for P in self.P]

    def encode(self, bits):
        m = self.m
        u2 = [int(bits[i]) * 2 + int(bits[i + 1]) for i in range(0, len(bits), 2)]
        s = 0
        for u in u2[-(m // 2):]:
            s = ((u << m) | s) >> 2
        out = np.zeros(len(u2), int)
        for i, u in enumerate(u2):
            out[i] = self.lab[s, u]
            s = ((u << m) | s) >> 2
        return out

    def viterbi(self, Lm, wrap=None):
        B, k, M = Lm.shape
        W = wrap or min(k, 5 * self.m)
        x = np.concatenate([Lm[:, k - W:], Lm, Lm[:, :W]], 1).astype(np.float32)
        n = x.shape[1]
        pm = np.zeros((B, self.S), np.float32)
        dec = np.zeros((n, B, self.S), np.int8)
        for t in range(n):
            xt = x[:, t]
            c = np.stack([pm[:, P] + xt[:, L] for P, L in zip(self.P, self.L)], 0)   # (4, B, S)
            dec[t] = c.argmax(0)
            pm = c.max(0)
            pm -= pm.max(1, keepdims=True)
        s = pm.argmax(1)
        out = np.zeros((B, n, 2), np.int8)
        ar = np.arange(B)
        Pm = np.stack(self.P)                                # (4, S)
        for t in range(n - 1, -1, -1):
            u = s >> (self.m - 2)
            out[:, t, 0] = u >> 1
            out[:, t, 1] = u & 1
            s = Pm[dec[t, ar, s], s]
        return out[:, W:W + k].reshape(B, -1)


_TR = {}


def trellis(c):
    key = (c.Kc, c.bps, c.bpsym)
    if key not in _TR:
        _TR[key] = Trellis(c.Kc, c.bps) if c.bpsym == 1 else Trellis2(c.Kc, c.bps)
    return _TR[key]


# ------------------------------------------------------------------ 블록 부호화 · 변조
def crc_init(c, first):
    return CRC_INIT_FIRST if (c.first_flag and first) else CRC_INIT


def encode_block(c, info_bits, first=False):
    """first: 메시지 첫 블록 (c.first_flag 일 때 CRC 시작값 CRC_INIT_FIRST)"""
    pos, tone, data_pos, keep, ilv = lay(c)
    assert len(info_bits) == c.KI
    u = list(info_bits) + crc16_bits(info_bits, crc_init(c, first))
    sy = trellis(c).encode(u)[ilv]                          # 가지 출력 = 심볼
    s = np.zeros(c.nsym, int)
    s[pos] = tone
    s[data_pos] = sy
    return s


def text_blocks(c, text):
    """→ [(심볼 배열, 블록 글자)] 스트리밍 순서"""
    t, _ = charset1.normalize(text)
    out = []
    while True:
        bits, n, eot = charset1.pack_block(t, c.KI, end=True)
        out.append((encode_block(c, bits, first=not out), t[:n]))
        t = t[n:]
        if eot:
            return out
        if not t and not eot:
            continue


def gfsk(c, syms, f0=0.0):
    """심볼 → 복소 기저대역 (PEP 1). 톤 m 주파수 = (m - M/2 + 0.5)/T + f0, 가우스 주파수 펄스 (BT)"""
    Ns = c.Ns
    fd = c.slot[np.asarray(syms)] / c.T
    f = np.repeat(fd, Ns)
    if c.chirp and c.chirp_smooth:
        f = f + np.tile(_chirp_f(c), len(syms))            # 처프 되돌림 점프도 같은 가우스로 부드럽게 (대역 좁게)
    if c.bt:
        sig = math.sqrt(math.log(2)) / (2 * math.pi * c.bt / c.T) * FS     # 가우스 표준편차 [샘플]
        h = np.exp(-0.5 * (np.arange(-3 * int(sig + 1), 3 * int(sig + 1) + 1) / sig) ** 2)
        h /= h.sum()
        f = np.convolve(np.concatenate([np.full(len(h), f[0]), f, np.full(len(h), f[-1])]), h, "same")[len(h):-len(h)]
    if c.chirp and not c.chirp_smooth:
        fc = np.tile(_chirp_f(c), len(syms))                # 처프는 따로 (기본: 부드럽게 하지 않음 → 역처프가 정확하게)
        if c.chirp_bt:                                      # 처프 되돌림만 더 날카로운 가우스 (BT chirp_bt) 로
            sg = math.sqrt(math.log(2)) / (2 * math.pi * c.chirp_bt / c.T) * FS
            h = np.exp(-0.5 * (np.arange(-3 * int(sg + 1), 3 * int(sg + 1) + 1) / sg) ** 2)
            h /= h.sum()
            fc = np.convolve(np.concatenate([np.full(len(h), fc[0]), fc, np.full(len(h), fc[-1])]), h, "same")[len(h):-len(h)]
        f = f + fc
    ph = 2 * math.pi * np.cumsum(f + f0) / FS
    return np.exp(1j * ph)


def _chirp_f(c):
    """심볼 안 처프 순간 주파수 (Ns,) : -chirp/2 → +chirp/2"""
    return c.chirp * ((np.arange(c.Ns) + 0.5) / c.Ns - 0.5)


def _dechirp(c):
    """심볼 창 (Ns) 에 곱할 역처프 + 톤 소수부 이동 (반 빈 등)"""
    frac = float(c.slot[0] - np.floor(c.slot[0]))
    ph = 2 * math.pi * (np.cumsum(_chirp_f(c)) / FS if c.chirp else 0.0) + 2 * math.pi * frac / c.T * np.arange(c.Ns) / FS
    return np.exp(-1j * ph)


def _bins(c, fsteps=1):
    """톤 m 의 FFT 빈 (역처프 · 소수부 이동 뒤, 길이 fsteps·Ns)"""
    frac = float(c.slot[0] - np.floor(c.slot[0]))
    return np.round(fsteps * (c.slot - frac)).astype(int)


def tx_stream(c, text, ramp_ms=20):
    blocks = text_blocks(c, text)
    syms = np.concatenate([b[0] for b in blocks])
    x = gfsk(c, syms)
    r = int(ramp_ms * FS / 1000)
    w = 0.5 - 0.5 * np.cos(np.linspace(0, math.pi, r))
    x[:r] *= w
    x[-r:] *= w[::-1]
    return x, blocks


def to_audio(x, fs_a=8000, fc=1500.0):
    """복소 1 kHz → 실수 오디오 (PEP 1 → 최고 진폭 1)"""
    from scipy.signal import resample_poly
    up = resample_poly(x, int(fs_a), int(FS), window=("kaiser", 10.0))   # 영상 (±1 kHz 간격) -72 → -109 dB (3부 측정)
    fsi, fci = int(fs_a), int(round(fc))
    if abs(fc - fci) < 1e-9 and fci > 0:                  # 16부: 반송파는 주기 표를 되풀이 (복소 exp 수백만 번 대신, 값은 같음)
        P = fsi // math.gcd(fsi, fci)
        car = np.resize(np.exp(2j * math.pi * fci * np.arange(P) / fs_a), len(up))
    else:
        car = np.exp(2j * math.pi * fc * np.arange(len(up)) / fs_a)
    return np.real(up * car)


def from_audio(a, fs_a, fc=1500.0):
    """실수 오디오 → 복소 1 kHz (중심 fc). 크기는 레벨 무관 (수신은 정규화)"""
    from scipy.signal import resample_poly
    t = np.arange(len(a)) / fs_a
    z = a * np.exp(-2j * math.pi * fc * t) * 2
    g = math.gcd(int(fs_a), int(FS))
    return resample_poly(z, int(FS) // g, int(fs_a) // g)


def likeness(a, fs_a, f0=0.0, bw=200.0, c=None):
    """실수 오디오 조각 (1 s 남짓) 이 TNG1 신호인가: 역처프 160 ms 창마다 (f0 ± bw) 안 최대 칸 에너지 ÷ 합, 기호 위상 4곳 중 중앙값 최대.
    TNG1 은 한 기호가 한 칸에 모임 (측정: 깨끗함 0.34~0.48 · -10 dB 0.16~0.26), TNG5 A 자리 0.05~0.084 · 잡음 0.05~0.06 ·
    말소리 ≤0.10 · FT8 ≤0.13 · 휘파람 ≤0.27 (09-26). 다른 모드 검출기가 TNG1 에 걸리는지 거를 때 씀"""
    c = c or CFGS["TNG1"]
    z = from_audio(np.asarray(a, float), fs_a)
    if len(z) < 4 * c.Ns:
        return 0.0
    P, hop = spectrogram(c, z)
    fr = np.fft.fftfreq(P.shape[1], 1.0 / FS)
    Pb = P[:, np.abs(fr - f0) <= bw]
    e = Pb.sum(1)
    conc = Pb.max(1) / (e + 1e-30)
    live = e >= 0.25 * np.percentile(e, 90)              # 무음 · 약한 꼬리 창은 뺌 (TNG1 이 창 가운데서 끝나도 판단되게)
    sh = c.Ns // hop
    v = [np.median(conc[k::sh][live[k::sh]]) for k in range(sh) if live[k::sh].sum() >= 3]
    return float(max(v)) if v else 0.0


# ------------------------------------------------------------------ 채널 (실험)
WATT = {"awgn": None, "moderate": (1.0, 0.5), "poor": (2.0, 1.0)}


def watterson_gain(n, spread, rng, fsg=50.0):
    """가우스 도플러 (2σ = spread) 복소 이득, fsg 로 만들고 선형 보간 → n 샘플 @FS"""
    ng = int(n / FS * fsg) + 8
    N = 1 << int(math.ceil(math.log2(ng + 64)))
    fr = np.fft.fftfreq(N, 1 / fsg)
    sig = spread / 2
    H = np.exp(-fr ** 2 / (4 * sig ** 2))
    w = rng.standard_normal(N) + 1j * rng.standard_normal(N)
    g = np.fft.ifft(w * H) * (N / np.sqrt(2 * (H ** 2).sum()))
    tg = np.arange(ng) / fsg
    tt = np.arange(n) / FS
    return np.interp(tt, tg, g[:ng].real) + 1j * np.interp(tt, tg, g[:ng].imag)


def channel(x, snr_db, kind, rng, df=0.0, drift_hz_min=0.0, level_db=0.0, pre=0, post=0, imp=None):
    """x (PEP 1) → 잡음 포함 수신. df: 주파수 오프셋, drift: 선형 표류 (Hz/분), 앞뒤 무신호 pre/post 샘플
    imp = (초당 횟수, 길이 ms, 잡음 대비 dB): 순간 잡음 (톡톡, 포아송 시점, 복소 가우스 덩어리)"""
    x = np.concatenate([np.zeros(pre, complex), x, np.zeros(post, complex)])
    n = len(x)
    y = x
    fad = WATT[kind]
    if fad:
        d = int(round(fad[0] * FS / 1000))
        xd = np.concatenate([np.zeros(d, complex), x[:n - d]])
        y = (watterson_gain(n, fad[1], rng) * x + watterson_gain(n, fad[1], rng) * xd) / math.sqrt(2)
    t = np.arange(n) / FS
    ph = 2 * math.pi * np.cumsum(df + drift_hz_min / 60.0 * t) / FS + rng.uniform(0, 2 * math.pi)
    y = y * np.exp(1j * ph) * 10 ** (level_db / 20)
    sig2 = FS / 2500.0 / 10 ** (snr_db / 10) * 10 ** (level_db / 10)
    y = y + (rng.standard_normal(n) + 1j * rng.standard_normal(n)) * math.sqrt(sig2 / 2)
    if imp:
        rate, ms, db = imp
        d = max(1, int(round(ms * FS / 1000)))
        for t0 in rng.choice(n - d, rng.poisson(rate * n / FS), replace=False) if n > d else []:
            y[t0:t0 + d] += (rng.standard_normal(d) + 1j * rng.standard_normal(d)) * math.sqrt(sig2 * 10 ** (db / 10) / 2)
    return y


def impulse_blank(y, k=10.0, guard=2, chunk=200):
    """순간 잡음 지우기: 샘플 전력이 주변 (chunk 샘플 조각 중앙값, 이웃 조각 보간) 의 k 배를 넘으면 앞뒤 guard 샘플까지 0.
    신호는 포락선이 일정하고 약해서 걸리지 않는다. → (y', 지운 비율)"""
    p = np.abs(y) ** 2
    nc = max(1, len(p) // chunk)
    med = np.array([np.median(p[i * chunk:(i + 1) * chunk]) for i in range(nc)])
    med = np.minimum.reduce([med, np.roll(med, 1), np.roll(med, -1)]) if nc > 2 else med   # 조각 안이 톡톡에 오염돼도 이웃 최소
    ref = np.interp(np.arange(len(p)), (np.arange(nc) + 0.5) * chunk, med) / math.log(2)
    m = p > k * ref
    if guard:
        m = np.convolve(m, np.ones(2 * guard + 1), "same") > 0
    y2 = y.copy()
    y2[m] = 0
    return y2, float(m.mean())


# ------------------------------------------------------------------ 복조 (위치 · 주파수 주어짐)
def tone_energy(c, y, t0, f0, nsym=None):
    """블록 시작 t0 (샘플, 실수 허용 → 반올림), 주파수 f0 → (nsym, M) 에너지"""
    nsym = nsym or c.nsym
    i0 = int(round(t0))
    seg = y[i0:i0 + nsym * c.Ns]
    if len(seg) < nsym * c.Ns:
        return None
    t = (i0 + np.arange(len(seg))) / FS
    z = (seg * np.exp(-2j * math.pi * f0 * t)).reshape(nsym, c.Ns)
    z = z * _dechirp(c)[None]                               # 역처프 + 톤 소수부 (반 빈) 이동
    Z = np.fft.fft(z, axis=1)
    idx = _bins(c) % c.Ns
    return np.abs(Z[:, idx]) ** 2


BURST_K = 3.0      # 반복 합치기: 심볼 전체 에너지가 평소 (잡음 기준) 의 이 배를 넘는 심볼은 가중 0 (15부, 다른 센 신호 · 순간 잡음 버스트)


def burst_rows(c, E, k=BURST_K):
    """사본 에너지 (nsym, M) 에서 에너지가 튀는 심볼 (True). 순간 잡음 지우기와 같은 원리 — 합칠 때만 씀"""
    sig = max(np.median(E) / math.log(2), 1e-30)
    return E.mean(1) / sig > k


def sym_metric(c, E, metric="ray", burst0=False):
    """(nsym, M) 에너지 → 데이터 심볼 로그 우도 (ND, M) 인터리버 풀린 순서, 추정 심볼 SNR.
    burst0: 에너지가 튀는 심볼 (burst_rows) 의 우도를 0 (지움) — 반복 합치기에서만 (15부)"""
    pos, tone, data_pos, keep, ilv = lay(c)
    bad = burst_rows(c, E) if burst0 else None
    sig = max(np.median(E) / math.log(2), 1e-30)
    if c.symnorm:                                          # 심볼 전체 에너지가 평소 (≈ 1 + es/M) 의 2배를 넘으면 그만큼 덜 믿음 (넓은 대역 순간 잡음)
        mt = E.mean(1) / sig
        E = E * np.minimum(1.0, 2.0 / mt)[:, None]
    es = max(np.mean(E[pos, tone]) / sig - 1.0, 0.05)
    Ed = E[data_pos] / sig
    if metric == "ray":
        Lm = es / (1.0 + es) * Ed                           # 레일리 최적 (제곱 법칙)
    else:
        from scipy.special import i0e
        a = 2 * np.sqrt(es * Ed)
        Lm = np.log(i0e(a)) + a                             # AWGN 최적 (log I0)
    if bad is not None and bad[data_pos].any():
        Lm = Lm.copy()
        Lm[bad[data_pos]] = 0.0
    out = np.zeros_like(Lm)
    out[ilv] = Lm
    return out, es


def decode_sym(c, Lm, known=None):
    """리스트 크기 c.lst: 좋은 순으로 CRC 확인 (거짓 통과 확률 ≈ lst · 2^-16). known: 사전 지식 비트 (1비트/심볼 격자만)"""
    tr = trellis(c)
    if known is not None:
        cands = [tr.viterbi(Lm[None], known=known)[0]]
    else:
        cands = [tr.viterbi(Lm[None])[0]] if c.lst <= 1 or c.bpsym > 1 else tr.viterbi_list(Lm, c.lst)
    for u in cands:
        info, crc = u[:c.KI], u[c.KI:]
        crc = list(crc)
        if crc == crc16_bits(info):
            return _bits(info, False), True
        if c.first_flag and crc == crc16_bits(info, CRC_INIT_FIRST):
            return _bits(info, True), True
    return None, False


def local_metric(c, E, u, w=4):
    """2차 복호용: 1차 복호 경로 (CRC 실패해도) 로 심볼별 신호 에너지 → ±w 심볼 이동 평균 → 심볼별 레일리 가중"""
    pos, tone, data_pos, keep, ilv = lay(c)
    sig = max(np.median(E) / math.log(2), 1e-30)
    lab = trellis(c).encode(u)[ilv]
    e = np.zeros(c.nsym)
    e[pos] = E[pos, tone] / sig - 1.0
    e[data_pos] = E[data_pos, lab] / sig - 1.0
    k = np.ones(2 * w + 1) / (2 * w + 1)
    es = np.maximum(np.convolve(np.concatenate([e[-w:], e, e[:w]]), k, "valid"), 0.05)    # 블록 경계는 순환 (근사)
    Lm = (es / (1.0 + es))[data_pos, None] * E[data_pos] / sig
    out = np.zeros_like(Lm)
    out[ilv] = Lm
    return out


def msd_metric(c, y, t0, f0, seq, W, es):
    """음 연결 (연속 위상) 복조: 심볼 t 의 가설 m 마다 [t-W, t+W] 창을 1차 결정 이웃 + m 으로 GFSK 그대로 만들어 결맞음 상관
    → 레일리 가중 |z|² (창 잡음 = (2W+1)·sig). 반환: 데이터 심볼 우도 (ND, M) 인터리버 풀린 순서"""
    pos, tone, data_pos, keep, ilv = lay(c)
    Ns = c.Ns
    i0 = int(round(t0))
    seg = y[i0:i0 + c.nsym * Ns]
    tt = (i0 + np.arange(len(seg))) / FS
    z0 = seg * np.exp(-2j * math.pi * f0 * tt)
    E = tone_energy(c, y, t0, f0)
    sig = max(np.median(E) / math.log(2), 1e-30)
    esw = (2 * W + 1) * es
    g = esw / (1.0 + esw)
    Lm = np.zeros((c.ND, c.M))
    for j, t in enumerate(data_pos):
        a, b = max(0, t - W), min(c.nsym, t + W + 1)
        lo, hi = max(0, a - 1), min(c.nsym, b + 1)                 # 가장자리 한 심볼은 부드럽게 하기 문맥
        sq = np.repeat(seq[lo:hi][None], c.M, 0)
        sq[:, t - lo] = np.arange(c.M)
        wv = np.stack([gfsk(c, r) for r in sq])[:, (a - lo) * Ns:(b - lo) * Ns]
        zz = wv.conj() @ z0[a * Ns:b * Ns]
        Lm[j] = g * np.abs(zz) ** 2 / ((b - a) * sig)
    out = np.zeros_like(Lm)
    out[ilv] = Lm
    return out


def _box_phase(c):
    """심볼 하나의 주파수 상자 (길이 Ns, 값 1/T) 를 GFSK 가우스로 부드럽게 한 뒤 누적한 위상 모양 (2π 제외).
    → (앞쪽 여유 샘플 수, 배열). 가설 m 은 기준 파형에 exp(j 2π (slot_m - slot_기준) · 이 모양) 를 곱하면 된다"""
    Ns = c.Ns
    if not c.bt:
        return 0, np.concatenate([np.cumsum(np.full(Ns, 1.0 / c.T)) / FS])
    sig = math.sqrt(math.log(2)) / (2 * math.pi * c.bt / c.T) * FS
    h = np.exp(-0.5 * (np.arange(-3 * int(sig + 1), 3 * int(sig + 1) + 1) / sig) ** 2)
    h /= h.sum()
    g = len(h) // 2
    box = np.zeros(Ns + 2 * g)
    box[g:g + Ns] = 1.0 / c.T
    return g, np.cumsum(np.convolve(box, h, "same")) / FS


def msd_metric_v(c, y, t0, f0, seq, W, es):
    """msd_metric 벡터화: 블록 전체 기준 파형 (1차 결정) 을 한 번 만들고, 가설마다 위상 보정만 곱해 창 상관"""
    pos, tone, data_pos, keep, ilv = lay(c)
    Ns = c.Ns
    i0 = int(round(t0))
    seg = y[i0:i0 + c.nsym * Ns]
    tt = (i0 + np.arange(len(seg))) / FS
    z0 = seg * np.exp(-2j * math.pi * f0 * tt)
    base = gfsk(c, seq)
    g, shape = _box_phase(c)
    E = tone_energy(c, y, t0, f0)
    sig = max(np.median(E) / math.log(2), 1e-30)
    esw = (2 * W + 1) * es
    gam = esw / (1.0 + esw)
    dslot = c.slot[np.arange(c.M)][:, None] - c.slot[np.asarray(seq)][None]      # (M, nsym)
    Lm = np.zeros((c.ND, c.M))
    for j, t in enumerate(data_pos):
        a, b = max(0, t - W), min(c.nsym, t + W + 1)
        lo, hi = a * Ns, b * Ns
        # 심볼 t 의 위상 모양을 창 [lo, hi) 에 놓음 (심볼 뒤는 누적값 그대로 유지 = 연속 위상)
        ph = np.zeros(hi - lo)
        s0 = t * Ns - g                                        # 모양 배열 0 번이 놓이는 절대 샘플
        k = np.arange(lo, hi) - s0
        inside = (k >= 0) & (k < len(shape))
        ph[inside] = shape[k[inside]]
        ph[k >= len(shape)] = shape[-1]
        wv = base[lo:hi][None] * np.exp(2j * math.pi * dslot[:, t][:, None] * ph[None])
        zz = wv.conj() @ z0[lo:hi]
        Lm[j] = gam * np.abs(zz) ** 2 / ((b - a) * sig)
    out = np.zeros_like(Lm)
    out[ilv] = Lm
    return out


def decode_at(c, y, t0, f0, metric="ray", iters=None):
    """iters: 2차 복호 (국소 SNR 가중) 횟수. None = c.iters"""
    E = tone_energy(c, y, t0, f0)
    if E is None:
        return None, False, 0.0
    Lm, es = sym_metric(c, E, metric)
    info, ok = decode_sym(c, Lm)
    it = c.iters if iters is None else iters
    while not ok and it > 0:
        u = trellis(c).viterbi(Lm[None])[0]
        Lm = local_metric(c, E, u)
        info, ok = decode_sym(c, Lm)
        it -= 1
    if not ok and c.msd:
        pos, tone, data_pos, _, ilv = lay(c)
        u = trellis(c).viterbi(Lm[None])[0]
        seq = np.zeros(c.nsym, int)
        seq[pos] = tone
        seq[data_pos] = trellis(c).encode(u)[ilv]
        info, ok = decode_sym(c, msd_metric_v(c, y, t0, f0, seq, c.msd, es))
    return info, ok, es


# ------------------------------------------------------------------ 동기 (접어 누적)
def spectrogram(c, y, hop_div=None, fsteps=2):
    """→ P (nt, nf): 홉 Ns/hop_div (Ns 를 나누어떨어지게 4 · 5 · 6 · 8 중), 주파수 간격 1/(fsteps·T), 창 없음. 범위 = 전체 FS"""
    hop_div = hop_div or next(d for d in (4, 5, 6, 8, 2, 1) if c.Ns % d == 0)
    hop = c.Ns // hop_div
    nt = (len(y) - c.Ns) // hop + 1
    idx = np.arange(c.Ns)[None] + hop * np.arange(nt)[:, None]
    Z = np.fft.fft(y[idx] * _dechirp(c)[None], n=fsteps * c.Ns, axis=1)   # 창마다 역처프 (창 = 심볼과 맞을 때 정확)
    return np.abs(Z) ** 2, hop


def fold_metric(c, P, hop, K, fsteps=2, fmax=60.0, clip=4.0):
    """접은 동기 지표 S (nt', nfo): 블록 시작 후보 (홉) × 주파수 오프셋 (1/(fsteps T) 간격, ±fmax)"""
    pos, tone, _, _, _ = lay(c)
    Lh = c.L // hop
    sh = c.Ns // hop
    nfft = P.shape[1]
    df = 1.0 / (fsteps * c.T)
    offs = np.arange(-int(fmax / df), int(fmax / df) + 1)
    # 톤 m 의 빈 = fsteps·(m - M/2 + 0.5) + off (반 빈 기준) → fsteps 짝수라 정수
    tb = _bins(c, fsteps)
    nt = P.shape[0] - (K * Lh)
    if nt <= 0:
        return None, offs * df
    # 각 (시간, 오프셋) 의 M 톤 평균 (대비 분모)
    G = np.zeros((P.shape[0], len(offs)))
    for m in range(c.M):
        G += P[:, (tb[m] + offs) % nfft]
    G = G / c.M + 1e-12
    S = np.zeros((nt, len(offs)))
    for k in range(K):
        for p, tn in zip(pos, tone):
            r = k * Lh + p * sh
            S += np.minimum(P[r:r + nt][:, (tb[tn] + offs) % nfft] / G[r:r + nt], clip)
    return S, offs * df


PS_MU, PS_SD = 0.99, 0.90    # 잡음 한 칸 항 min(P/G, 4) 의 평균 · 표준편차 (partial_sync 정규화, 측정: 잡음 20 창)


def partial_sync(c, P, hop, fmax=60.0, fsteps=2, clip=4.0):
    """끝나지 않은 블록 포함 부분 동기 지표 (TNG1 감지 표시용, 10부): 시작 후보 t 마다 지금까지 들어온 동기 칸만 합 (fold_metric K=1 과 같은 항).
    → (z (nt, nf) = (합 - m·μ) / (√m·σ), m (nt) = 쓴 동기 칸 수, 주파수 오프셋). 엿보기 스펙트로그램을 그대로 씀 (새 FFT 없음)"""
    pos, tone, _, _, _ = lay(c)
    sh = c.Ns // hop
    nfft = P.shape[1]
    df = 1.0 / (fsteps * c.T)
    offs = np.arange(-int(fmax / df), int(fmax / df) + 1)
    tb = _bins(c, fsteps)
    G = np.zeros((P.shape[0], len(offs)))
    for m_ in range(c.M):
        G += P[:, (tb[m_] + offs) % nfft]
    G = G / c.M + 1e-12
    nP = P.shape[0]
    S = np.zeros((nP, len(offs)))
    m = np.zeros(nP)
    for p, tn in zip(pos, tone):
        r = p * sh
        if r >= nP:
            continue
        S[:nP - r] += np.minimum(P[r:, (tb[tn] + offs) % nfft] / G[r:], clip)
        m[:nP - r] += 1
    z = (S - m[:, None] * PS_MU) / (np.sqrt(np.maximum(m, 1))[:, None] * PS_SD)
    return z, m, offs * df


def shape_detect(c, P, hop, n=8, fmax=60.0, fsteps=2):
    """TNG1 소리 모양 빠른 감지 (15부, 표시 전용): 역처프 뒤 한 심볼의 에너지가 16음 가운데 한 칸에 몰리는 정도
    (집중도 = 최대 칸 / 16칸 합, 잡음 약 0.21) 를 동기음 · 데이터음 모두로 봄. 심볼 창이 맞을 때 (정렬) 와 반 심볼 어긋날 때의
    집중도 차 D = 평균(정렬) - 평균(반 어긋남), 최근 n 심볼. 휘파람 · 고정음 (역처프로 퍼짐) · 느린 쓸기 (창 위치 무관) 는 D ≈ 0.
    → (D, 정렬 시작 홉 (최근 n 심볼 창의 첫 심볼), 주파수 오프셋 Hz, 신호 시작 추정 홉) 또는 None"""
    sh = c.Ns // hop
    nP, nfft = P.shape
    t_hi = nP - 1 - (n - 1) * sh                       # 창 끝이 가장 최근 프레임
    t_lo = t_hi - sh + 1
    if t_lo - sh // 2 < 0:
        return None
    df = 1.0 / (fsteps * c.T)
    offs = np.arange(-int(fmax / df), int(fmax / df) + 1)
    tb = _bins(c, fsteps)
    idx = (tb[None, :] + offs[:, None]) % nfft                         # (nO, 16)
    V = P[:, idx]                                                       # (nP, nO, 16)
    conc = V.max(2) / (V.sum(2) + 1e-30)                                # (nP, nO)
    best = None
    for t in range(t_lo, t_hi + 1):
        rows = t + sh * np.arange(n)
        A = conc[rows].mean(0)
        B = conc[rows - sh // 2].mean(0)
        D = A - B
        j = int(np.argmax(D))
        if best is None or D[j] > best[0]:
            best = (float(D[j]), t, j, float(A[j]))
    D, t, j, A = best
    mid = 0.21 + 0.5 * (A - 0.21)                                        # 신호 시작: 같은 위상 · 주파수로 뒤로 집중도가 절반 아래로 두 번 연속일 때까지
    k, miss = t, 0
    while k - sh >= 0 and miss < 2:
        k -= sh
        miss = miss + 1 if conc[k, j] < mid else 0
    start = k + (miss) * sh
    return D, t, float(offs[j] * df), min(start, t)


def candidates(S, fr, Lh, n, excl_t=6, excl_f=3):
    """S 에서 상위 n 개 (블록 주기 위상으로 접은 뒤 비최대 억제) → [(위상 홉, 주파수, 값)]"""
    nt = S.shape[0]
    # 블록 주기로 한 번 더 접기 (위상별 최대) — 스트림 어디서든 같은 위상
    ph = np.full((Lh, S.shape[1]), -1.0)
    for t0 in range(0, nt, Lh):
        seg = S[t0:t0 + Lh]
        ph[:len(seg)] = np.maximum(ph[:len(seg)], seg)
    out = []
    A = ph.copy()
    for _ in range(n):
        i = np.unravel_index(np.argmax(A), A.shape)
        if A[i] < 0:
            break
        out.append((int(i[0]), float(fr[i[1]]), float(A[i])))
        for dt in range(-excl_t, excl_t + 1):
            A[(i[0] + dt) % Lh, max(0, i[1] - excl_f):i[1] + excl_f + 1] = -1
    return out


def fine_search(c, y, t0, f0, dts, dfs):
    """동기 톤 에너지 합 최대 (t, f)"""
    pos, tone, _, _, _ = lay(c)
    best = (-1, t0, f0)
    for dt in dts:
        for dfq in dfs:
            E = tone_energy(c, y, t0 + dt, f0 + dfq)
            if E is None:
                continue
            v = E[pos, tone].sum() / (E.mean() * len(pos))
            if v > best[0]:
                best = (v, t0 + dt, f0 + dfq)
    return best


# 잡음만의 접은 동기 지표: 블록 주기당 최대의 99% 분위 ~ 0.6 시간 최대 (A16, NS=12, 150 주기, 09-26 측정)
#   K=1 30.2~30.5 · K=2 47.8~50.3 · K=3 64.3~67.5 · K=4 79.8~80.9 → 99% 분위 쪽. 잡음 후보가 넘을 확률 ~1%/주기 × CRC16 거짓 통과
# TNG1 1.0 (chirp50s) 다시 측정 (09-26, 300 주기 1.2 시간 분량): 99% 분위 K=1 30.7 · 2 47.9 · 3 64.4 · 4 80.5 (최대 32.7 · 50.1 · 68.1 · 90.5)
THR_TAB = {1: 30.7, 2: 48.0, 3: 64.5, 4: 80.5}


@dataclasses.dataclass
class RxParams:
    K: int = 4
    ncand: int = 8
    blank: float = 10.0         # 순간 잡음 지우기 문턱 (0 = 끔). 기본 켬 (2부 측정: 깨끗한 잡음 손해 0)
    msd_top: int = 2            # 음 연결 복조 (c.msd) 는 동기 지표 상위 이만큼 후보에만 (CPU: 실패 블록당 약 0.2 s)
    nstore: int = 2             # 반복 합치기용 (5부: 4 로 늘려 봤으나 시도 수가 크게 늘어 되돌림): CRC 실패 블록 에너지를 돌려줄 상위 후보 수 (0 = 안 돌려줌)
    thr_single: bool = True     # CRC 1개로 표시: 접은 동기 지표 ≥ THR_TAB[K] × NS/12 (False = 2블록 규칙만)
    fsearch: float = 3.0        # 블록별 주파수 미세 탐색 ±Hz
    fmax: float = 60.0


def ridge(c, dt):
    """처프 변형: 시간을 dt 샘플 옮기면 맞는 주파수도 (처프 기울기 × dt) 만큼 옮겨짐 (처프 없으면 0)"""
    return c.chirp / c.T * dt / FS if c.chirp else 0.0


def sync_scan(c, y, t_ph, f0, nblk, hop, fsearch):
    """후보 격자의 블록별 동기 지표 v[b, dt, df] (동기 톤 에너지 합 / 평균 에너지)"""
    pos, tone, _, _, _ = lay(c)
    dts = np.arange(-hop // 2, hop // 2 + 1, max(1, hop // 8))
    st = 0.125 / c.T                                        # 주파수 격자: 톤 간격의 1/8, 대칭, 범위 ±max(fsearch, 0.3/T)
    nst = int(math.ceil(max(fsearch, 0.3 / c.T) / st))
    dfs = np.arange(-nst, nst + 1) * st
    bl = [b for b in range(-1, nblk + 1) if t_ph + b * c.L + dts[0] >= 0 and t_ph + b * c.L + dts[-1] + c.L <= len(y)]
    v = np.zeros((len(bl), len(dts), len(dfs)))
    for i, b in enumerate(bl):
        for j, dt in enumerate(dts):
            for k, dfq in enumerate(dfs):
                E = tone_energy(c, y, t_ph + b * c.L + dt, f0 + ridge(c, dt) + dfq)
                v[i, j, k] = E[pos, tone].sum() / (E.mean() * len(pos))
    return bl, dts, dfs, v


def sync_scan_v(c, y, t_ph, f0, nblk, hop, fsearch):
    """sync_scan 과 같은 값 (벡터화): 시간 위치 dt 마다 동기 심볼만 한 번 FFT (8배 채움 → 1/8 톤 간격) 해서
    모든 주파수 오프셋 (ridge 포함) 을 한꺼번에 읽는다. 분모 (평균 에너지) 는 전체 심볼 16톤 평균 대신
    동기 심볼 16톤 평균의 블록 평균 — 원래 식 (블록 전체 E.mean) 과 기대값이 같고 순위는 거의 같음 (확인: 테스트)"""
    pos, tone, _, _, _ = lay(c)
    dts = np.arange(-hop // 2, hop // 2 + 1, max(1, hop // 8))
    st = 0.125 / c.T
    nst = int(math.ceil(max(fsearch, 0.3 / c.T) / st))
    dfs = np.arange(-nst, nst + 1) * st
    bl = [b for b in range(-1, nblk + 1) if t_ph + b * c.L + dts[0] >= 0 and t_ph + b * c.L + dts[-1] + c.L <= len(y)]
    v = np.zeros((len(bl), len(dts), len(dfs)))
    if not bl:
        return bl, dts, dfs, v
    Ns, OS = c.Ns, 8
    dch = _dechirp(c)
    bins = _bins(c)[tone] * OS                                   # 동기 톤 (8배 격자)
    allb = _bins(c) * OS
    tt = np.arange(Ns) / FS
    for j, dt in enumerate(dts):
        rk = int(round(ridge(c, dt) / st))                        # 처프: 시간 옮김에 맞는 주파수 옮김 (격자 칸)
        starts = np.array([[int(round(t_ph + b * c.L + dt)) + p * Ns for p in pos] for b in bl])   # (nb, NS)
        idx = starts[..., None] + np.arange(Ns)
        t0 = starts / FS
        z = y[idx] * np.exp(-2j * math.pi * f0 * (t0[..., None] + tt)) * dch
        Z = np.abs(np.fft.fft(z, n=OS * Ns, axis=-1)) ** 2 / OS ** 0       # (nb, NS, OS·Ns)
        ks = np.arange(-nst, nst + 1) + rk
        sel = Z[:, np.arange(len(pos))[:, None], (bins[:, None] + ks[None]) % (OS * Ns)]   # (nb, NS, ndf)
        den = Z[:, :, (allb[:, None] + ks[None]) % (OS * Ns)].mean(2)                     # (nb, NS, ndf) 16톤 평균
        v[:, j, :] = sel.sum(1) / (den.mean(1) * len(pos) + 1e-30)
    return bl, dts, dfs, v


def receive(c, y, rp=RxParams(), log=None, abs0=0, cache=None):
    """버퍼 전체 → dict tracks: [{t, f, S, blocks {시작 샘플: (글자, 주파수, es, 정보 비트)}, shown}]
    v2: 시간 위치는 모든 블록 합으로 한 번에 (격자 공통), 주파수는 블록별 최대를 가중 직선 맞춤 (표류) → 블록마다 맞춤값 · 블록 최대 두 곳 복호"""
    if not np.any(y):                                       # 무음 (멈춤 동안 채운 0 등): 처리할 것 없음
        return {"tracks": [], "fails": [], "cands": [], "K": 0}
    if rp.blank:
        y = impulse_blank(y, rp.blank)[0]
    P, hop = spectrogram(c, y)
    Lh = c.L // hop
    nblk = len(y) // c.L
    K = max(1, min(rp.K, nblk - 1))
    S, fr = fold_metric(c, P, hop, K, fmax=rp.fmax)
    if S is None:
        K = 1
        S, fr = fold_metric(c, P, hop, 1, fmax=rp.fmax)
        if S is None:
            return {"tracks": [], "fails": [], "cands": [], "K": 0}
    cands = candidates(S, fr, Lh, rp.ncand)
    tracks = []
    fails = []
    for ci, (ph_h, f0, sv) in enumerate(cands):
        t_ph = ph_h * hop
        bl, dts, dfs, v = sync_scan_v(c, y, t_ph, f0, nblk, hop, rp.fsearch)
        if not bl:
            continue
        j = int(np.argmax(v.max(2).sum(0)))                  # 공통 시간 위치
        vb = np.nan_to_num(v[:, j, :])
        fb = f0 + ridge(c, dts[j]) + dfs[vb.argmax(1)]
        w = np.maximum(vb.max(1) - 1.0, 0.05)
        bb = np.array(bl, float)
        if len(bl) >= 3:
            A = np.stack([np.ones_like(bb), bb], 1) * np.sqrt(w)[:, None]
            coef = np.linalg.lstsq(A, fb * np.sqrt(w), rcond=None)[0]
            ffit = coef[0] + coef[1] * bb
        else:
            ffit = np.full(len(bl), np.average(fb, weights=w))
        ok_blocks = {}
        cc = c if (not c.msd or ci < rp.msd_top) else dataclasses.replace(c, msd=0)
        for i, b in enumerate(bl):
            t0 = t_ph + b * c.L + dts[j]
            for k_, fq in enumerate(dict.fromkeys([round(ffit[i], 3), round(fb[i], 3)])):
                ck = (abs0 + int(t0), round(float(fq), 2), k_ == 0 and cc.msd > 0)      # 스트리밍: 같은 절대 위치 · 주파수 복호는 다시 안 함
                if cache is not None and ck in cache:
                    info, ok, es = cache[ck]
                else:
                    info, ok, es = decode_at(cc if k_ == 0 else dataclasses.replace(cc, msd=0), y, t0, fq)   # 음 연결은 맞춤 주파수 한 번만
                    if cache is not None:
                        cache[ck] = (info, ok, es)
                if ok:
                    ok_blocks[t0] = (charset1.unpack_block(info)[0], fq, es, info, getattr(info, "first", False))
                    break
            if not ok and ci < rp.nstore and t0 >= 0 and t0 + c.L <= len(y):
                fails.append({"t": t0, "f": ffit[i], "S": float(vb[i].max()), "Sfold": sv,     # S = 이 블록 혼자의 동기 지표 (보관 순위)
                              "E": tone_energy(c, y, t0, ffit[i])})
        if ok_blocks:
            thr1 = (THR_TAB[K] if K in THR_TAB else THR_TAB[4] + 16.0 * (K - 4)) * c.NS / 12.0
            shown = len(ok_blocks) >= 2 or (rp.thr_single and sv >= thr1)
            tracks.append({"t": t_ph, "f": f0, "S": sv, "K": K, "blocks": ok_blocks, "shown": shown})
    return {"tracks": dedupe(c, tracks), "cands": cands, "K": K, "fails": fails, "S": S, "fr": fr, "hop": hop}


def dedupe(c, tracks):
    """같은 내용 블록이 여러 후보에서 풀리면 (시작 ±L/2) 동기 지표가 큰 후보 하나만 남김 (처프 변형은 시간 · 주파수가 묶여 흔함).
    표시 여부는 원래 판정을 유지 (남은 블록이 없으면 후보 제거)"""
    kept = []
    for tr in sorted(tracks, key=lambda t: -t["S"]):
        for tb in list(tr["blocks"]):
            info = tr["blocks"][tb][3]
            if any(abs(tb - t2) < c.L / 2 and np.array_equal(info, i2) for t2, i2 in kept):
                del tr["blocks"][tb]
            else:
                kept.append((tb, info))
    return [t for t in tracks if t["blocks"]]


# ------------------------------------------------------------------ 반복 합치기 (표시 없이, 송신 형식 그대로)
def copy_score(c, E, info, burst0=False):
    """정보 비트로 만든 부호어 (전송 순서 심볼) 가 이 사본 에너지와 맞는 정도: Σ(E[t, 부호어]/sig - 1)/√ND.
    잡음 · 다른 내용이면 약 N(0,1), 같은 내용이면 ND·es/√ND"""
    pos, tone, data_pos, _, ilv = lay(c)
    lab = trellis(c).encode(list(info) + crc16_bits(info, crc_init(c, getattr(info, "first", False))))[ilv]
    # 11부: 기호마다 그 기호 16음 평균으로 정규화 (전: 블록 전체 중앙값). 센 버스트 (다른 모드 신호 · 순간 잡음) 가 걸친 기호는
    # 16음이 모두 커서 어느 부호어든 점수가 커지던 것 → 중립 (ui_check_tng1 4번: TNG44 자리 사본 10.6 으로 가짜 짝 통과)
    Ed = E[data_pos]
    r = Ed / (Ed.mean(1, keepdims=True) + 1e-30)
    v = r[np.arange(len(data_pos)), lab] - 1.0
    if burst0:                                   # 15부: 버스트 심볼은 빼고 셈 (남은 심볼 수로 정규화)
        ok = ~burst_rows(c, E)[data_pos]
        return float(v[ok].sum() / math.sqrt(max(int(ok.sum()), 1)))
    return float(v.sum() / math.sqrt(c.ND))


def copy_sim(c, Ea, Eb):
    """두 사본이 같은 내용인지 미리 보기: 데이터 심볼 정규화 에너지 (E/sig - 1) 의 상관 × √N.
    내용이 다르거나 잡음이면 약 N(0,1), 같은 내용이면 대략 ND·es²/√(ND·M)"""
    _, _, data_pos, _, _ = lay(c)
    a = Ea[data_pos] / max(np.median(Ea) / math.log(2), 1e-30) - 1.0
    b = Eb[data_pos] / max(np.median(Eb) / math.log(2), 1e-30) - 1.0
    return float((a * b).sum() / math.sqrt((a * a).sum() * (b * b).sum() + 1e-30) * math.sqrt(a.size))


class Combiner:
    """CRC 실패 블록 (에너지) 을 보관해 새 실패 블록과 짝 (2개) · 셋 (3개) 으로 심볼 우도를 더해 복호.
    확정 = CRC16 통과 + 모든 사본이 각각 copy_score ≥ zmin. 같은 송신 (tx id) 안 블록끼리는 합치지 않음 (내용이 다름)"""

    def __init__(self, c, nmax=16, zmin=3.0, triples=True, simmin=2.0, maxpair=4):
        """simmin: 짝 미리 보기 문턱 (copy_sim). None = 안 봄. 셋은 모든 짝이 문턱을 넘을 때만"""
        self.c, self.nmax, self.zmin, self.triples, self.simmin, self.maxpair = c, nmax, zmin, triples, simmin, maxpair
        self.burst0 = True                       # 15부: 에너지가 튀는 심볼 가중 0 (ui_check_tng1 4번 가짜 짝)
        self.store = []
        self.ok_log = []
        self.tries = 0
        self.tried = []
        self.seen = []

    def _try(self, group):
        c = self.c
        Lm = sum(sym_metric(c, g["E"], burst0=self.burst0)[0] for g in group)
        self.tries += 1
        self.tried.append(tuple(g["t"] for g in group))          # 시험용 기록 (최근 200개)
        del self.tried[:-200]
        info, ok = decode_sym(c, Lm)
        if ok:
            sc = [copy_score(c, g["E"], info, burst0=self.burst0) for g in group]
            if all(v >= self.zmin for v in sc):
                self.ok_log.append({"pos": [g["t"] for g in group], "score": sc, "S": [g.get("S") for g in group]})   # 시험용 (최근 50)
                del self.ok_log[:-50]
                return info
        return None

    def add(self, fails, txid):
        """fails: receive()["fails"]. → 새로 확정된 [(정보 비트, 합친 사본 수)]"""
        out = []
        new = [dict(f, tx=txid) for f in fails]
        for f in new:
            for g in self.store:                                     # 시험용 기록: 보관소의 모든 짝 (위치 · 송신 id 같음 · 미리 보기)
                self.seen.append((f["t"], g["t"], g["tx"] == txid, copy_sim(self.c, f["E"], g["E"])))
            del self.seen[:-2000]
            # 짝 후보: 같은 자리 (같은 블록을 다른 후보가 다시 보관한 것, ±L/2) 만 뺌. 예전의 '같은 송신 id (격자 위상 1/8 · 5 Hz)
            # 는 안 합침' 규칙은 다시 보내기가 같은 위상에 오면 (약 1/8) 합치지 못해 뺐다 (09-26 5부). 다른 내용끼리는
            # 미리 보기 (copy_sim) · CRC16 · 사본별 copy_score 가 거름. 미리 보기가 큰 순서로 maxpair 개까지만 시도 (시도 수 상한)
            cand = [(copy_sim(self.c, f["E"], g["E"]), g) for g in self.store if abs(g["t"] - f["t"]) >= self.c.L / 2]
            if self.simmin is not None:
                cand = [x for x in cand if x[0] >= self.simmin]
            cand.sort(key=lambda x: -x[0])
            others = [g for _, g in cand[:self.maxpair]]
            got = None
            for g in others:
                got = self._try([f, g])
                if got is not None:
                    out.append((got, 2))
                    break
            if got is None and self.triples:
                for i in range(len(others)):
                    for j in range(i + 1, len(others)):
                        if abs(others[i]["t"] - others[j]["t"]) < self.c.L / 2:
                            continue
                        if self.simmin is not None and copy_sim(self.c, others[i]["E"], others[j]["E"]) < self.simmin:
                            continue
                        got = self._try([f, others[i], others[j]])
                        if got is not None:
                            out.append((got, 3))
                            break
                    if got is not None:
                        break
        # 보관: 동기 지표 (S) 가 큰 것부터 nmax 개 (연속 수신에서는 잡음 후보가 계속 들어오므로 도착 순서로 자르면
        # 진짜 신호 블록이 밀려남 — 09-26 3부 확인). 시간 상한은 Tng1Stream 이 자름
        self.store = sorted(self.store + new, key=lambda g: -g.get("S", 0.0))[:self.nmax]
        return out


# ------------------------------------------------------------------ 실시간 스트리밍 수신 (블록 단위)
class Tng1Stream:
    """복소 1 kHz 입력을 조금씩 받아 반 블록 (STEP) 마다 최근 (K+2) 블록 창을 처리한다.
    · 창 밖으로 나간 샘플은 버림 (메모리 고정), 복호 결과는 절대 위치 · 주파수로 캐시 (같은 블록 다시 복호 안 함)
    · 표시 = CRC16 + 표시 규칙 (2블록 또는 동기 문턱) 을 통과한 블록만, 절대 시작 ± L/2 · 같은 내용이면 한 번만
    · 반복 합치기: CRC 실패 블록 (상위 후보 nstore) 을 Combiner 에 (같은 격자 = 같은 송신 id, 한 블록은 한 번만)
    · 보관 상한: Combiner 16 블록 · HOLD_S 초 (오래된 것 버림), 캐시 · 표시 기록도 창 + HOLD_S 까지만
    step() → 새로 표시할 [{"abs", "f", "text", "info", "es", "n_comb", "eot"}] (abs = 입력 1 kHz 절대 샘플)"""
    HOLD_S = 600.0
    STORE_DEDUP = False                     # True = 같은 블록 재보관 막기 (반 기호 · 3 Hz, 9부 시험: 보관 · 시도 수 안 줄어 끔). False = 예전 100 샘플 칸

    def __init__(self, c=None, rp=None):
        self.c = c or CFGS["TNG1"]
        self.rp = rp or RxParams()
        self.win = (self.rp.K + 2) * self.c.L
        self.step_n = self.c.L // 2
        self.z = np.zeros(0, complex)
        self.base = 0                       # self.z[0] 의 절대 번호
        self.last = 0                       # 마지막 처리 때의 끝 (절대)
        self.last_reg = 0                   # 마지막 정규 (반 블록) 처리 때의 끝 — 보관 · 짝 시도 박자
        self.cache = {}
        self.shown = []                     # [(abs, info bytes)]
        self.stored = []                    # 보관한 실패 블록 [(절대 위치, 주파수, 동기 지표)] — 같은 블록 재보관 막기
        self.comb = Combiner(self.c, zmin=5.0)     # 사본별 일치 점수 문턱 3 → 5 (09-26 9부): 잡음 사본은 약 N(0,1) → 가짜 통과 0.13% → 3e-7 / 시도,
                                                   # 진짜 사본은 -24 dB 약 14 · -27 dB 약 7 (계산)
        self.cpu = 0.0
        self.fold = None                    # 전문가 창: 마지막 동기 누적 지도 {S, fr, hop, K, cands, base}
        self.calls = 0

    @property
    def end(self):
        return self.base + len(self.z)

    def push(self, zc):
        self.z = np.concatenate([self.z, np.asarray(zc, complex)])
        if len(self.z) > self.win:
            cut = len(self.z) - self.win
            self.z = self.z[cut:]
            self.base += cut

    def skip(self, n):
        """n 샘플을 신호 없이 건너뜀 (수신 멈춤 동안): 창에는 0 을 넣고 절대 시각은 그대로 이어짐"""
        n = int(n)
        k = min(n, self.win)
        self.base += n - k
        self.push(np.zeros(k, complex))
        self.last = self.last_reg = self.end

    def step(self, force=False):
        """반 블록마다 (force = 블록 끝 직후 바로). 입력 창 (z, base) 은 시작 때 한 번 잡아 끝까지 씀 —
        앱은 이 계산을 수신 스레드의 잠금 밖에서 돌리므로 그 사이 push · skip 이 self.z 를 바꿔도 이 처리는 같은 창을 본다"""
        z, base = self.z, self.base
        end = base + len(z)
        # 정규 처리 = 반 블록마다 (6부 전과 같은 박자, 즉시 처리와 따로 셈). 반복 합치기용 실패 블록 보관 · 짝 시도는 정규 처리에서만 —
        # 블록 끝 즉시 처리 · 엿보기 처리 (force) 는 화면 표시용. 6부 뒤 즉시 처리마다 보관 · 짝 시도를 해서 합친 복호 시도가
        # 잡음 10분 295 → 약 1400 번으로 늘어 (9부 측정) 가짜 결합 기회 · 대기 CPU 가 함께 늘었다 (10부)
        regular = end - self.last_reg >= self.step_n
        if not force and not regular:
            return []
        if len(z) < self.c.L:
            return []
        import time as _t
        t0 = _t.perf_counter()
        self.last = end
        if regular:
            self.last_reg = end
        rp = self.rp if regular else dataclasses.replace(self.rp, nstore=0)
        r = receive(self.c, z, rp, abs0=base, cache=self.cache)
        if r.get("S") is not None:
            self.fold = {"S": r["S"], "fr": r["fr"], "hop": r["hop"], "K": r["K"], "cands": r["cands"], "base": base}
        out = []
        for tr in r["tracks"]:
            if not tr["shown"]:
                continue
            for tb, v in sorted(tr["blocks"].items()):
                a = base + tb
                key = np.asarray(v[3], np.int8).tobytes()
                if any(abs(a - a2) < self.c.L / 2 and k2 == key for a2, k2 in self.shown):
                    continue
                self.shown.append((a, key))
                txt, eot = charset1.unpack_block(v[3])
                out.append({"abs": a, "f": v[1], "text": txt, "eot": eot, "info": v[3], "es": v[2], "n_comb": 1,
                            "E": tone_energy(self.c, z, tb, v[1]),
                            "first": bool(v[4]) if len(v) > 4 else False})
        new_fails = []
        for f in r.get("fails", []):
            a = base + f["t"]
            if any(abs(a - a2) < self.c.L / 2 for a2, _ in self.shown):
                continue
            # 같은 블록 (반 기호 · 3 Hz 안) 은 한 번만 보관. 더 센 (동기 지표) 것이 오면 보관본만 바꿈 (새 짝 시도 없음).
            # 전: 100 샘플 칸 키라 처리 시점마다 격자가 조금씩 다른 같은 블록이 여러 번 보관 → 합친 복호 시도 4.7배 (6부 뒤, 측정) →
            # 잡음 자리 사본과 옛 진짜 사본의 가짜 짝 통과 기회도 그만큼 (ui_check_tng1 4번)
            same = [i for i, (a2, f2, _) in enumerate(self.stored) if abs(a - a2) < self.c.Ns / 2 and abs(f["f"] - f2) < 3.0]                 if self.STORE_DEDUP else [i for i, (a2, _, _) in enumerate(self.stored) if int(round(a / 100.0)) == int(round(a2 / 100.0))]
            if same:
                i = same[0]
                if f["S"] > self.stored[i][2]:
                    old_a = self.stored[i][0]
                    self.stored[i] = (a, f["f"], f["S"])
                    for g in self.comb.store:
                        if g["t"] == old_a:
                            g.update(t=a, f=f["f"], S=f["S"], E=f["E"])
                continue
            self.stored.append((a, f["f"], f["S"]))
            ph = int(round((a % self.c.L) / (self.c.L / 8.0))) % 8
            new_fails.append(dict(f, t=a, tx=(ph, int(round(f["f"] / 5.0)))))
        for f in new_fails:                                   # 송신 id (격자 위상 · 주파수) 별로 넣음
            for info, n in self.comb.add([f], f["tx"]):
                # 11부: 결합 블록 자리 = 짝 가운데 가장 새 사본 (전: 새로 보관한 사본 f — 옛 사본이 늦게 보관되면 옛 자리로 나옴)
                pos = self.comb.ok_log[-1]["pos"] if self.comb.ok_log else [f["t"]]
                t_new = max(pos)
                g_new = next((g for g in [f] + self.comb.store if g["t"] == t_new), f)
                key = np.asarray(info, np.int8).tobytes()
                if any(abs(t_new - a2) < self.c.L / 2 and k2 == key for a2, k2 in self.shown):
                    continue
                # 늦은 결합 (7부 결정 · 11부): 같은 내용의 더 새 송신이 이미 있음 = 표시된 같은 내용이 더 뒤에 있거나,
                # 보관 사본 가운데 더 뒤 자리 것이 이 내용과 맞음 (copy_score ≥ zmin). 다시 보내기의 정상 결합은 가장 새 사본이라 해당 없음
                late = any(a2 > t_new + self.c.L / 2 and k2 == key for a2, k2 in self.shown) or                     any(g["t"] > t_new + self.c.L / 2 and copy_score(self.c, g["E"], info) >= self.comb.zmin for g in self.comb.store)
                self.shown.append((t_new, key))
                txt, eot = charset1.unpack_block(info)
                out.append({"abs": t_new, "f": g_new["f"], "text": txt, "eot": eot, "info": info, "es": None, "n_comb": n, "E": g_new["E"],
                            "first": bool(getattr(info, "first", False)), "late": late})
        # 보관 상한 (시간)
        lim = end - int(self.HOLD_S * FS)
        self.comb.store = [g for g in self.comb.store if g["t"] >= lim]
        self.shown = [x for x in self.shown if x[0] >= lim]
        self.stored = [x for x in self.stored if x[0] >= lim]
        self.cache = {k: v for k, v in self.cache.items() if k[0] >= base - self.c.L}
        self.cpu += _t.perf_counter() - t0
        self.calls += 1
        return sorted(out, key=lambda e: e["abs"])
