"""
오류정정부호 (FEC) 와 인터리버.

  * 부호   : 구속장 K=7, 부호율 1/2 컨볼루션 부호 (G1=0o133, G2=0o171)
             — HF 디지털 모드에서 널리 쓰이는 고전적인 선택
  * 복호   : soft-decision 비터비. 수신 NN 이 내놓는 로짓을 **그대로 LLR 로** 받는다.
             (BCE 로 학습했으므로 로짓이 곧 log P(b=1)/P(b=0) 이다)
  * 인터리버: 고정 시드 의사난수 치환. 다중경로로 한 구간이 통째로 망가져도
             그 오류가 부호어 전체에 흩어지도록 한다.

비터비는 **여러 후보를 한꺼번에** 돌릴 수 있게 배치로 구현했다.
프레임 정렬 후보 수백 개를 동시에 복호해 경로 metric 이 가장 큰 것을 고르는 데 쓴다.
"""

import numpy as np

K = 7                      # 구속장
M = K - 1                  # 메모리 (상태 비트 수) = 6
N_STATES = 1 << M          # 64
G1, G2 = 0o133, 0o171
TAIL = M                   # 종단용 꼬리 비트 수
INTERLEAVER_SEED = 0x5EED


def _parity(x: int) -> int:
    return bin(x).count("1") & 1


# --- 격자(trellis) 표: 상태 s 에서 입력 u 일 때의 다음 상태와 출력 2비트 ---
_NEXT = np.zeros((N_STATES, 2), dtype=np.int64)
_OUT0 = np.zeros((N_STATES, 2), dtype=np.float32)
_OUT1 = np.zeros((N_STATES, 2), dtype=np.float32)
for _s in range(N_STATES):
    for _u in (0, 1):
        _r = (_u << M) | _s                     # 레지스터 = [새 비트, 이전 6비트]
        _NEXT[_s, _u] = (_r >> 1) & (N_STATES - 1)
        _OUT0[_s, _u] = _parity(_r & G1)
        _OUT1[_s, _u] = _parity(_r & G2)

# 다음 상태 ns 로 들어오는 두 개의 이전 상태와 그때의 입력 비트
_NS = np.arange(N_STATES)
_U_OF_NS = (_NS >> (M - 1)) & 1                 # ns 의 최상위 비트 = 그 단계의 입력
_PREV = np.stack([2 * (_NS & ((N_STATES >> 1) - 1)),
                  2 * (_NS & ((N_STATES >> 1) - 1)) + 1], axis=1)   # (64, 2)


def conv_encode(bits: np.ndarray) -> np.ndarray:
    """정보 비트 (n,) → 부호 비트 (2*(n+6),). 꼬리 비트로 상태를 0 으로 종단한다."""
    b = np.concatenate([np.asarray(bits, dtype=np.int64).ravel(), np.zeros(TAIL, dtype=np.int64)])
    out = np.empty(2 * len(b), dtype=np.uint8)
    s = 0
    for i, u in enumerate(b):
        out[2 * i] = _OUT0[s, u]
        out[2 * i + 1] = _OUT1[s, u]
        s = _NEXT[s, u]
    return out


def viterbi_metrics(llr: np.ndarray) -> np.ndarray:
    """
    여러 후보의 최종 경로 metric 만 빠르게 구한다 (역추적 없음).

    llr: (B, 2*T) — 부호 비트의 LLR (양수면 1). B 개 후보를 동시에 처리.
    반환: (B,) — 상태 0 으로 종단했을 때의 경로 metric. 클수록 그럴듯하다.
    """
    llr = np.asarray(llr, dtype=np.float32)
    B, n = llr.shape
    T = n // 2
    l0 = llr[:, 0::2]
    l1 = llr[:, 1::2]

    met = np.full((B, N_STATES), -1e30, dtype=np.float32)
    met[:, 0] = 0.0
    for t in range(T):
        # 가지 metric: 출력이 1 인 비트마다 해당 LLR 을 더한다
        bm = (_OUT0[:, :] * l0[:, t, None, None] + _OUT1[:, :] * l1[:, t, None, None])  # (B,64,2)
        u = _U_OF_NS                                       # (64,)
        c0 = met[:, _PREV[:, 0]] + bm[:, _PREV[:, 0], u]   # (B, 64)
        c1 = met[:, _PREV[:, 1]] + bm[:, _PREV[:, 1], u]
        met = np.maximum(c0, c1)
    return met[:, 0]


_TORCH_TABLES = {}


def viterbi_metrics_torch(llr, device="cuda"):
    """
    viterbi_metrics 와 같은 계산을 GPU 에서 배치로 한다 (정렬 후보 수백~수천 개를 한 번에).
    numpy 판과 결과가 같은지는 fec.py 자가 검사에서 확인한다.
    """
    import torch
    dev = torch.device(device)
    if dev not in _TORCH_TABLES:
        p0, p1 = _PREV[:, 0], _PREV[:, 1]
        u = _U_OF_NS
        t = lambda a: torch.as_tensor(np.asarray(a), device=dev)
        _TORCH_TABLES[dev] = (t(p0), t(p1),
                              t(_OUT0[p0, u]).float(), t(_OUT1[p0, u]).float(),
                              t(_OUT0[p1, u]).float(), t(_OUT1[p1, u]).float())
    P0, P1, o00, o01, o10, o11 = _TORCH_TABLES[dev]
    x = torch.as_tensor(np.asarray(llr, dtype=np.float32), device=dev)
    B, n = x.shape
    l0, l1 = x[:, 0::2], x[:, 1::2]
    met = torch.full((B, N_STATES), -1e30, device=dev)
    met[:, 0] = 0.0
    with torch.no_grad():
        for k in range(n // 2):
            a, b = l0[:, k:k + 1], l1[:, k:k + 1]
            c0 = met[:, P0] + o00 * a + o01 * b
            c1 = met[:, P1] + o10 * a + o11 * b
            met = torch.maximum(c0, c1)
    return met[:, 0].cpu().numpy()


USE_GPU = False              # 28부: 앱은 GPU 안 씀 (연구 스크립트가 필요하면 True)


def viterbi_metrics_auto(llr):
    """후보가 많고 USE_GPU 면 GPU, 아니면 numpy."""
    llr = np.asarray(llr, dtype=np.float32)
    if USE_GPU and llr.shape[0] >= 64:
        try:
            import torch
            if torch.cuda.is_available():
                return viterbi_metrics_torch(llr, "cuda")
        except Exception:
            pass
    return viterbi_metrics(llr)


def viterbi_decode(llr: np.ndarray, n_info: int) -> np.ndarray:
    """
    soft-decision 비터비 복호 (단일 후보).

    llr: (2*T,) 부호 비트 LLR,  n_info: 꼬리를 뺀 정보 비트 수
    반환: (n_info,) 0/1
    """
    llr = np.asarray(llr, dtype=np.float32).ravel()
    T = len(llr) // 2
    l0, l1 = llr[0::2], llr[1::2]

    met = np.full(N_STATES, -1e30, dtype=np.float32)
    met[0] = 0.0
    back = np.zeros((T, N_STATES), dtype=np.uint8)
    for t in range(T):
        bm = _OUT0 * l0[t] + _OUT1 * l1[t]                 # (64, 2)
        u = _U_OF_NS
        c0 = met[_PREV[:, 0]] + bm[_PREV[:, 0], u]
        c1 = met[_PREV[:, 1]] + bm[_PREV[:, 1], u]
        back[t] = (c1 > c0).astype(np.uint8)
        met = np.maximum(c0, c1)

    s = 0                                                   # 꼬리 비트로 0 에 종단됨
    bits = np.zeros(T, dtype=np.uint8)
    for t in range(T - 1, -1, -1):
        bits[t] = (s >> (M - 1)) & 1
        s = _PREV[s, back[t, s]]
    return bits[:n_info]


# ------------------------------------------------------------------ 인터리버
_CACHE = {}


def interleaver(n: int) -> np.ndarray:
    """길이 n 의 고정 치환 (송수신이 같은 값을 만든다)"""
    if n not in _CACHE:
        _CACHE[n] = np.random.default_rng(INTERLEAVER_SEED + n).permutation(n)
    return _CACHE[n]


def interleave(x: np.ndarray) -> np.ndarray:
    return np.asarray(x)[interleaver(len(x))]


def deinterleave(y: np.ndarray) -> np.ndarray:
    p = interleaver(y.shape[-1])
    out = np.empty_like(y)
    out[..., p] = y
    return out


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    info = rng.integers(0, 2, 200)
    cw = conv_encode(info)
    assert len(cw) == 2 * (200 + TAIL)

    # 잡음 없는 경우 완전 복원
    llr = (2.0 * cw - 1.0) * 10.0
    assert np.array_equal(viterbi_decode(llr, 200), info), "무잡음 복호 실패"

    # 인터리버 왕복
    assert np.array_equal(deinterleave(interleave(cw)), cw), "인터리버 왕복 실패"

    # 오류 정정 능력 확인
    for p in (0.02, 0.05, 0.08, 0.11):
        nerr = 0
        for _ in range(20):
            info = rng.integers(0, 2, 200)
            cw = conv_encode(info)
            rx = cw ^ (rng.random(len(cw)) < p)
            llr = (2.0 * rx - 1.0) * 2.0          # hard 값을 LLR 로 (soft 이득 없음)
            nerr += int(not np.array_equal(viterbi_decode(llr, 200), info))
        print("  부호비트 오류율 {:.0%} → 블록 복호 실패 {}/20".format(p, nerr))

    # 배치 metric 과 단일 복호의 정렬 판별력
    info = rng.integers(0, 2, 200)
    cw = conv_encode(info)
    good = (2.0 * cw - 1.0) * 3.0
    bad = rng.normal(0, 3.0, len(cw)).astype(np.float32)
    mets = viterbi_metrics(np.stack([good, bad, np.roll(good, 7)]))
    print("  경로 metric: 정답 {:.0f} / 무작위 {:.0f} / 어긋남 {:.0f}".format(*mets))
    assert mets[0] > mets[1] and mets[0] > mets[2], "정렬 판별 실패"

    # GPU 배치 비터비가 numpy 판과 같은 값을 내는지
    try:
        import torch
        if torch.cuda.is_available():
            big = rng.normal(0, 3.0, (300, len(cw))).astype(np.float32)
            a, b = viterbi_metrics(big), viterbi_metrics_torch(big)
            print("  GPU 배치 비터비: numpy 와 최대 차이 {:.2e} (값 크기 약 {:.0f})".format(
                np.max(np.abs(a - b)), np.mean(np.abs(a))))
            assert np.allclose(a, b, rtol=1e-4, atol=1e-2), "GPU 비터비 불일치"
    except ImportError:
        pass
    print("fec 자가 검사 통과")
