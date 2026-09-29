"""TNG5 앱에서 사용하는 기존 K=7 부호화 · 연판정 비터비. 배포판에서는 시험 실행부만 제외."""
import numpy as np

GENS = {2: (0o171, 0o133), 3: (0o133, 0o165, 0o171), 4: (0o117, 0o127, 0o155, 0o171)}   # K=7 표준 · ODP 생성다항식
K = 7
NS = 1 << (K - 1)


def parity(v):
    return bin(v).count("1") & 1


def tables(n):
    """상태 s (최근 6비트), 입력 u → 다음 상태, 출력 비트 (n,)"""
    nxt = np.zeros((NS, 2), int)
    out = np.zeros((NS, 2, n), int)
    for s in range(NS):
        for u in range(2):
            reg = (u << (K - 1)) | s
            nxt[s, u] = reg >> 1
            out[s, u] = [parity(reg & g) for g in GENS[n]]
    return nxt, out


def encode(bits, n):
    nxt, out = tables(n)
    s, c = 0, []
    for u in bits:
        c.append(out[s, u])
        s = nxt[s, u]
    return np.concatenate(c)


def viterbi(llr, n, T):
    """llr: (B, T*n) — 양수 = 1. 반환 (B, T) 추정 비트 (꼬리 포함, 끝 상태 0)"""
    nxt, out = tables(n)
    B = llr.shape[0]
    sgn = 2.0 * out - 1.0                                   # (NS, 2, n)
    pm = np.full((B, NS), -1e18)
    pm[:, 0] = 0.0
    prev_s = np.zeros((NS, 2), int)
    # 다음 상태 t 로 들어오는 (이전 상태, 입력) 두 쌍
    into = [[] for _ in range(NS)]
    for s in range(NS):
        for u in range(2):
            into[nxt[s, u]].append((s, u))
    src = np.array([[p[0] for p in into[t]] for t in range(NS)])   # (NS, 2)
    inp = np.array([[p[1] for p in into[t]] for t in range(NS)])
    dec = np.zeros((T, B, NS), np.int8)
    L3 = llr.reshape(B, T, n)
    for k in range(T):
        bm = np.einsum("bn,sun->bsu", L3[:, k], sgn)        # (B, NS, 2)
        cand = pm[:, src] + bm[:, src, inp]                 # (B, NS, 2)
        c = np.argmax(cand, axis=2)
        dec[k] = c
        pm = np.take_along_axis(cand, c[..., None], 2)[..., 0]
    s = np.zeros(B, int)
    bits = np.zeros((B, T), np.int8)
    for k in range(T - 1, -1, -1):
        c = dec[k, np.arange(B), s]
        bits[:, k] = inp[s, c]
        s = src[s, c]
    return bits


