"""
7단계 링크 계층 — 스트리밍 부호화 (긴 메시지용). 근거는 README '7단계' 와 results/longmsg_eval.log.

송신
    페이로드(UTF-8) 를 32바이트 구간으로 나눠 구간마다 CRC16 을 붙이고,
    [구간0][CRC][구간1][CRC]... 전체를 K=7 R=1/2 컨볼루션 부호로 **한 번에** 부호화한다 (꼬리는 끝에 한 번).
    부호 비트는 지연순 정렬 인터리버(가지 12, 지연 단위 24, 최대 이동 약 3.2초)를 거쳐 96비트 프레임으로 나간다.
    마지막 구간은 NUL 로 채워 프레임 경계까지 쓴다 → 수신측은 프레임 수만으로 구조 전체를 안다.
    가드 프레임은 4단계 가드 패턴의 **비트 반전**을 쓴다 → 수신측이 형식(단일 블록/스트리밍)을 가려낸다.

  v2 (9단계, 기본): 첫 구간 앞 2바이트에 메시지 길이(페이로드 바이트 수, 빅엔디언)를 넣는다.
    수신측은 첫 구간만 통과하면 전체 구간 수와 끝을 미리 안다 (코스타스 B 를 못 받아도 끝낼 수 있다).
    가드 = 4단계 패턴의 홀수 번째 비트만 반전 (48비트) → 단일 블록 · v1 가드와 상관이 0 이라 셋이 구별된다.
    길이가 코스타스 B 로 정해지는 프레임 수보다 크면 송신 중지된 메시지다 (B 우선, 길이는 상한).

수신 (StreamDecoder)
    정렬 가설 19개(-18..+18샘플, 2샘플 간격)마다 비터비를 이어서 돌린다.
    구간이 끝나고 48정보비트(제한 깊이) 가 더 지나면 그 구간을 결정한다:
      가설마다 역추적 → 재부호화 → 구간 정규화 로그우도 점수 → 직전 정렬 ±4샘플 안에서 최대 점수 가설 → CRC16.
      연속 2구간 실패하면 다음 구간은 전 범위(±18)에서 고른다 (잠금 상실 대책).
    프레임을 도착하는 대로 feed() 하면 구간 단위로 결과가 나온다. 전부 받은 뒤 finish() 하면 끝까지 복호한다.
"""

import numpy as np

from fec import TAIL, M, N_STATES, G1, G2, _PREV, _U_OF_NS, _OUT0, _OUT1
from link4 import GUARD_BITS, BITS_PER_FRAME
from framing import crc16_ccitt

SEG_BYTES = 32
CRC_BYTES = 2
SEG_INFO_BITS = 8 * (SEG_BYTES + CRC_BYTES)
ILV_J, ILV_M = 12, 24                  # 지연순 정렬 인터리버: 최대 이동 약 1584비트 = 3.2초
GUARD_STREAM = (1.0 - GUARD_BITS).astype(np.float32)             # v1 (7단계)
GUARD_V2 = np.abs(GUARD_BITS - np.tile([0.0, 1.0], 48)).astype(np.float32)   # v2: 절반 비트 반전
GUARDS = {"block": GUARD_BITS, "stream1": GUARD_STREAM, "stream2": GUARD_V2}
LEN_BYTES = 2
OFFSETS = tuple(range(-18, 19, 2))     # 정렬 가설 [샘플 @2000Hz]
TRACK_WINDOW = 4                       # 순차 추적 창 ±샘플
RELOCK_N = 2                           # 연속 실패 이만큼이면 전 범위 재탐색
DEPTH = 48                             # 제한 깊이 [정보비트]. 측정: 전체 대비 손실 0.01자/s 이하
PLACEHOLDER = "□"                      # 실패 구간 자리 표시 (부분 텍스트용)


# ------------------------------------------------------------------ 구조
def layout(nbytes):
    """페이로드 nbytes → [(바이트 시작, 바이트 수)]"""
    return [(s, min(SEG_BYTES, nbytes - s)) for s in range(0, nbytes, SEG_BYTES)]


def coded_len(nbytes):
    return 2 * (sum(8 * (n + CRC_BYTES) for _, n in layout(nbytes)) + TAIL)


def frames_for(nbytes):
    return int(np.ceil(coded_len(nbytes) / BITS_PER_FRAME))


def capacity(n_frames):
    """프레임 n_frames 개에 들어가는 최대 페이로드 바이트 (송신은 이 길이까지 NUL 로 채운다)"""
    lim = n_frames * BITS_PER_FRAME
    n = max(1, (lim // 2 - TAIL) // 8)            # 위에서부터 내려오며 찾는다
    while n > 1 and coded_len(n) > lim:
        n -= 1
    return n if coded_len(n) <= lim else 0


_ILV = {}


def interleaver_src(n):
    """
    지연순 정렬 인터리버. 부호 비트 p 의 공칭 시각 tau = p + (p mod J)*J*M 순서로 내보낸다.
    반환 q: 부호 비트 p 가 송신열 q[p] 자리로 간다 (수신: c = y[q]).
    끝 이외의 송신 순서는 n 과 무관하다 (tau 가 n 보다 작은 비트끼리의 순서라서).
    """
    if n not in _ILV:
        p = np.arange(n)
        tau = p + (p % ILV_J) * ILV_J * ILV_M
        order = np.lexsort((p, tau))
        q = np.empty(n, dtype=np.int64)
        q[order] = np.arange(n)
        _ILV[n] = q
    return _ILV[n]


def conv_encode_rows(u):
    """(R, n) → (R, 2n), 시작 상태 0. fec.conv_encode 와 같은 부호 (link7 자가검사에서 확인)"""
    u = np.atleast_2d(np.asarray(u, dtype=np.int64))
    R, n = u.shape
    up = np.concatenate([np.zeros((R, M), dtype=np.int64), u], axis=1)
    c0 = np.zeros((R, n), dtype=np.int64)
    c1 = np.zeros((R, n), dtype=np.int64)
    for i in range(M + 1):
        sh = up[:, i:i + n]
        if (G1 >> i) & 1:
            c0 ^= sh
        if (G2 >> i) & 1:
            c1 ^= sh
    out = np.empty((R, 2 * n), dtype=np.int64)
    out[:, 0::2], out[:, 1::2] = c0, c1
    return out


def _bits(b):
    return np.unpackbits(np.frombuffer(bytes(b), dtype=np.uint8)).astype(np.int64)


# ------------------------------------------------------------------ 송신
def stream_bytes(P, version=2, declared=None):
    """페이로드 → 구간에 실을 바이트열. v2 = 길이 2바이트 + 페이로드 (declared: 송신 중지 때 원래 길이)"""
    if version == 2:
        n = len(P) if declared is None else declared
        return int(n).to_bytes(LEN_BYTES, "big") + P
    return P


def encode_stream(text, version=2):
    """텍스트 → (데이터 프레임 비트 (n_frames, 96), 정보 dict). 실제 송신은 [가드] + 프레임들 + [가드]."""
    P = text.replace("\x00", "").encode("utf-8")
    if not P:
        P = b" "
    frames, info = encode_raw(stream_bytes(P, version))
    info.update({"payload": len(P), "version": version})
    return frames, info


def encode_raw(P):
    """구간 바이트열(v2 는 길이 포함) → 프레임. 마지막 구간을 NUL 로 프레임 경계까지 채운다."""
    nf = frames_for(len(P))
    cap = capacity(nf)
    Pp = P + b"\x00" * (cap - len(P))
    info = []
    for s, n in layout(len(Pp)):
        d = Pp[s:s + n]
        info += [_bits(d), _bits(crc16_ccitt(d).to_bytes(CRC_BYTES, "big"))]
    info = np.concatenate(info + [np.zeros(TAIL, np.int64)])
    c = conv_encode_rows(info)[0]
    q = interleaver_src(len(c))
    y = np.empty_like(c)
    y[q] = c
    fill = np.random.default_rng(0xF111).integers(0, 2, nf * BITS_PER_FRAME - len(y))
    frames = np.concatenate([y, fill]).reshape(nf, BITS_PER_FRAME).astype(np.float32)
    return frames, {"n_data": nf, "payload": len(P), "capacity": cap,
                    "n_segments": len(layout(cap)), "n_coded": len(c)}


# ------------------------------------------------------------------ 수신
class StreamDecoder:
    """
    정렬 가설 H 개에 대한 연속 비터비 + 구간별 순차 추적.
    feed(llr) : llr (H, k, 96) — 도착한 프레임 k 개의 가설별 LLR (가설 순서 = offsets)
    finish()  : 모든 프레임을 받은 뒤 끝까지 복호 (꼬리로 상태 0 종단)
    segments  : 결정된 구간 목록 [{'index','ok','data','hyp','offset','score','relock'}]
    """

    def __init__(self, n_frames, offsets=OFFSETS, depth=DEPTH, window=TRACK_WINDOW, relock_n=RELOCK_N):
        self.nf = int(n_frames)
        self.cap = capacity(self.nf)
        if self.cap <= 0:
            raise ValueError("프레임 수가 너무 적다: {}".format(n_frames))
        self.lay = layout(self.cap)
        self.n_coded = coded_len(self.cap)
        self.T = self.n_coded // 2
        self.q = interleaver_src(self.n_coded)
        need = np.maximum(self.q[0::2], self.q[1::2]) // BITS_PER_FRAME
        self.need = np.maximum.accumulate(need)          # 단계 t 에 필요한 마지막 프레임 번호
        self.offs = np.asarray(offsets)
        self.H = len(self.offs)
        self.depth, self.window, self.relock_n = depth, window, relock_n
        self.y = np.zeros((self.H, self.nf * BITS_PER_FRAME), dtype=np.float32)
        self.n_rx = 0
        self.t = 0
        self.met = np.full((self.H, N_STATES), -1e30, dtype=np.float64)
        self.met[:, 0] = 0.0
        self.dec = np.zeros((self.T, self.H, N_STATES), dtype=bool)
        p0, p1, u = _PREV[:, 0], _PREV[:, 1], _U_OF_NS
        self._P0, self._P1 = p0, p1
        self._o = (_OUT0[p0, u], _OUT1[p0, u], _OUT0[p1, u], _OUT1[p1, u])
        self.seg_bits = []                                  # (정보비트 시작, 끝, 바이트 시작, 바이트 수)
        ib = 0
        for s, n in self.lay:
            self.seg_bits.append((ib, ib + 8 * (n + CRC_BYTES), s, n))
            ib += 8 * (n + CRC_BYTES)
        self.next_seg = 0
        self.center = 0.0
        self.fails = 0
        self.segments = []
        self.relocks = 0

    # ---------------------------------------------------------- 입력
    def feed(self, llr):
        llr = np.asarray(llr, dtype=np.float32)
        k = llr.shape[1]
        if self.n_rx + k > self.nf:
            raise ValueError("프레임이 너무 많다")
        self.y[:, self.n_rx * BITS_PER_FRAME:(self.n_rx + k) * BITS_PER_FRAME] = llr.reshape(self.H, -1)
        self.n_rx += k
        self._advance()
        return self._emit(final=False)

    def finish(self):
        if self.n_rx < self.nf:
            raise ValueError("프레임을 다 받지 못했다 ({}/{})".format(self.n_rx, self.nf))
        self._advance()
        return self._emit(final=True)

    # ---------------------------------------------------------- 비터비
    def _advance(self):
        o00, o01, o10, o11 = self._o
        P0, P1 = self._P0, self._P1
        q = self.q
        while self.t < self.T and self.need[self.t] < self.n_rx:
            t = self.t
            l0 = self.y[:, q[2 * t]][:, None]
            l1 = self.y[:, q[2 * t + 1]][:, None]
            c0 = self.met[:, P0] + o00 * l0 + o01 * l1
            c1 = self.met[:, P1] + o10 * l0 + o11 * l1
            d = c1 > c0
            self.dec[t] = d
            m = np.where(d, c1, c0)
            self.met = m - m.max(axis=1, keepdims=True)
            self.t += 1

    def _traceback(self, t_end, state, t_stop):
        """단계 t_end 의 상태 state (H,) 에서 거슬러 올라가 정보비트 [t_stop, t_end) (H, n)"""
        H = self.H
        ar = np.arange(H)
        s = state.copy()
        out = np.empty((H, t_end - t_stop), dtype=np.int64)
        for k in range(t_end - 1, t_stop - 1, -1):
            out[:, k - t_stop] = (s >> (M - 1)) & 1
            s = np.where(self.dec[k][ar, s], self._P1[s], self._P0[s])
        return out

    # ---------------------------------------------------------- 구간 결정
    def _emit(self, final):
        done = final and self.t == self.T
        ready = []
        while self.next_seg < len(self.seg_bits):
            a, e, s, n = self.seg_bits[self.next_seg]
            if not (done or self.t >= min(e + self.depth, self.T)):
                break
            ready.append(self.next_seg)
            self.next_seg += 1
        if not ready:
            return []
        a0 = max(0, self.seg_bits[ready[0]][0] - M)
        state = (np.zeros(self.H, dtype=np.int64) if done else self.met.argmax(axis=1))
        bits = self._traceback(self.t, state, a0)                 # (H, t - a0)
        out = []
        for i in ready:
            a, e, s, n = self.seg_bits[i]
            ctx = bits[:, max(0, a - M) - a0:e - a0]
            re = conv_encode_rows(ctx)[:, 2 * (a - max(0, a - M)):]    # 이 구간 단계의 부호 비트
            L = self.y[:, self.q[2 * a:2 * e]]
            score = (re * L - np.logaddexp(0.0, L)).mean(axis=1)
            relock = self.fails >= self.relock_n
            if relock:
                cand = np.ones(self.H, dtype=bool)
                self.relocks += 1
            else:
                cand = np.abs(self.offs - self.center) <= self.window
            h = int(np.argmax(np.where(cand, score, -np.inf)))
            seg = bits[h, a - a0:e - a0]
            byt = np.packbits(seg.astype(np.uint8)).tobytes()
            data, crc_rx = byt[:n], int.from_bytes(byt[n:n + CRC_BYTES], "big")
            ok = crc16_ccitt(data) == crc_rx
            if ok:
                self.center = float(self.offs[h])
                self.fails = 0
            else:
                self.fails += 1
            r = {"index": i, "ok": ok, "data": data if ok else None, "hyp": h, "offset": int(self.offs[h]),
                 "score": float(score[h]), "relock": bool(relock), "bytes": (s, n)}
            self.segments.append(r)
            out.append(r)
        return out

    # ---------------------------------------------------------- 결과
    def result(self, version=1):
        """
        (완전 텍스트 또는 None, 부분 텍스트, 정보).
        v2: 첫 구간이 통과하면 길이로 필요한 구간 수를 안다. 필요한 구간 수가 이 메시지(프레임 수) 보다 많으면
            송신 중지된 메시지 (info['stopped']). 첫 구간이 실패하면 v1 처럼 NUL 로 끝을 찾는다.
        """
        segs = sorted(self.segments, key=lambda r: r["index"])
        ok = [r["ok"] for r in segs]
        n_av = len(self.seg_bits)
        L = None
        if version == 2 and segs and segs[0]["index"] == 0 and segs[0]["ok"]:
            L = int.from_bytes(segs[0]["data"][:LEN_BYTES], "big")
        stopped, need = False, None
        if L is not None:
            need = -(-(LEN_BYTES + L) // SEG_BYTES)
            stopped = need > n_av
            end_seg = min(need, len(segs))
            complete = (not stopped) and len(segs) >= need and all(ok[:need])
        else:
            # 끝의 NUL: 통과한 마지막 구간에 NUL 이 있으면 그 뒤는 채움뿐이다
            end_seg = len(segs)
            for r in segs:
                if r["ok"] and b"\x00" in r["data"][(LEN_BYTES if version == 2 and r["index"] == 0 else 0):]:
                    end_seg = r["index"] + 1
                    break
            complete = len(segs) == n_av and all(ok[:end_seg]) and version == 1
        # 페이로드 조각 (v2: 첫 구간 앞 2바이트 = 길이는 뺀다, 길이를 알면 그 길이까지만)
        left = L if L is not None else None
        buf, parts, whole = bytearray(), [], bytearray()
        for r in segs[:end_seg]:
            skip = LEN_BYTES if (version == 2 and r["index"] == 0) else 0
            if r["ok"]:
                d = r["data"][skip:]
                if left is not None:
                    d, left = d[:max(left, 0)], left - len(d)
                buf += d
                whole += d
            else:
                parts.append(_utf8_prefix(bytes(buf)))
                parts.append(PLACEHOLDER)
                buf = bytearray()
                if left is not None:
                    left -= SEG_BYTES - skip
        parts.append(_utf8_prefix(bytes(buf)))
        partial = "".join(parts).split("\x00")[0]
        text = None
        if complete:
            try:
                text = bytes(whole).split(b"\x00")[0].decode("utf-8")
            except UnicodeDecodeError:
                text = None
        info = {"n_segments": n_av, "segments_ok": int(sum(ok[:end_seg])),
                "segments_used": end_seg, "relocks": self.relocks,
                "offsets": [r["offset"] for r in segs], "seg_ok": ok,
                "version": version, "length": L, "declared_segments": need, "stopped": stopped}
        return text, partial, info


def _utf8_prefix(b):
    """구간 경계에서 잘린 UTF-8 은 버리고 온전한 글자만 (실패 구간 앞뒤)"""
    return b.decode("utf-8", errors="ignore")


if __name__ == "__main__":
    import _io_utf8  # noqa: F401
    from fec import conv_encode
    rng = np.random.default_rng(3)
    u = rng.integers(0, 2, 300)
    assert np.array_equal(conv_encode_rows(np.concatenate([u, np.zeros(TAIL, int)]))[0], conv_encode(u)), "부호기 불일치"
    for t in ("HI", "CQ CQ DE HL1ABC HL1ABC PSE KN.", "가" * 200 + "끝"):
        fr, inf = encode_stream(t)
        assert inf["capacity"] >= inf["payload"] and coded_len(inf["capacity"]) <= fr.size
        llr = np.repeat(((2 * fr - 1) * 4.0)[None], len(OFFSETS), axis=0)
        llr[np.arange(len(OFFSETS)) != 9] = rng.normal(0, 1, llr[np.arange(len(OFFSETS)) != 9].shape)
        d = StreamDecoder(inf["n_data"])
        early = 0
        for k in range(inf["n_data"]):                  # 프레임을 하나씩 흘려 넣는다 (실시간과 같은 경로)
            early += len(d.feed(llr[:, k:k + 1]))
        d.finish()
        text, partial, info = d.result(version=2)
        assert text == t.replace("\x00", ""), (t[:10], text, info)
        print("  '{}' {}B → 프레임 {}, 구간 {} (끝 전에 결정된 구간 {}), 정렬 {}".format(
            t[:12], inf["payload"], inf["n_data"], info["n_segments"],
            early,
            sorted(set(info["offsets"]))))
    # v1 (7단계) 호환 · 송신 중지 (길이 > 실제 구간)
    t = "가나다 ABC " * 40
    fr, inf = encode_stream(t, version=1)
    d = StreamDecoder(inf["n_data"])
    d.feed(np.repeat(((2 * fr - 1) * 4.0)[None], len(OFFSETS), axis=0))
    d.finish()
    assert d.result(version=1)[0] == t
    P = t.encode()
    S = stream_bytes(P, 2)[:SEG_BYTES * 3]                    # 앞 3구간만 보내고 중지 (길이는 원래 값)
    fr, inf = encode_raw(S)
    d = StreamDecoder(inf["n_data"])
    d.feed(np.repeat(((2 * fr - 1) * 4.0)[None], len(OFFSETS), axis=0))
    d.finish()
    text, partial, info = d.result(version=2)
    assert text is None and info["stopped"] and info["declared_segments"] == -(-(len(P) + 2) // SEG_BYTES), info
    assert P[:SEG_BYTES * 3 - 2].decode(errors="ignore").startswith(partial[:20])
    print("  v1 호환 통과 · 송신 중지 메시지: 길이 {}B → 필요 {}구간 중 {}구간 받음 (중지 판정)".format(
        info["length"], info["declared_segments"], info["segments_used"]))
    g = {k: 2 * v - 1 for k, v in GUARDS.items()}
    print("  가드 상관 (96이면 같음): v2·단일블록 {:+.0f} · v2·v1 {:+.0f} · v1·단일블록 {:+.0f}".format(
        g["stream2"] @ g["block"], g["stream2"] @ g["stream1"], g["stream1"] @ g["block"]))
    print("link7 자가 검사 통과")
