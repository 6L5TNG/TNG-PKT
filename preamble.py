"""
프리앰블/포스트앰블 — 7x7 코스타스 배열.

송신 (복소 기저대역, fs_base = 2000Hz, 0Hz = 오디오 1500Hz)
    [희생 톤 300ms] [코스타스 A 224ms]  ← 하나의 연속 위상 FSK 로 만든다
    [AI 데이터 — 4단계 그대로]
    [코스타스 B 224ms]
    톤 k 의 주파수 = (k-3) × 62.5Hz  → 오디오 1312.5 ~ 1687.5Hz

수신
    1) 탐색 전 저역통과 (±330Hz) 로 대역 밖 잡음을 버린다
    2) 시간 후보마다  r(t)·conj(s(t))  를 FFT 하면 그 결과가 곧 주파수 오차별 상관값이다.
       → 시작 위치와 주파수 오차(±100Hz)를 한 번에 찾는 시간-주파수 2차원 상관.
       CPFSK 라 코스타스 224ms 전체가 위상이 이어진 하나의 파형이므로 **통째로 코히런트**
       하게 상관할 수 있다 (톤 7개를 따로 더하는 것보다 처리 이득이 크다).
    3) 정규화 상관  rho^2 = |<r, s·e^{j2πΔf t}>|^2 / (|r|^2 |s|^2)  ∈ [0, 1]
       — 수신 레벨과 무관하다. 문턱은 잡음 전용 입력 시험으로 정했다.
    4) 같은 위치에서 A 와 B 중 더 잘 맞는 쪽으로 판정한다 (서로를 오검출하지 않도록).
"""

import itertools

import numpy as np
import scipy.fft as _sfft
from scipy import signal as sg

from waveform import design_lpf


# ------------------------------------------------------------------ 코스타스 배열
def is_costas(p):
    seen = set()
    for i in range(len(p)):
        for j in range(i + 1, len(p)):
            v = (j - i, p[j] - p[i])
            if v in seen:
                return False
            seen.add(v)
    return True


def all_costas(n=7):
    return [p for p in itertools.permutations(range(n)) if is_costas(p)]


def hit_map(a, b):
    """두 배열을 (시간이동 dt, 톤이동 df) 만큼 어긋냈을 때 겹치는 점의 개수"""
    n = len(a)
    return {(dt, df): sum(1 for i in range(n) if 0 <= i + dt < n and b[i + dt] == a[i] + df)
            for dt in range(-(n - 1), n) for df in range(-(n - 1), n)}


# ------------------------------------------------------------------ 파형 생성
def _samples(ms, fs):
    return int(round(ms * 1e-3 * fs))


def cpfsk(tone_seq, pc, fs, lead_tone_samples=0):
    """
    연속 위상 FSK. 톤 순서 → 복소 기저대역 (진폭 1).

    lead_tone_samples > 0 이면 맨 앞에 가운데 톤(0Hz)을 그만큼 붙인다 (희생 톤).
    순간 주파수를 짧은 레이즈드 코사인으로 부드럽게 이어 톤 전환부 스플래터를 줄인다.
    """
    ns = _samples(pc.symbol_ms, fs)
    mid = (pc.n_tones - 1) / 2.0
    f = [np.zeros(lead_tone_samples)] if lead_tone_samples else []
    for k in tone_seq:
        f.append(np.full(ns, (k - mid) * pc.tone_spacing_hz))
    f = np.concatenate(f)
    m = _samples(pc.freq_smooth_ms, fs)
    if m > 1:
        w = np.hanning(m + 2)[1:-1]
        w /= w.sum()
        f = np.convolve(np.concatenate([np.full(m, f[0]), f, np.full(m, f[-1])]), w, mode="same")[m:-m]
    phase = 2 * np.pi * np.cumsum(f) / fs
    return np.exp(1j * phase)


def _ramp(n, up, down):
    e = np.ones(n)
    if up > 0:
        e[:up] = 0.5 * (1 - np.cos(np.pi * np.arange(up) / up))
    if down > 0:
        e[-down:] = 0.5 * (1 + np.cos(np.pi * np.arange(1, down + 1) / down))
    return e


def _bandlimit_full(x, pc, fs):
    """고정 LPF 의 '전체' 합성곱 출력 (앞뒤 꼬리 포함, 길이 len+2*hd). 반환: (y, hd)"""
    h = design_lpf(pc.lpf_cutoff_hz, fs, pc.lpf_taps)
    return np.convolve(x, h, mode="full"), (len(h) - 1) // 2


def _preamble_raw(pc, fs):
    """희생 톤 + 코스타스 A (하나의 연속 위상 파형), 앞쪽 시작 램프·뒤쪽 경계 램프"""
    x = cpfsk(pc.costas_a, pc, fs, lead_tone_samples=_samples(pc.tone_ms, fs))
    return x * _ramp(len(x), _samples(pc.ramp_start_ms, fs), _samples(pc.ramp_join_ms, fs))


def _postamble_raw(pc, fs):
    """코스타스 B, 앞쪽 경계 램프·뒤쪽 송신 끝 램프"""
    x = cpfsk(pc.costas_b, pc, fs)
    return x * _ramp(len(x), _samples(pc.ramp_join_ms, fs), _samples(pc.ramp_end_ms, fs))


def build_preamble(pc, fs):
    """대역제한 전 원형 (길이 측정·표시용)"""
    return _preamble_raw(pc, fs)


def build_postamble(pc, fs):
    return _postamble_raw(pc, fs)


def assemble(pc, fs, data):
    """
    [프리앰블][데이터][포스트앰블] 을 하나의 연속 스트림으로 조립한다.

    프리앰블/포스트앰블만 고정 LPF 를 통과시키고, 필터 꼬리(각 hd 샘플)는 이웃 구간에
    중첩 합산한다. 데이터 구간(AI 송신 출력)은 손대지 않는다.
    반환: (스트림, 정보 dict)
    """
    pre_f, hd = _bandlimit_full(_preamble_raw(pc, fs), pc, fs)
    post_f, _ = _bandlimit_full(_postamble_raw(pc, fs), pc, fs)
    lp, lq, ld = len(pre_f) - 2 * hd, len(post_f) - 2 * hd, len(data)
    n_costas = _samples(pc.symbol_ms, fs) * pc.n_tones
    x = np.zeros(lp + ld + lq + 2 * hd, dtype=complex)
    x[0:len(pre_f)] += pre_f
    d0 = hd + lp
    x[d0:d0 + ld] += data
    x[d0 + ld - hd:d0 + ld - hd + len(post_f)] += post_f

    # 구간 경계 평활화.
    # 4단계 데이터는 AI 송신 LPF 를 거친 **뒤에** 양 끝에 20ms 램프를 곱한다. 곱셈이 대역제한을
    # 깨서 경계에서 대역 밖 성분이 생긴다 (측정: 4단계 단독 짧은 메시지 -60dB 514~517Hz).
    # 경계에서 **데이터 쪽 가드 프레임 안 100ms** 만 고정 LPF 를 거친 신호로 갈아 끼운다.
    # 코스타스 쪽에는 40ms 페이드만 걸린다. 그 안의 톤(A 의 마지막 톤 -62.5Hz, B 의 첫 톤 +125Hz)은
    # LPF 평탄 구간 안이라 LPF 를 한 번 더 거쳐도 거의 변하지 않는다 (수신 템플릿과 어긋나지 않음).
    # 가드 프레임 192ms 중 100+40ms 만 쓰므로 첫/끝 데이터 프레임과 그 수신 문맥(16ms)은 그대로다.
    # 데이터 내부는 이미 같은 LPF 로 대역제한돼 있어 교차 페이드에서도 새 성분이 생기지 않는다.
    zone, fade = _samples(100.0, fs), _samples(40.0, fs)
    h = design_lpf(pc.lpf_cutoff_hz, fs, pc.lpf_taps)
    xf = np.convolve(x, h, mode="same")
    up = 0.5 * (1 - np.cos(np.pi * np.arange(fade) / fade))       # 0 → 1
    w = np.zeros(len(x))
    je = d0 + ld
    # 시작 경계: [d0-fade, d0) 페이드 인, [d0, d0+zone) 1, [d0+zone, +fade) 페이드 아웃
    w[d0 - fade:d0] = up
    w[d0:d0 + zone] = 1.0
    w[d0 + zone:d0 + zone + fade] = up[::-1]
    # 끝 경계: [je-zone-fade, je-zone) 페이드 인, [je-zone, je) 1, [je, je+fade) 페이드 아웃
    w[je - zone - fade:je - zone] = np.maximum(w[je - zone - fade:je - zone], up)
    w[je - zone:je] = 1.0
    w[je:je + fade] = up[::-1]
    x = x + w * (xf - x)
    info = {"hd": hd, "pre": hd + lp, "data": ld, "post": lq + hd,
            "a_start": d0 - n_costas, "data_start": d0, "data_end": d0 + ld,
            "b_start": d0 + ld}
    return x, info


_TEMPLATE_CACHE = {}


def template(pc, fs, which="A"):
    """수신 상관용 기준 파형 — 송신에 들어간 (대역제한된) 코스타스 부분과 똑같다. (한 번 만들고 재사용)"""
    import dataclasses
    key = (dataclasses.astuple(pc), float(fs), which)
    if key not in _TEMPLATE_CACHE:
        _TEMPLATE_CACHE[key] = _template(pc, fs, which)
    return _TEMPLATE_CACHE[key]


def _template(pc, fs, which="A"):
    n_costas = _samples(pc.symbol_ms, fs) * pc.n_tones
    if which == "A":
        y, hd = _bandlimit_full(_preamble_raw(pc, fs), pc, fs)
        end = len(y) - hd
        return y[end - n_costas:end]
    y, hd = _bandlimit_full(_postamble_raw(pc, fs), pc, fs)
    return y[hd:hd + n_costas]


# ------------------------------------------------------------------ 검출
class CostasDetector:
    def __init__(self, pc, fs_base):
        self.pc = pc
        self.fs = float(fs_base)
        self.tA = template(pc, fs_base, "A")
        self.tB = template(pc, fs_base, "B")
        self.n = len(self.tA)                               # 448 샘플
        self.ridges, self.rejected = [], 0                 # 봉우리 모양 검사 기록 (측정 · 로그용)
        h = design_lpf(pc.search_lpf_hz, fs_base, 129)
        self._h = h
        self._hd = (len(h) - 1) // 2
        self._zcache = {}                                  # 30부: {절대 번호: ((|Z|² 최대 A, 위치), (B, 위치))}
        self._cache = {}                                   # 29부: 실시간 탐색 재사용 {절대 기저대역 번호: (rA, fA, rB, fB)}

    def _prefilter(self, bb):
        z = sg.lfilter(self._h, 1.0, np.concatenate([bb, np.zeros(len(self._h))]))
        return z[self._hd:self._hd + len(bb)]

    def _corr_grid(self, x, starts, tmpl, nfft):
        """시간 후보들 × 주파수 격자의 정규화 상관 rho^2  (행 = 시간, 열 = 주파수 bin)"""
        n = self.n
        idx = starts[:, None] + np.arange(n)[None, :]
        r = x[idx]                                          # (T, n)
        prod = r * np.conj(tmpl)[None, :]
        Z = np.fft.fft(prod, n=nfft, axis=1)
        es = np.sum(np.abs(tmpl) ** 2)

        # 정규화 분모 = max(현재 창, 왼쪽 창, 오른쪽 창) 의 에너지.
        # 창 자기 에너지로만 나누면, 창이 신호 가장자리에 걸쳐 한 슬롯 + 무음만 덮을 때
        # 그 한 슬롯만으로 rho^2 가 1/7 까지 올라간다 (측정: 희생 톤만 있을 때 0.149).
        # 양옆 창의 에너지를 함께 보면, 가장자리 창은 옆의 큰 에너지로 나뉘어 작아진다.
        # (처음엔 '슬롯 에너지 균일성' 검사를 썼으나, 다중경로 골에 톤 하나가 빠지면
        #  참 피크를 버려 높은 SNR 에서 오히려 검출이 실패했다 — 측정으로 확인 후 교체.)
        er = self._win_energy(x, starts)
        el = self._win_energy(x, starts - n)
        eright = self._win_energy(x, starts + n)
        den = np.maximum(np.maximum(er, el), eright)
        return (np.abs(Z) ** 2) / np.maximum(den[:, None] * es, 1e-20)

    REUSE_LEFT = True          # 29부: 왼쪽 가장자리 후보도 재사용 (예전: 창 가장자리 영향 받은 값으로 매번 다시)
    Z_M = 96                   # 30부: |Z| 가 정해졌다고 보는 오른쪽 여유 (앞 필터 반길이 64 + 기저대역 가장자리 10 + 여유)
    CACHE_M = 256              # 29부: 재사용 후보의 창 가장자리 여유 (기저대역 샘플 = 0.128 s, 앞 필터 129탭 · 기저대역 변환 가장자리)

    def _zmax2(self, x, starts, nfft, band):
        """30부: 분모 없이 A · B 의 대역 |Z|^2 최대값 · 위치 (분모는 행마다 같은 양수라 최대 위치가 같다)"""
        n = self.n
        if getattr(self, "_conj_t", None) is None:
            self._conj_t = (np.conj(self.tA), np.conj(self.tB))
            self._es = (np.sum(np.abs(self.tA) ** 2), np.sum(np.abs(self.tB) ** 2))
        idx = starts[:, None] + np.arange(n)[None, :]
        r = x[idx]
        buf = np.zeros((len(starts), nfft), np.complex128)
        out = []
        for ct in self._conj_t:
            np.multiply(r, ct[None, :], out=buf[:, :n])
            P = np.abs(_sfft.fft(buf, axis=1)[:, band]) ** 2
            j = np.argmax(P, axis=1)
            out.append((P[np.arange(len(starts)), j], j))
        return out

    def _den(self, x, starts):
        n = self.n
        return np.maximum(np.maximum(self._win_energy(x, starts), self._win_energy(x, starts - n)),
                          self._win_energy(x, starts + n))

    def _corr_grid2(self, x, starts, nfft, band):
        """_corr_grid 를 A · B 두 템플릿에 한 번에 (창 자르기 · 분모 에너지 공유, 계산식 같음). 반환 (gA, gB) = 대역 열만"""
        n = self.n
        if getattr(self, "_conj_t", None) is None:
            self._conj_t = (np.conj(self.tA), np.conj(self.tB))
            self._es = (np.sum(np.abs(self.tA) ** 2), np.sum(np.abs(self.tB) ** 2))
        idx = starts[:, None] + np.arange(n)[None, :]
        r = x[idx]
        er = self._win_energy(x, starts)
        el = self._win_energy(x, starts - n)
        eright = self._win_energy(x, starts + n)
        den = np.maximum(np.maximum(er, el), eright)
        res = []
        buf = np.zeros((len(starts), nfft), np.complex128)          # 30부: 0 채움을 미리 (scipy 가 매번 새로 만들던 복사 없앰, 값 같음)
        for ct, es in zip(self._conj_t, self._es):
            np.multiply(r, ct[None, :], out=buf[:, :n])
            Z = _sfft.fft(buf, axis=1, overwrite_x=False)[:, band]   # scipy.fft (numpy 와 차이 1e-16)
            res.append((np.abs(Z) ** 2) / np.maximum(den[:, None] * es, 1e-20))
        return res

    def _win_energy(self, x, starts):
        """길이 n 창의 에너지 (범위 밖은 0 으로 본다) — 누적합으로 빠르게"""
        if getattr(self, "_cs_src", None) is not x:
            self._cs = np.concatenate([[0.0], np.cumsum(np.abs(x) ** 2)])
            self._cs_src = x
        cs = self._cs
        a = np.clip(starts, 0, len(x))
        b = np.clip(starts + self.n, 0, len(x))
        return cs[b] - cs[a]

    use_gpu = False          # 28부: 앱은 GPU 안 씀 (True 면 GPU 있을 때 시간-주파수 상관을 GPU 배치로)

    def _scan_gpu(self, x, starts, band):
        """_corr_grid 와 같은 계산을 GPU 배치로 (모든 시간 후보 × A/B 템플릿을 한 번에)."""
        import torch
        dev = torch.device("cuda")
        if getattr(self, "_gpu_t", None) is None:
            self._gpu_t = {k: torch.as_tensor(np.conj(t).astype(np.complex64), device=dev)
                           for k, t in (("A", self.tA), ("B", self.tB))}
        pc, n = self.pc, self.n
        xt = torch.as_tensor(x.astype(np.complex64), device=dev)
        st = torch.as_tensor(starts, device=dev)
        r = xt[st[:, None] + torch.arange(n, device=dev)[None, :]]              # (T, n)
        den = np.maximum(np.maximum(self._win_energy(x, starts), self._win_energy(x, starts - n)),
                         self._win_energy(x, starts + n))
        es = float(np.sum(np.abs(self.tA) ** 2))
        dent = torch.as_tensor(np.maximum(den * es, 1e-20).astype(np.float32), device=dev)[:, None]
        bt = torch.as_tensor(np.where(band)[0], device=dev)
        out = {}
        with torch.no_grad():
            for k in ("A", "B"):
                Z = torch.fft.fft(r * self._gpu_t[k][None, :], n=pc.coarse_nfft, dim=1)[:, bt]
                g = (Z.real ** 2 + Z.imag ** 2) / dent
                v, j = torch.max(g, dim=1)
                out[k] = (v.cpu().numpy().astype(np.float64), j.cpu().numpy())
        return out

    def scan(self, bb, chunk=512, key0=None):
        """
        전체 버퍼를 훑어 시간 후보별 최대 rho^2 와 그때의 주파수 오차를 구한다.
        반환: starts, rhoA, dfA, rhoB, dfB
        """
        pc = self.pc
        x = self._prefilter(np.asarray(bb, dtype=np.complex128))
        if len(x) < self.n + 1:
            e = np.zeros(0)
            return e.astype(int), e, e, e, e
        starts = np.arange(0, len(x) - self.n, pc.coarse_step)
        if self.use_gpu and len(starts) >= 256:
            try:
                import torch
                if torch.cuda.is_available():
                    f = np.fft.fftfreq(pc.coarse_nfft, 1.0 / self.fs)
                    band = np.abs(f) <= pc.max_freq_off_hz
                    fb = f[band]
                    o = self._scan_gpu(x, starts, band)
                    return starts, o["A"][0], fb[o["A"][1]], o["B"][0], fb[o["B"][1]]
            except Exception:
                pass                                           # GPU 가 안 되면 아래 CPU 경로
        f = np.fft.fftfreq(pc.coarse_nfft, 1.0 / self.fs)
        band = np.abs(f) <= pc.max_freq_off_hz
        fb = f[band]
        out = {k: np.empty(len(starts)) for k in ("rA", "fA", "rB", "fB")}
        # 29부: 실시간 (key0 = bb[0] 의 절대 기저대역 번호) 이면 앞 탐색 창에서 이미 계산한 시간 후보는 다시 계산하지 않는다.
        # 저장 · 재사용은 두 창 모두에서 안쪽 (가장자리 필터 · 옆 창 에너지 영향 밖, CACHE_M) 인 후보만 → 값이 같다
        inner = (starts >= self.n + self.CACHE_M) & (starts + 2 * self.n + self.CACHE_M <= len(x))
        todo = np.ones(len(starts), bool)
        if key0 is not None:
            cache = self._cache
            if cache and (min(cache) > key0 + len(x) or max(cache) < key0 - len(x)):
                cache.clear()
            left = starts < self.n + self.CACHE_M                 # 왼쪽 가장자리: 앞 창에서 안쪽일 때 계산한 값 (가장자리 영향 없는 값) 을 씀
            for i in np.nonzero(inner | (left if self.REUSE_LEFT else False))[0]:
                v = cache.get(key0 + int(starts[i]))
                if v is not None:
                    out["rA"][i], out["fA"][i], out["rB"][i], out["fB"][i] = v
                    todo[i] = False
        idx_todo = np.nonzero(todo)[0]
        if key0 is not None and len(idx_todo):
            # 30부: 오른쪽 가장자리 후보 = 창 (n) · 앞 필터 여유 (ZM) 까지 들어온 것은 |Z|^2 최대가 이미 정해짐 → 저장해 두고 분모 (옆 창 에너지) 만 새로
            zc = self._zcache
            if zc and (min(zc) > key0 + len(x) or max(zc) < key0 - len(x)):
                zc.clear()
            zfinal = (starts + self.n + self.Z_M <= len(x)) & (starts >= self.Z_M)
            den = self._den(x, starts)
            need = []
            for i in idx_todo:
                v = zc.get(key0 + int(starts[i])) if zfinal[i] else None
                if v is None:
                    need.append(i)
                    continue
                (mA, jA), (mB, jB) = v
                out["rA"][i], out["fA"][i] = mA / max(den[i] * self._es[0], 1e-20), fb[jA]
                out["rB"][i], out["fB"][i] = mB / max(den[i] * self._es[1], 1e-20), fb[jB]
            need = np.array(need, int)
            for c in range(0, len(need), chunk):
                ii = need[c:c + chunk]
                (mA, jA), (mB, jB) = self._zmax2(x, starts[ii], pc.coarse_nfft, band)
                d_ = np.maximum(den[ii] * self._es[0], 1e-20)
                out["rA"][ii], out["fA"][ii] = mA / d_, fb[jA]
                d_ = np.maximum(den[ii] * self._es[1], 1e-20)
                out["rB"][ii], out["fB"][ii] = mB / d_, fb[jB]
                for q, i in enumerate(ii):
                    if zfinal[i]:
                        zc[key0 + int(starts[i])] = ((mA[q], jA[q]), (mB[q], jB[q]))
            for k_ in [k_ for k_ in zc if k_ < key0]:
                del zc[k_]
            idx_todo = idx_todo[:0]
        for c in range(0, len(idx_todo), chunk):
            ii = idx_todo[c:c + chunk]
            s = starts[ii]
            gA, gB = self._corr_grid2(x, s, pc.coarse_nfft, band)
            for key, g in (("A", gA), ("B", gB)):
                j = np.argmax(g, axis=1)
                out["r" + key][ii] = g[np.arange(len(s)), j]
                out["f" + key][ii] = fb[j]
        if key0 is not None:
            for k_ in [k_ for k_ in self._cache if k_ < key0]:
                del self._cache[k_]
            for i in np.nonzero(inner & todo)[0]:
                self._cache[key0 + int(starts[i])] = (out["rA"][i], out["fA"][i], out["rB"][i], out["fB"][i])
        return starts, out["rA"], out["fA"], out["rB"], out["fB"]

    def refine(self, bb, start, df0, which="A", span=6):
        """시작 위치 ±span 샘플, 주파수 ±4Hz 를 정밀 격자로 다시 찾는다."""
        pc = self.pc
        x = self._prefilter(np.asarray(bb, dtype=np.complex128))
        t = self.tA if which == "A" else self.tB
        lo, hi = max(0, start - span), min(len(x) - self.n - 1, start + span)
        if hi < lo:
            return start, df0, 0.0
        starts = np.arange(lo, hi + 1)
        g = self._corr_grid(x, starts, t, pc.fine_nfft)
        f = np.fft.fftfreq(pc.fine_nfft, 1.0 / self.fs)
        band = np.abs(f - df0) <= 4.0
        gb = g[:, band]
        ti, fi = np.unravel_index(np.argmax(gb), gb.shape)
        fsel = f[band]
        # 포물선 보간으로 주파수를 한 번 더 다듬는다
        dfv = fsel[fi]
        if 0 < fi < gb.shape[1] - 1:
            y0, y1, y2 = gb[ti, fi - 1], gb[ti, fi], gb[ti, fi + 1]
            den = y0 - 2 * y1 + y2
            if abs(den) > 1e-20:
                dfv += 0.5 * (y0 - y2) / den * (fsel[1] - fsel[0])
        return int(starts[ti]), float(dfv), float(gb[ti, fi])

    # 봉우리 모양 검사: 코스타스 배열은 시간 · 주파수를 어떻게 옮겨도 톤이 최대 1개만 겹친다 (엄지압정 모양, 곁봉 ≈ 1/7).
    # 미끄러지는 음(휘파람 등)은 시간을 1~2 심볼 옮겨도 주파수 대각선을 따라 상관이 크게 남는다 (비스듬한 줄 모양).
    # ridge = 시간 ±1 · ±2 심볼, 주파수 ±160 Hz 안 최대 rho^2 / 봉우리 rho^2.  문턱 근거: README (fake_check / det_check)
    shape_check = True
    SHAPE_MAX = 0.5

    def ridge(self, x, start, df0, which="A"):
        t = self.tA if which == "A" else self.tB
        sym = self.n // self.pc.n_tones
        st = np.array([start - 2 * sym, start - sym, start + sym, start + 2 * sym])
        st = st[(st >= 0) & (st + self.n < len(x))]
        if len(st) == 0:
            return 0.0
        g = self._corr_grid(x, st, t, self.pc.coarse_nfft)
        f = np.fft.fftfreq(self.pc.coarse_nfft, 1.0 / self.fs)
        return float(g[:, np.abs(f - df0) <= 160.0].max())

    def detect(self, bb, threshold=None, key0=None):
        """
        A/B 검출 목록을 돌려준다: [{'kind','start','df','rho'}...] (시간순)
        같은 위치에서 A 와 B 중 큰 쪽으로 판정하고, 한 배열 길이 안에서는 가장 센 것만 남긴다.
        """
        thr = self.pc.detect_threshold if threshold is None else threshold
        starts, rA, fA, rB, fB = self.scan(bb, key0=key0)
        # 마지막 탐색의 최대 상관값 (실시간 수신의 동작 기록용)
        self.last_max = float(max(rA.max(), rB.max())) if len(starts) else 0.0
        found = []
        for kind, r, f, other in (("A", rA, fA, rB), ("B", rB, fB, rA)):
            cand = np.where((r >= thr) & (r > other))[0]
            order = cand[np.argsort(r[cand])[::-1]]
            taken = []
            for i in order:
                if any(abs(starts[i] - starts[j]) < self.n for j in taken):
                    continue
                taken.append(i)
            for i in taken:
                s, dfv, rho = self.refine(bb, int(starts[i]), float(f[i]), kind)
                rid = self.ridge(self._prefilter(np.asarray(bb, dtype=np.complex128)), s, dfv, kind) / max(rho, 1e-9)
                self.ridges.append(rid)
                if self.shape_check and rid > self.SHAPE_MAX:
                    self.rejected += 1
                    continue                                   # 줄 모양 봉우리 → 코스타스 아님
                found.append({"kind": kind, "start": s, "df": dfv, "rho": rho,
                              "rho_coarse": float(r[i]), "ridge": rid})
        return sorted(found, key=lambda d: d["start"])
