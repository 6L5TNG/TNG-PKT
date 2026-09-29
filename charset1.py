"""
charset1 — TNG1 문자표 (독립 모듈, 다른 모드에서도 쓸 수 있게 외부 의존 없음)

57자 + 끝 표식 EOT, 정적 canonical Huffman (최장 MAXLEN 비트). 표는 아래 WEIGHTS 로 결정되고 공중 형식의 일부다
(WEIGHTS 를 바꾸면 wire rev 를 올린다).
  · 소문자 → 대문자, 집합 밖 문자는 normalize() 가 알려주고 뺀다
  · 블록 채우기 pack_block(text, nbits): 들어가는 만큼 글자를 넣고 나머지를 1 로 채운다.
    채움 길이 < 다음 글자 길이 ≤ MAXLEN 이고 canonical 부호의 마지막 부호가 1…1 (MAXLEN) 이므로 채움은 절대 완성 부호가 되지 않는다
  · 메시지 끝: EOT 부호 (들어가면 넣는다). 복호는 EOT 에서 멈추고, 끝의 미완성 비트는 버린다
  · 오류 전파: 블록마다 CRC 로 확인하므로 틀린 블록은 통째로 버린다. 블록 경계를 넘는 문맥이 없다
측정 (exp_tng1_charset.py, 교신 예문 시험 650자): 평균 4.84 비트/자
"""
import heapq

CHARS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 \n.,?'/-=()!:+\"@&;*#%"
EOT = "\x04"
assert len(CHARS) == 57
# 교신 예문 빈도 (exp_tng1_charset.py 의 A+B) + 0.5. 순서가 곧 동률 처리 순서
_CNT = {'A': 90, 'B': 38, 'C': 46, 'D': 43, 'E': 113, 'F': 18, 'G': 34, 'H': 44, 'I': 57, 'J': 6, 'K': 38, 'L': 54,
        'M': 41, 'N': 73, 'O': 70, 'P': 31, 'Q': 33, 'R': 66, 'S': 50, 'T': 73, 'U': 24, 'V': 10, 'W': 34, 'X': 28,
        'Y': 21, 'Z': 9, '0': 20, '1': 29, '2': 23, '3': 20, '4': 11, '5': 16, '6': 6, '7': 19, '8': 3, '9': 10,
        ' ': 353, '\n': 39, '.': 71, ',': 7, '?': 10, "'": 0, '/': 6, '-': 7, '=': 2, '(': 2, ')': 2, '!': 3,
        ':': 8, '+': 0, '"': 0, '@': 0, '&': 0, ';': 0, '*': 0, '#': 0, '%': 1, EOT: 12}
WEIGHTS = {c: v + 0.5 for c, v in _CNT.items()}
SYMS = CHARS + EOT


def _lengths(w):
    h = [(v, i, (c,)) for i, (c, v) in enumerate(w.items())]
    heapq.heapify(h)
    L = {c: 0 for c in w}
    k = len(h)
    while len(h) > 1:
        a, _, x = heapq.heappop(h)
        b, _, y = heapq.heappop(h)
        for c in x + y:
            L[c] += 1
        k += 1
        heapq.heappush(h, (a + b, k, x + y))
    return L


def _canonical(L):
    order = sorted(L, key=lambda c: (L[c], SYMS.index(c)))
    code, prev, out = 0, L[order[0]], {}
    for c in order:
        code <<= (L[c] - prev)
        prev = L[c]
        out[c] = tuple(int(b) for b in format(code, "0%db" % L[c]))
        code += 1
    return out


LEN = _lengths(WEIGHTS)
CODE = _canonical(LEN)
MAXLEN = max(LEN.values())
assert CODE[sorted(LEN, key=lambda c: (LEN[c], SYMS.index(c)))[-1]] == (1,) * MAXLEN
_DEC = {v: k for k, v in CODE.items()}


def normalize(text):
    """→ (보낼 문자열, 집합 밖 문자 목록). \\r\\n → \\n, 탭 → 공백"""
    t = text.upper().replace("\r\n", "\n").replace("\r", "\n").replace("\t", " ")
    bad = sorted({c for c in t if c not in CHARS})
    return "".join(c for c in t if c in CHARS), bad


def bits_of(text):
    return [b for c in text for b in CODE[c]]


def pack_block(text, nbits, end=False):
    """text 앞에서부터 nbits 에 들어가는 만큼 → (비트 list 길이 nbits, 넣은 글자 수, EOT 넣었는지).
    end=True 이면 text 를 다 넣은 뒤 자리가 있으면 EOT 를 넣는다"""
    out, n = [], 0
    for c in text:
        if len(out) + LEN[c] > nbits:
            break
        out += CODE[c]
        n += 1
    eot = False
    if end and n == len(text) and len(out) + LEN[EOT] <= nbits:
        out += CODE[EOT]
        eot = True
    out += [1] * (nbits - len(out))
    return out, n, eot


def unpack_block(bits):
    """→ (글자열, EOT 만났는지). 끝의 미완성 비트는 버린다"""
    s, cur = [], ()
    for b in bits:
        cur += (int(b),)
        c = _DEC.get(cur)
        if c is not None:
            if c == EOT:
                return "".join(s), True
            s.append(c)
            cur = ()
    return "".join(s), False


def selftest():
    import random
    r = random.Random(1)
    for _ in range(2000):
        t = "".join(r.choice(CHARS) for _ in range(r.randint(0, 30)))
        nb = r.randint(11, 120)
        b, n, e = pack_block(t, nb, end=True)
        assert len(b) == nb
        s, e2 = unpack_block(b)
        assert s == t[:n] and e == e2, (t, nb, s, n)
    avg = sum(LEN[c] * WEIGHTS[c] for c in CHARS) / sum(WEIGHTS[c] for c in CHARS)
    print("charset1 자가 확인 통과: 57자 + EOT, 최장 %d 비트, EOT %d 비트, 학습 빈도 평균 %.2f 비트/자" % (MAXLEN, LEN[EOT], avg))


if __name__ == "__main__":
    selftest()
