"""
프레임 구성: 텍스트 ↔ 비트 프레임.

프레임 하나 = 96비트 = 12바이트

    +--------+--------+----------------------------+--------+
    | 헤더0  | 헤더1  |      페이로드 8바이트      | CRC16  |
    | seq    | 플래그 |                            | 2바이트|
    +--------+--------+----------------------------+--------+

  헤더0 : 순번 seq (0~255, 넘치면 되돌아감)
  헤더1 : bit7 = 마지막 프레임, bit6 = 가드 프레임, bit3~0 = 페이로드 유효 바이트 수(0~8)
  CRC16 : CCITT-FALSE (다항식 0x1021, 초기값 0xFFFF) — 헤더 2바이트 + 페이로드 8바이트에 대해 계산

가드 프레임은 송신 시작/끝의 램프 구간을 덮는 더미 프레임이다.
내용은 고정 패턴이고 CRC 도 정상이라 수신측 프레임 위치 탐색에 그대로 쓰인다.
"""

import numpy as np

HEADER_BYTES = 2
PAYLOAD_BYTES = 8
CRC_BYTES = 2
FRAME_BYTES = HEADER_BYTES + PAYLOAD_BYTES + CRC_BYTES      # 12
FRAME_BITS = FRAME_BYTES * 8                                # 96

FLAG_LAST = 0x80
FLAG_GUARD = 0x40

GUARD_PATTERN = bytes([0x5A, 0xC3, 0x96, 0x69, 0x3C, 0xA5, 0xD2, 0x2B])


# ------------------------------------------------------------------ CRC16
def crc16_ccitt(data: bytes, init: int = 0xFFFF) -> int:
    """CRC-16/CCITT-FALSE"""
    crc = init
    for b in data:
        crc ^= b << 8
        for _ in range(8):
            crc = ((crc << 1) ^ 0x1021) & 0xFFFF if (crc & 0x8000) else (crc << 1) & 0xFFFF
    return crc


# ------------------------------------------------------------------ 바이트 ↔ 비트
def bytes_to_bits(data: bytes) -> np.ndarray:
    """바이트열 → 0/1 배열 (각 바이트는 MSB 부터)"""
    return np.unpackbits(np.frombuffer(data, dtype=np.uint8)).astype(np.float32)


def bits_to_bytes(bits) -> bytes:
    b = np.asarray(bits).round().astype(np.uint8).reshape(-1)
    assert len(b) % 8 == 0, "비트 수가 8의 배수가 아니다"
    return np.packbits(b).tobytes()


# ------------------------------------------------------------------ 프레임 만들기
def build_frame(seq: int, payload: bytes, last: bool = False, guard: bool = False,
                nvalid: int = None) -> bytes:
    """12바이트 프레임 하나를 만든다."""
    if nvalid is None:
        nvalid = len(payload)
    payload = payload[:PAYLOAD_BYTES].ljust(PAYLOAD_BYTES, b"\x00")
    flags = (FLAG_LAST if last else 0) | (FLAG_GUARD if guard else 0) | (nvalid & 0x0F)
    head = bytes([seq & 0xFF, flags])
    body = head + payload
    crc = crc16_ccitt(body)
    return body + bytes([(crc >> 8) & 0xFF, crc & 0xFF])


def parse_frame(frame: bytes):
    """
    12바이트 프레임을 해석한다.
    반환: dict(ok, seq, last, guard, nvalid, payload)  — ok 가 False 면 CRC 불일치
    """
    if len(frame) != FRAME_BYTES:
        return {"ok": False}
    body, crc_rx = frame[:-2], (frame[-2] << 8) | frame[-1]
    ok = (crc16_ccitt(body) == crc_rx)
    flags = body[1]
    return {
        "ok": ok,
        "seq": body[0],
        "last": bool(flags & FLAG_LAST),
        "guard": bool(flags & FLAG_GUARD),
        "nvalid": flags & 0x0F,
        "payload": body[2:],
    }


def guard_frame(seq: int = 0) -> bytes:
    return build_frame(seq, GUARD_PATTERN, last=False, guard=True, nvalid=0)


# ------------------------------------------------------------------ 텍스트 ↔ 프레임
def text_to_frames(text: str, add_guards: bool = True):
    """
    텍스트 → 프레임 바이트열 목록.
    앞뒤에 가드 프레임을 하나씩 붙인다(송신 램프 구간을 덮기 위함).
    """
    data = text.encode("utf-8")
    chunks = [data[i:i + PAYLOAD_BYTES] for i in range(0, len(data), PAYLOAD_BYTES)] or [b""]
    frames = []
    if add_guards:
        frames.append(guard_frame(0))
    for i, c in enumerate(chunks):
        frames.append(build_frame(i & 0xFF, c, last=(i == len(chunks) - 1), nvalid=len(c)))
    if add_guards:
        frames.append(guard_frame(1))
    return frames


def frames_to_bits(frames) -> np.ndarray:
    """프레임 목록 → (프레임수, 96) 0/1 배열"""
    return np.stack([bytes_to_bits(f) for f in frames])


def frames_to_text(parsed_list):
    """
    parse_frame 결과 목록 → (복원된 텍스트, 통계 dict)

    CRC 를 통과한 데이터 프레임만 순번 순서로 이어 붙인다.
    순번이 비면 그 자리는 복원 실패이므로 통계에 남긴다.
    """
    good = {}
    n_ok = n_bad = n_guard = 0
    last_seq = None
    for p in parsed_list:
        if not p.get("ok"):
            n_bad += 1
            continue
        if p["guard"]:
            n_guard += 1
            continue
        n_ok += 1
        good[p["seq"]] = p
        if p["last"]:
            last_seq = p["seq"]

    if not good:
        return "", {"ok": n_ok, "bad": n_bad, "guard": n_guard,
                    "missing": [], "complete": False}

    end = last_seq if last_seq is not None else max(good)
    missing = [s for s in range(end + 1) if s not in good]
    data = b""
    for s in range(end + 1):
        if s in good:
            p = good[s]
            n = p["nvalid"] if (p["last"] or p["nvalid"] < PAYLOAD_BYTES) else PAYLOAD_BYTES
            data += p["payload"][:n]
    text = data.decode("utf-8", errors="replace")
    complete = (last_seq is not None) and (len(missing) == 0)
    return text, {"ok": n_ok, "bad": n_bad, "guard": n_guard,
                  "missing": missing, "complete": complete}


if __name__ == "__main__":
    # 간단 자가 검사
    assert crc16_ccitt(b"123456789") == 0x29B1, "CRC16/CCITT-FALSE 검증 실패"
    t = "안녕하세요, CQ CQ DE HL1ABC. 신경망 변조 시험 중입니다."
    fr = text_to_frames(t)
    back, st = frames_to_text([parse_frame(f) for f in fr])
    assert back == t, (back, t)
    assert st["complete"]
    # 한 프레임을 고의로 깨뜨리면 CRC 가 잡아내야 한다
    bad = bytearray(fr[2]); bad[3] ^= 0x01
    assert not parse_frame(bytes(bad))["ok"]
    print("framing 자가 검사 통과")
    print("  프레임 {}개, 프레임당 {}비트".format(len(fr), FRAME_BITS))
    print("  원문 {}바이트 → 순 페이로드 {}바이트".format(len(t.encode()), len(fr[1:-1]) * 8))
