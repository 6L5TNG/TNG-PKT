"""
4단계 링크 계층 — FEC + 인터리빙을 포함한 패킷 구성.

AI 모뎀과 기존 방식(QPSK) 기준선이 **똑같이** 이 계층을 쓴다. 그래야 비교가 공정하다.

패킷
    [길이 2바이트][페이로드][CRC16 2바이트][0 패딩]
      → 구속장 7, 부호율 1/2 컨볼루션 부호 (+ 꼬리 6비트)
      → 전체 부호어에 걸친 의사난수 인터리빙
      → 96비트씩 잘라 프레임으로

패딩은 '정보 비트' 쪽에 넣는다. 그래야 부호어 길이가 정확히 프레임 경계에 떨어지고,
수신측은 프레임 수만 알면 정보 비트 수를 역산할 수 있다.

프레임 앞뒤에는 가드 프레임을 하나씩 붙인다(송신 램프가 덮는 구간, 부호어 밖).
"""

import numpy as np

from tngpkt.fec import conv_encode, viterbi_decode, viterbi_metrics_auto, interleave, deinterleave, TAIL
from tngpkt.framing import crc16_ccitt, bytes_to_bits, bits_to_bytes

BITS_PER_FRAME = 96
INFO_PER_FRAME = BITS_PER_FRAME // 2          # 부호율 1/2
LEN_BYTES = 2
CRC_BYTES = 2

# 가드 프레임의 고정 비트 패턴 (부호어에 포함되지 않는다)
GUARD_BITS = np.unpackbits(
    np.frombuffer(bytes([0x5A, 0xC3, 0x96, 0x69, 0x3C, 0xA5, 0xD2, 0x2B,
                         0x0F, 0xF0, 0x33, 0xCC]), dtype=np.uint8)).astype(np.float32)
assert len(GUARD_BITS) == BITS_PER_FRAME


def frames_needed(payload_len: int) -> int:
    n_info = (LEN_BYTES + payload_len + CRC_BYTES) * 8
    return int(np.ceil((n_info + TAIL) * 2 / BITS_PER_FRAME))


def info_bits_for(n_frames: int) -> int:
    """프레임 수 → 꼬리를 뺀 정보 비트 수 (패딩 포함)"""
    return n_frames * INFO_PER_FRAME - TAIL


def encode_text(text: str):
    """
    텍스트 → (데이터 프레임 비트 (n_frames, 96), 정보)
    실제 송신 프레임은 [가드] + 데이터 프레임들 + [가드] 이다.
    """
    payload = text.encode("utf-8")
    n_frames = frames_needed(len(payload))
    n_info = info_bits_for(n_frames)

    body = len(payload).to_bytes(LEN_BYTES, "big") + payload
    pkt = body + crc16_ccitt(body).to_bytes(CRC_BYTES, "big")

    info = np.zeros(n_info, dtype=np.int64)
    b = bytes_to_bits(pkt).astype(np.int64)
    assert len(b) <= n_info, "패킷이 프레임에 들어가지 않는다"
    info[:len(b)] = b

    coded = conv_encode(info)                      # 길이 = n_frames*96
    assert len(coded) == n_frames * BITS_PER_FRAME
    return interleave(coded).reshape(n_frames, BITS_PER_FRAME).astype(np.float32), n_frames


def decode_llr(llr_block: np.ndarray):
    """
    데이터 프레임들의 LLR (n_frames, 96) → (텍스트 또는 None, 정보 dict)
    """
    n_frames = llr_block.shape[0]
    n_info = info_bits_for(n_frames)
    bits = viterbi_decode(deinterleave(llr_block.reshape(-1)), n_info)

    nbytes = n_info // 8
    data = bits_to_bytes(bits[:nbytes * 8])
    plen = int.from_bytes(data[:LEN_BYTES], "big")
    if plen + LEN_BYTES + CRC_BYTES > len(data):
        return None, {"crc_ok": False, "reason": "길이 필드가 범위를 벗어남", "plen": plen}
    body = data[:LEN_BYTES + plen]
    crc_rx = int.from_bytes(data[LEN_BYTES + plen:LEN_BYTES + plen + CRC_BYTES], "big")
    if crc16_ccitt(body) != crc_rx:
        return None, {"crc_ok": False, "reason": "CRC 불일치", "plen": plen}
    try:
        text = body[LEN_BYTES:].decode("utf-8")
    except UnicodeDecodeError:
        return None, {"crc_ok": True, "reason": "UTF-8 해석 실패", "plen": plen}
    return text, {"crc_ok": True, "plen": plen}


def search_and_decode(llr_grid: np.ndarray, n_data: int, start_slots=(0, 1, 2), top_k: int = 4):
    """
    프레임 정렬을 스스로 찾아 복호한다.

    llr_grid : (n_off, n_slots, 96) — 정렬 후보 × 프레임 자리 × 비트 LLR
    n_data   : 데이터 프레임 수
    방법     : 모든 (정렬, 시작자리) 조합을 **배치 비터비**에 넣어 경로 metric 을 구하고,
               metric 이 큰 순서대로 실제 복호해 CRC 로 확인한다.
               사람이 만든 동기 워드나 상관기를 쓰지 않는다.
    """
    n_off, n_slots, _ = llr_grid.shape
    cands, blocks = [], []
    for j in start_slots:
        if j + n_data > n_slots:
            continue
        blk = llr_grid[:, j:j + n_data, :].reshape(n_off, -1)
        blocks.append(np.stack([deinterleave(r) for r in blk]))
        cands += [(o, j) for o in range(n_off)]
    if not blocks:
        return None, {"crc_ok": False, "reason": "프레임 자리가 모자람"}

    allblk = np.concatenate(blocks, axis=0)

    # 후보끼리 비교하려면 경로 metric 을 '정규화된 로그우도' 로 바꿔야 한다.
    #   log P(y|c) = sum_i [ c_i*L_i - log(1+exp(L_i)) ]
    # 비터비가 최대화하는 sum c_i*L_i 는 LLR 크기에 비례하므로, 그것만 비교하면
    # '틀린 정렬인데 과신해서 |LLR| 이 큰' 후보가 이겨 버린다.
    # 두 번째 항은 경로와 무관하지만 후보마다 다르므로 반드시 빼 줘야 한다.
    met = viterbi_metrics_auto(allblk) - np.logaddexp(0.0, allblk).sum(axis=1)
    order = np.argsort(met)[::-1]

    for rank, idx in enumerate(order[:top_k]):
        o, j = cands[idx]
        text, info = decode_llr(llr_grid[o, j:j + n_data, :])
        if text is not None:
            info.update({"offset": o, "start_slot": j, "rank": rank,
                         "metric": float(met[idx]),
                         "metric_margin": float(met[order[0]] - np.median(met)),
                         "llr": llr_grid[o, j:j + n_data, :].copy(),      # 시각화용 (수신 NN 소프트 출력)
                         "n_candidates": len(cands)})
            return text, info
    o, j = cands[order[0]]
    return None, {"crc_ok": False, "reason": "상위 {}개 후보 모두 CRC 실패".format(top_k),
                  "offset": o, "start_slot": j,
                  "metric_margin": float(met[order[0]] - np.median(met)),
                  "llr": llr_grid[o, j:j + n_data, :].copy(), "n_candidates": len(cands)}


if __name__ == "__main__":
    t = "CQ CQ DE HL1ABC. 신경망이 만든 변조 방식 시험 중입니다. 73!"
    fr, n = encode_text(t)
    print("  '{}...' ({}바이트) → 데이터 프레임 {}개".format(t[:20], len(t.encode()), n))
    llr = (2 * fr - 1) * 8.0
    back, info = decode_llr(llr)
    assert back == t, (back, info)
    print("  무잡음 복호 OK:", info)

    # 잡음을 섞어도 FEC 가 복구하는지
    rng = np.random.default_rng(1)
    for sigma in (1.0, 2.0, 3.0, 4.0):
        ok = sum(decode_llr((2 * fr - 1) * 4.0 + rng.normal(0, sigma, fr.shape))[0] == t
                 for _ in range(20))
        print("  LLR 잡음 sigma={:.0f} → 복원 {}/20".format(sigma, ok))

    # 정렬 탐색 (가드 1개를 앞에 둔 상태를 흉내)
    grid = np.concatenate([rng.normal(0, 1, (1, BITS_PER_FRAME)),
                           (2 * fr - 1) * 6.0,
                           rng.normal(0, 1, (1, BITS_PER_FRAME))], axis=0)
    grid = np.stack([np.roll(grid, k, axis=0) for k in range(5)])      # 가짜 정렬 5개
    text, info = search_and_decode(grid, n, start_slots=(0, 1, 2))
    assert text == t, info
    print("  정렬 탐색 OK:", {k: info[k] for k in ("offset", "start_slot", "rank")})
    print("link4 자가 검사 통과")
