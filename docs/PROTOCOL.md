# TNG PKT protocol design and operation

[English](PROTOCOL.md) | [한국어](PROTOCOL.ko.md)

This document describes the **TNG PKT 0.11.2 Beta / Build 1009** public source, dated 2026-09-29. It covers the active transmit and receive paths, not initial designs or legacy experiment configurations.

## Overview

TNG PKT is intended for open amateur-radio communication. Its current on-air paths do not provide message encryption, secret-key communication or a message-concealment feature. Text coding, CRC, FEC, interleaving, synchronization and the fixed public whitening described here serve communication functions, not secrecy. Public model weights are part of the TNG44/TNG5 waveform definition and are not secret keys.

TNG PKT carries conversational text through the audio path of an SSB transmitter and receiver. Its three modes are independent communication modes, with different framing, protection and acquisition choices; they are not speed presets of one common wire format. A station selects the mode before transmitting or receiving.

TNG44 prioritizes throughput on relatively good HF channels. TNG5 spends more airtime on redundancy and interleaving for weak signals and fading. TNG1 accepts long blocks and a very low text rate to prioritize recovery, including combining observations of repeated content. These are design goals, not measured channel thresholds or guarantees. The numbers in the names do not specify current characters per second.

Read the common paths below for the overall processing order, then the mode sections for the exact wire structure. The text tables and waveform definitions are public so another implementation can reconstruct the encoding from this document, the linked source and the supplied model weights.

## Scope and conventions

| Mode | Intended role | Protocol version | Wire revision | Main implementation |
|---|---|---|---|---|
| TNG44 | Standard | 0.1.0 | 1 | [link7](../link7.py), [modem5](../modem5.py), [realtime_rx](../realtime_rx.py) |
| TNG5 | Robust | 0.4.0 | 4 | [tng5](../tng5.py), [tng5_app](../tng5_app.py) |
| TNG1 | Robust+ | 0.1.1 | 2 | [tng1](../tng1.py), [tng1_app](../tng1_app.py) |

Versions come from [registry.py](../registry.py). Some source comments and local module version strings refer to older implementations; they do not override these active definitions.

- Byte fields are sent most-significant bit first through NumPy's default `unpackbits`/`packbits` ordering. Two-byte lengths/counts and byte-oriented CRC values use big-endian order.
- Array positions below are zero-based. Code bits are read left to right.
- A **segment** is a TNG44/TNG5 CRC unit. A learned-modem **frame** carries 96 coded bits and lasts 192 ms. A TNG1 **block** is a different structure: 92 tone symbols lasting 14.72 seconds.
- The default audio center is 1500 Hz. TNG44/TNG5 use 2000 complex baseband samples/s; TNG1 uses 1000 samples/s. Audio is obtained by resampling and taking `real(baseband * exp(j*2*pi*center*t))`.
- Center-frequency selection, output gain, audio resampling and optional VOX lead tone/PTT delay operate outside the information coding. See [modem](../modem.py), [freqshift](../freqshift.py) and the UI transmit path in [neuromod_app](../neuromod_app.py).
- Live reception uses the selected mode only. These are independent communication modes, not an automatic receive-everything protocol.

The 96-bit legacy header/payload format described at the top of [framing.py](../framing.py) is **not** the active TNG44/TNG5 segment format. The active paths reuse its CRC function; their framing is defined separately below.

## Common transmit path

The common outline is:

```text
Text -> byte or character encoding -> segmentation / block framing
     -> CRC -> FEC -> mode-specific ordering and waveform processing
     -> synchronization-bearing waveform -> audio output to SSB transmitter
```

This outline groups functions by purpose; it does not impose an identical pipeline on all modes. In particular, **TNG5 whitening occurs after CRC calculation but before FEC**, and TNG1 inserts sync symbols into its tone sequence before modulation.

| Stage | What the active implementation does |
|---|---|
| Text encoding | TNG44 removes NUL and encodes UTF-8 bytes. TNG5/TNG1 normalize restricted text and pack complete public Huffman character codes. |
| Framing | TNG44 prefixes a payload-byte count and divides the padded byte stream into segments. TNG5 uses fixed-size segments with a segment count in the first one. TNG1 packs independent text blocks and continues until EOT is included. |
| CRC | Each segment or block gets a public checksum for error detection. CRC does not correct an error or authenticate a sender. |
| FEC | TNG44 continuously convolutionally encodes the segment stream. TNG5 encodes and repeats each segment codeword. TNG1 tail-biting encodes each block into tone labels. FEC adds redundancy so a receiver can use uncertain observations to recover information. |
| Mode-specific ordering | All modes interleave coded bits or symbols. TNG5 alone whitens segment data before FEC. Interleaving spreads neighboring channel errors over the decoder's input; it adds no secret mapping. |
| Synchronization and modulation | TNG44/TNG5 map coded bits and guards through their public waveform models, then assemble mode-specific lead/sync signals around or within the data stream. TNG1 inserts known sync tones throughout each block, then modulates the entire tone sequence with chirped 16-GFSK. |
| Audio output | The baseband waveform is resampled and shifted to the selected audio center for the radio's SSB audio input. Output gain and optional PTT/VOX timing belong to the application/radio interface. |

Interleaving and FEC protect different aspects of reception: a fade may damage consecutive transmitted positions, while interleaving redistributes their observations before error correction. Neither ensures that a lost segment or block can be recovered.

## Common receive path

```text
Received audio -> complex baseband -> synchronization / timing / frequency acquisition
               -> soft demodulation -> deinterleaving
               -> mode-specific combining -> FEC decoding
               -> mode-specific de-whitening -> CRC validation
               -> byte / character decoding -> recovered text
```

Soft demodulation preserves uncertainty rather than immediately deciding every bit or tone. TNG44/TNG5 use signed bit log-likelihood values (LLRs); TNG1 uses metrics for the possible tone labels. Viterbi decoding uses these observations to select an information sequence consistent with the convolutional code.

- **TNG44:** detect start/end sync, estimate timing and frequency, obtain model LLRs, deinterleave and continuously Viterbi-decode. Check segment CRCs, recover the byte-count field when possible, then assemble valid bytes and decode UTF-8.
- **TNG5:** acquire start/end or intermediate sync, exclude mid-sync samples from data windows, obtain model LLRs and deinterleave. Sum four aligned codeword copies, Viterbi-decode each segment, undo whitening, then check CRC against the original data and Huffman-decode valid segments. De-whitening cannot precede FEC decoding because the whitening covers information bytes, not the transmitted coded bits.
- **TNG1:** use scattered sync tones to acquire blocks, de-chirp audio and measure tone energies, restore symbol order, and tail-biting Viterbi-decode. Check CRC and receiver confirmation rules before Huffman decoding. Failed candidates can be retried or combined with other observations as described below; TNG1 has no de-whitening step.

Only CRC-confirmed content is eligible for text display, with additional acquisition/confirmation checks where required by the receiver. Missing portions can leave placeholders or a missing-prefix indicator. Recovering a later segment or block does not by itself recover the earlier text.

## Registered reference rates

These values are registered in [registry.py](../registry.py), not new RF measurements. The data-section rate is a long-message incremental reference; TNG5 includes recurring mid-sync overhead. The whole-transmission reference uses a 24-character message, including synchronization. Short messages, UTF-8 byte lengths, Huffman code lengths and reception failures change effective throughput.

| Mode | Data-section characters/s | Information bits/s | Whole-transmission characters/s (24-character reference) | Information bits/s (same reference) | Registered −60 dB occupied bandwidth |
|---|---:|---:|---:|---:|---:|
| TNG44 | 29.5 | 236 | 11.2 | 89 | 470 Hz |
| TNG5 | 4.8 | 24.6 | 3.2 | 17.0 | 423 Hz |
| TNG1 | 0.90 | 4.35 | 0.82 | 4.08 | 356 Hz |

TNG44 character rates count ASCII bytes; arbitrary Unicode text has no fixed characters/s conversion. TNG5/TNG1 rates depend on the shared Huffman encoding and message content. Registered interleaver spans are 3.2 s, 9.6 s and 14.7 s respectively; TNG1's value describes its block-local interleaver, not an additional delay after the block.

## Public text mappings

### TNG44: UTF-8

The transmitter removes U+0000, encodes the remaining string with Python UTF-8, and carries those bytes without a Huffman character table. An empty direct encoder input is represented by one space; the UI rejects empty messages. ASCII occupies one byte per character; a Hangul syllable normally occupies three bytes.

The leading length field is a **byte count**, not a character count. Successful reception assembles the payload bytes and decodes UTF-8. For partial reception, adjacent valid segments are joined before decoding; incomplete UTF-8 sequences at a missing-segment boundary are discarded and failed segments are represented by a display placeholder. The placeholder is a receiver display convention, not a transmitted character.

### Shared TNG5/TNG1 canonical Huffman codebook

Both modes use [charset1.py](../charset1.py), with 57 input characters plus an internal EOT terminator. Input normalization is exactly:

```python
t = text.upper().replace("\r\n", "\n").replace("\r", "\n").replace("\t", " ")
```

Characters outside `CHARS` are reported and removed by normalization. The UI warns before transmitting text with those characters omitted. Neither mode carries general UTF-8 text.

The public symbol order is:

```python
CHARS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 \n.,?'/-=()!:+\"@&;*#%"
SYMS = CHARS + "\x04"  # internal EOT
```

The following is the exact current `charset1.CODE` table, extracted from that source. EOT is an internal control symbol and is not an additional user-input character.

| Symbol | Code bits (left to right) |
|---|---|
| `A` | `0100` |
| `B` | `101100` |
| `C` | `01100` |
| `D` | `101101` |
| `E` | `0101` |
| `F` | `1110010` |
| `G` | `101110` |
| `H` | `01101` |
| `I` | `01110` |
| `J` | `11110010` |
| `K` | `101111` |
| `L` | `01111` |
| `M` | `110000` |
| `N` | `10000` |
| `O` | `10001` |
| `P` | `110001` |
| `Q` | `110010` |
| `R` | `10010` |
| `S` | `10011` |
| `T` | `10100` |
| `U` | `110011` |
| `V` | `11110011` |
| `W` | `110100` |
| `X` | `110101` |
| `Y` | `1110011` |
| `Z` | `11110100` |
| `0` | `1110100` |
| `1` | `110110` |
| `2` | `110111` |
| `3` | `1110101` |
| `4` | `11110101` |
| `5` | `1110110` |
| `6` | `11110110` |
| `7` | `1110111` |
| `8` | `111111010` |
| `9` | `11110111` |
| SPACE (U+0020) | `00` |
| LF (U+000A) | `111000` |
| `.` | `10101` |
| `,` | `11111000` |
| `?` | `11111001` |
| `'` | `111111111000` |
| `/` | `11111010` |
| `-` | `11111011` |
| `=` | `1111111100` |
| `(` | `111111011` |
| `)` | `111111100` |
| `!` | `111111101` |
| `:` | `11111100` |
| `+` | `111111111001` |
| `"` | `111111111010` |
| `@` | `111111111011` |
| `&` | `111111111100` |
| `;` | `111111111101` |
| `*` | `111111111110` |
| `#` | `111111111111` |
| `%` | `1111111101` |
| EOT (U+0004, internal terminator) | `1111000` |

The table is deterministic, not negotiated or generated from a private seed. For regeneration, `charset1._lengths` builds a Huffman tree from the public `_CNT` counts plus 0.5 for each symbol, breaking heap ties with public symbol/insertion order. `_canonical` sorts by (code length, index in `SYMS`), starts at code 0, left-shifts when the length increases, and increments the code after each assignment.

A block/segment is packed by appending complete character codes while they fit; character codes never cross a packing-unit boundary. EOT is inserted after the final text if it fits. Remaining bits are ones. Decoding stops on EOT and discards an unfinished code at the unit end. The longest code is 12 bits. TNG1 sends another block when necessary to fit EOT; TNG5's segment count also delimits its message, so a final segment need not contain EOT if it does not fit.

## TNG44: Standard

### Design purpose and trade-offs

TNG44 targets faster text exchange on relatively good HF channels and prioritizes throughput among the three modes. Its byte-oriented UTF-8 transport carries unrestricted text without reducing it to the robust modes' character table. A payload-byte count lets the receiver delimit a message independently of the number of displayed characters.

The active path uses segment CRCs, a continuous K7 R1/2 convolutional code, a delay interleaver and explicit start/end synchronization. It spends less airtime on redundancy than TNG5's R1/4 code with four full copies, and it does not use TNG1's long tone symbols and repeated-content recovery. The resulting trade-off is less redundancy per information bit, with more text carried in the same time; this does not establish a universal performance ranking.

The learned signal-processing components are active in this mode: `StreamTransmitter` maps coded bit pairs and their frame positions to I/Q samples, while `ContextReceiver` converts contextual I/Q windows to soft bit values. They implement waveform mapping and demodulation, leaving framing, CRC and FEC to explicit algorithms. The [public model section](#tng44tng5-waveform-mapping-and-public-models) defines their technical role and compatibility requirements. The registered 29.5 ASCII characters/s is a long-message incremental rate, not the meaning of “44.”

### Transmit information path

```text
text -> remove NUL -> UTF-8 bytes
     -> 2-byte payload-byte count + payload
     -> capacity padding -> segments + CRC16
     -> continuous K7 R1/2 FEC + final tail
     -> deterministic delay interleaver -> 96-bit frames
     -> leading/trailing guard -> public waveform model
     -> tone + start sync + modem data + end sync -> audio
```

1. Prefix the payload with its two-byte big-endian length. The first segment therefore contains up to 30 text bytes after the length field; later full segments carry 32 text bytes.
2. Select the smallest number of 96-bit frames that fits the unpadded segment stream and FEC tail. Pad the length-prefixed byte stream with NUL bytes to `capacity(n_frames)`, the maximum byte count whose segment CRCs and FEC tail fit those frames.
3. Split this byte stream into up-to-32-byte segments. Append two CRC bytes to **each segment**, including its padding and, in the first segment, its length field.
4. Concatenate all segment-and-CRC bits, append **six zero tail bits once at the end of the message**, and apply the convolutional encoder continuously across segment boundaries. Segments are not independent FEC codewords.
5. Interleave the coded bits. Fill any remaining coded-frame positions with the fixed public NumPy sequence generated by `default_rng(0xF111).integers(0, 2, count)`. This fill is outside the encoded information/CRC.
6. Add one 96-bit guard frame before and after the data frames.

For payload length `L` bytes, before capacity padding:

```text
N = L + 2
S = ceil(N / 32)
coded_bits = 2 * (8 * (N + 2*S) + 6)
n_frames = ceil(coded_bits / 96)
```

`capacity` applies the same coded-length rule to possible padded byte counts. The guard frames are not included in `coded_bits` or `n_frames`.

### CRC and FEC

The segment CRC is the byte-oriented CRC-16/CCITT-FALSE function in [framing.py](../framing.py): polynomial `0x1021`, initial register `0xFFFF`, non-reflected input/output, no final XOR. The CRC is appended high byte first.

FEC is K=7, R1/2 with generators **0o133, 0o171 in that output order**. The initial six-bit state is zero. For input bit `u` and state `s`, `reg=(u<<6)|s`; emit the parity of `reg & generator` for each generator and update `s=reg>>1`. The six final zero inputs return the state to zero. See [fec.py](../fec.py) and `link7.conv_encode_rows`.

There is no payload whitening stage or dedicated repeated-message combiner in the active TNG44 path. XOR inside CRC/parity calculations is error-control arithmetic, not encryption.

### Interleaving and guards

For coded length `n`, `link7.interleaver_src` defines:

```python
p = arange(n)
tau = p + (p % 12) * 12 * 24
order = lexsort((p, tau))  # primary tau, secondary p
q = empty(n, integer)
q[order] = arange(n)
transmitted[q[p]] = coded[p]
```

Reception reads `received[q]` to restore coded order. There is no random/private interleaver seed; the registered span is approximately 3.2 seconds.

The current streaming guard is the public base guard with every odd-indexed bit inverted, equivalent to XORing each packed byte with `0x55`. Its packed bytes are:

```text
0f 96 c3 3c 69 f0 87 7e 5a a5 66 99
```

This guard-pattern transformation selects a public framing signature; it is not payload whitening or secret mapping.

### Synchronization and receiver

At baseband, the stream contains a 300 ms zero-offset lead tone, start Costas sequence A, the modem stream including its two guards, and end Costas sequence B:

```text
A = (3, 1, 4, 0, 6, 5, 2)
B = (0, 6, 4, 5, 1, 3, 2)
tone offset for index k = (k - 3) * 62.5 Hz
symbol duration = 32 ms; each seven-symbol sync = 224 ms
```

Synchronization uses continuous phase with 4 ms frequency-transition smoothing. [preamble.assemble](../preamble.py) applies the public 200 Hz/129-tap LPF to sync waveforms, overlaps filter tails with neighboring regions, and smooths the data/sync boundaries; it does not insert another information field.

The receiver downconverts audio, correlates public sync templates over time/frequency offsets, corrects frequency and evaluates contextual modem windows for 96 bit LLRs per frame. It deinterleaves, performs soft Viterbi decoding and verifies each segment CRC. The first valid segment supplies the payload-byte count; end sync can also determine the received duration or trigger recovery from buffered audio when start sync was missed. Alignment hypotheses and relocking are receiver algorithms, not secret mappings. Partial text is displayed only from CRC-confirmed segments. There are no periodic mid-sync join markers.

The registered 29.5 characters/s is an ASCII long-message **incremental** rate, including recurring CRC/FEC overhead; fixed lead/sync/guard overhead cancels in that comparison. For example 240 and 960 ASCII characters require 44 and 171 data frames: `720 / ((171-44)*0.192) = 29.5276`. It is not a general UTF-8 or whole-message throughput guarantee.

## TNG5: Robust

### Design purpose and trade-offs

TNG5 targets weak-signal and fading conditions by spending more airtime on each unit of text. Restricting input to the public 57-character plus EOT Huffman codebook reduces the information needed for typical supported text, while K7 R1/4 FEC and four full codeword copies supply substantially more redundancy than TNG44. The longer interleaver distributes the copies and coded observations over time, giving the decoder information from different parts of a fading interval. The cost is lower throughput and longer waits for sufficient observations.

Each segment packs complete characters independently. This avoids carrying Huffman decoding context across a missing segment and permits valid later segments to produce partial text. Public fixed-seed XOR whitening reduces long identical bit patterns before error coding; its seed **4040** and exact mask are specified below. It is deterministic and reversible and has no secrecy function.

Periodic mid-sync markers give the receiver new timing/frequency references during a long transmission and support attempts to join after missing the start. A join still needs sufficient coded observations and CRC confirmation; the receiver cannot reconstruct samples that were never captured. TNG5 uses its own public waveform-model weights, so it is not TNG44 slowed down by a speed control. Its four codeword copies belong to a single transmission, rather than the later-content combining used by TNG1.

### Transmit information path

```text
normalized text -> public Huffman codes
 -> 18-byte segments (first includes 2-byte segment count)
 -> CRC16 of original segment
 -> XOR whitening of the 18 data bytes only
 -> whitened data + unchanged CRC -> six zero tail bits
 -> K7 R1/4 convolutional code -> four complete copies
 -> public interleaver -> 96-bit frames + guards -> waveform model
 -> lead tone/start sync, periodic mid-sync, end sync -> audio
```

The first segment contains the total segment count as two big-endian bytes and **128 Huffman text bits**. Later segments have **144 Huffman text bits**. Each segment is exactly 18 bytes before CRC. The count includes the first segment; it is not a byte or character count. Characters are packed independently at segment boundaries using the shared codebook, EOT-if-it-fits and one-bit padding rules.

### CRC and public XOR whitening

The original, unwhitened 18-byte segment is protected by the same byte CRC-16/CCITT-FALSE as TNG44: `0x1021`, initial `0xFFFF`, non-reflected, no final XOR; high byte first.

[tng5.py](../tng5.py), `WHITEN`, `whiten` and `seg_bits` specify:

```python
WHITEN = np.random.default_rng(4040).integers(
    0, 256, 18
).astype(np.uint8).tobytes()
whitened = bytes(x ^ y for x, y in zip(segment, WHITEN))
wire_information = whitened + crc16_ccitt(segment).to_bytes(2, "big")
```

The exact current public mask, in hexadecimal byte order, is:

```text
78 22 bf 5a 6f 0e 1e 81 18 f6 20 7e 46 1a 1e 56 1e 6f
```

This is **fixed public seed 4040**, reversible XOR whitening. The mask restarts at byte zero for every segment and covers all 18 segment bytes, including the first segment's count field and padding. The CRC bytes are not whitened.

**The purpose is bit-pattern whitening, not secrecy; no secret key is used.** The code identifies long repeated bit patterns as the reason for this stage. Reception applies the same XOR to the recovered 18 bytes, then checks CRC against the resulting **de-whitened data**. No password, negotiated key, random private nonce or runtime secret mapping enters this operation.

### FEC, repetition, interleaver and guards

Each segment contains 160 information bits (18 data + 2 CRC bytes). Append six zero tail bits, then encode from state zero with K=7, R1/4 generators:

```text
(0o117, 0o127, 0o155, 0o171)
```

Use `reg=(u<<6)|s`, emit parity in that generator order, and update `s=reg>>1`. This creates 664 coded bits. Repeat the **entire 664-bit codeword four times**, creating 2656 bits per segment (effective R1/16). After deinterleaving, the receiver sums the four aligned LLR copies before Viterbi decoding; this is within-transmission FEC repetition, not TNG1's later-transmission combining.

For total coded length `n=2656*n_segments`, the exact mapping is:

```python
perm = np.random.default_rng(55).permutation(2656)
base = concatenate([perm + k*2656 for k in range(n_segments)])
p = arange(n)
tau = p + (p % 16) * 16 * 18  # floor(4800 / (16*16)) = 18
order = lexsort((p, tau))
q = empty(n, integer)
q[order] = arange(n)
mapping = q[base]
transmitted[mapping] = coded
```

The receiver selects `received[mapping]`. The public fixed seed is **55**; the default `ILV_SPAN` is 4800, with a registered span of about 9.6 seconds. This permutation distributes errors and separates repeated copies; it does not conceal the message.

Pad the last 96-bit frame with `default_rng(n).integers(0,2,count)`; the seed is the public coded length `n`. Prepend/append the public 96-bit guard generated by `default_rng(5005).integers(0,2,96)`:

```text
100110011111110000110011110110111101110011111011111101011100110000100101001010111011100100101010
```

Padding and guards are not CRC-bearing user information.

### Synchronization and shaping

The current transmit structure is:

```text
300 ms lead tone | start A12 | guard | data frames with M_k | guard | end B12
```

- Lead tone offset: `-2.5 * 125/3` Hz (approximately −104.1667 Hz).
- Start A12: `(1,3,7,2,5,11,10,8,4,9,6,0)`, 64 ms per tone, 31.25 Hz spacing, total 768 ms.
- End B12: `(9,8,6,0,5,1,11,3,4,7,2,10)`, 32 ms per tone, 31.25 Hz spacing, total 384 ms.
- Mid-sync `M_k` is inserted **before data frame 20*k only when 20*k < n_frames** (zero-based frames, guards excluded), lasting 336 ms: seven tones of 48 ms, spacing `125/3` Hz. Four patterns cycle by `(k-1)%4`:

```text
M1: (1,4,3,5,2,0,6)
M2: (2,0,6,5,1,3,4)
M3: (2,1,6,4,0,3,5)
M4: (2,3,5,0,4,1,6)
```

For sync sequence index `v`, the frequency offset is `(v-(max(sequence)+min(sequence))/2)*spacing`. The sync modulation uses continuous phase with Gaussian frequency shaping, **BT=0.5**, and **+2.25 dB** amplitude-relative-to-data boost as implemented by `10**(BOOST_DB/20)`. The sync sequences are sent once; the old A×4/B×4 sync repetition is not the current structure.

[tng5.build_tx](../tng5.py) filters the assembled waveform with the public 200 Hz/129-tap LPF. It clips data/guard envelope peaks (excluding mid-sync) at data RMS +3 dB and applies the LPF for three iterations. These waveform operations do not add encrypted fields.

### Receive reconstruction and joining

The receiver detects start/end sync and mid-sync patterns, estimates timing/frequency, and supplies data-only windows to the contextual receiver, skipping inserted mid-sync samples. It restores the public interleaver order, sums the four coded copies, runs terminated soft Viterbi decoding, removes six tail bits, repacks bytes, **de-whitens**, verifies CRC and Huffman-decodes each valid segment. The first segment's count determines message extent; failed segments use display placeholders.

Start confirmation uses the expected first mid-sync or a valid segment CRC. The active `M_JOIN=True` path also considers intermediate-sync hypotheses when the start was missed, using CRC to confirm a join. Buffered missing-prefix samples are erasures, not secret data. Joining can recover partial text and can fail; it does not guarantee recovery of the missing beginning. Receiver retries can combine LLR observations, but no TNG1-style cross-message repeat-combining protocol is defined for TNG5.

## TNG1: Robust+

### Design purpose and trade-offs

TNG1 prioritizes recovery in difficult weak-signal and fading channels over conversational speed. It uses the same 57-character plus EOT Huffman path as TNG5, but each block carries only 64 text/control/padding bits in 14.72 seconds. Long 160 ms tone symbols, K13 R1/4 tail-biting FEC and block-local interleaving accept a very low text rate to give the receiver longer observations and a different error-control structure.

Known synchronization tones are scattered through every block. The receiver can accumulate acquisition evidence from several blocks and attempt to join without receiving a separate start preamble. This sync accumulation uses the repeated known layout; it does not assume that successive blocks carry the same text. Each block's CRC and the receiver's confirmation checks determine whether its decoded content can be accepted.

For repeated content, failed tone-energy observations can be retained and combined in pairs or triples. A weak observation may still contribute useful evidence for symbols that were stronger in another copy. Combining requires CRC success and agreement checks on individual observations, so unrelated or unusable candidates are not accepted merely because their energies were added. Resend provides another transmission of the previous text; it does not create an automatic acknowledged retransmission exchange.

The current TNG1 path uses chirped 16-GFSK, spectral energy measurements and explicit convolutional decoding. It uses **no neural network or model checkpoint**. Its acquisition latency, partial reception and combining limits remain conditional on captured audio and receiver state.

### Information, CRC and convolutional tone labels

```text
normalized text -> public Huffman packing into 64-bit units
 -> append 16-bit CRC -> 80 information bits
 -> tail-biting K13 R1/4 -> 80 labels in 0..15
 -> public 80-symbol interleaver
 -> insert 12 public sync tones -> 92 tone symbols
 -> 16-GFSK with chirp -> audio; repeat blocks as needed
```

Each block carries 64 text/control/padding bits and 16 CRC bits. The shared Huffman codes are packed without crossing block boundaries. The transmitter continues until EOT has actually been inserted; it can append an EOT-only block. Blocks have no sequence-number field, secret address field or retransmission marker.

[tng1.crc16_bits](../tng1.py) processes all 64 packed text bits MSB-first with polynomial **0x1021**. Normally the initial register is **0xFFFF**. The first block of a message uses **0x5A5A**, a public first-block indication introduced by current wire revision 2. Each input bit is XORed with the register's top bit, the register shifts left, and the polynomial is XORed when that result is one. There is no reflection or final XOR. Append the resulting register's 16 bits MSB-first. The receiver accepts the appropriate CRC initialization to distinguish a first block; these initial values are not keys.

The tail-biting convolutional encoder has **K=13**, 4096 states and generators in this exact order:

```text
(0o13635, 0o12055, 0o15727, 0o17453)
```

Unlike TNG44/TNG5, it appends no zero termination tail. Initialize the state by feeding the last 12 bits of the 80-bit information-plus-CRC sequence through the state update. For each bit `u`:

```text
reg = (u << 12) | state
label = concatenate parity(reg & g) for the four generators, MSB first
state = reg >> 1
```

The label is an integer **0..15** and directly selects a tone. There is no additional Gray or private substitution table. The final state equals the tail-biting initial state.

### Fixed block layout and interleaver

Each block has **80 data symbols + 12 synchronization symbols**, at **160 ms per symbol**, totaling **92*0.160 = 14.72 seconds**. The same layout is used in every block.

[tng1.layout](../tng1.py) uses public NumPy `default_rng(11)`. It consumes a 12-value uniform draw for sync-position jitter, then a 16-value tone permutation, then an 80-value data-symbol permutation. No runtime entropy or secret seed is supplied. For this configuration the exact arrays are:

```text
sync positions = [3,11,19,26,34,43,49,57,66,73,80,88]
sync tones     = [4,15,0,13,6,9,11,10,2,1,12,8]
```

Fill the remaining block positions, in ascending position order, with `encoded_labels[ilv]`, where:

```text
ilv = [16,36,62,78,33,24,18,39,6,32,20,59,53,47,70,55,48,64,56,76,65,41,77,15,40,71,63,52,9,37,38,30,13,68,21,43,60,3,11,74,26,35,4,34,29,7,75,73,25,2,67,28,5,44,31,1,79,49,58,72,54,45,23,17,69,14,46,57,61,19,8,51,22,10,50,66,12,0,27,42]
```

Receiver restoration uses `original_metrics[ilv] = transmitted_data_metrics`. This is a block-local permutation; it is not a 14.72-second added delay or a cryptographic shuffle.

There is **no byte whitening/scrambling stage** in active TNG1. Its CRC/FEC XOR arithmetic and deterministic layout are public error-control/mapping operations.

### Actual modulation mapping

TNG1 uses **16-GFSK with an upward per-symbol chirp**, not a neural model. For tone label `m`, symbol duration `T=0.160` seconds:

```text
slot[m] = m - 7.5
nominal tone frequency offset = slot[m] / T = (m-7.5)*6.25 Hz
```

At 1000 baseband samples/s, repeat each nominal tone frequency for 160 samples and Gaussian-smooth the tone-frequency stream with **BT=2**. The public chirp samples are:

```text
chirp[n] = 50 * ((n+0.5)/160 - 0.5) Hz, n=0..159
```

The chirp is added per symbol, with its return smoothed separately at **BT=8** (`chirp_smooth=False`, `chirp_bt=8`). Integrate the sum to phase: `phase=2*pi*cumsum(f)/1000`; output `exp(j*phase)`. [tng1.gfsk](../tng1.py) defines the Gaussian kernel truncation and endpoint extension exactly. The stream has 20 ms start/end ramps, then is resampled and shifted to the selected audio center.

There is no separate start preamble, end Costas sequence or TNG5-style mid-sync. Twelve scattered synchronization tones in **every block** support block acquisition and joining without scheduled time slots.

### Receive reconstruction and repeat combining

Audio is downconverted to 1000-sample/s complex baseband. The receiver applies impulse blanking, computes de-chirped spectral energies and folds the known sync positions across up to four blocks to find timing/frequency candidates. A symbol window is de-chirped and shifted by the known fractional-tone offset before FFT energy is read at the 16 public tone bins.

Data-symbol metrics are deinterleaved and decoded by tail-biting soft Viterbi; the receiver checks the normal and first-block CRC rules, then Huffman-decodes the 64 information bits. If the initial CRC fails, the configured multi-symbol retry uses a two-symbol neighborhood on either side (`msd=2`). Accepted blocks must also meet receiver sync/confirmation rules. EOT ends a message; signal-loss handling can end an incomplete one. A receiver may join a later block and display a missing-prefix indicator.

The **Resend** action sends the last TNG1 text again from a new block boundary. There is no on-air retry number or hidden repeated-message identifier. `Tng1Stream` stores failed candidate energies and `Combiner` attempts pairs and triples using summed symbol metrics. It requires a valid CRC and per-copy agreement with the re-encoded candidate; current `Tng1Stream` sets `zmin=5`, keeps at most 16 ranked candidates and ages them after 600 seconds. Copy similarity limits which combinations are attempted. This can improve recovery of matching repeated content; it is not guaranteed and is not encryption.

## TNG44/TNG5 waveform mapping and public models

The actual data modulation is specified by the public code **and the bundled numeric weights**, rather than a fixed four-point constellation table. The mapping is position-dependent. The code and weights publicly define this mapping.

| Mode | Public checkpoint | SHA-256 |
|---|---|---|
| TNG44 | [stage3.pt](../models/stage3.pt) | `e73e59a5e2d7dade6907140a6243eb87793b48dcea6570be543a5cf32533d30c` |
| TNG5 | [stage3_strong.pt](../models/stage3_strong.pt) | `2cbd875d00c3408459d147a6dac28d8a09713c0592b2253b81e1115e59981c4c` |

Both supplied checkpoint configurations specify 2000 Hz baseband, 8 samples per modulation symbol, 2 coded bits per symbol and 48 symbols per frame. Thus 96 coded bits map to 384 samples (192 ms). They use a 16-value position embedding, transmitter hidden widths (256,256), 200 Hz/129-tap fixed FIR, and receiver context of four symbols on each side (448-sample receive windows).

[models3.StreamTransmitter](../models3.py) defines the mapping:

1. Group coded stream bits into ordered pairs; prepend/append the appropriate public 96-bit guard before grouping.
2. For modulation-symbol index `j`, use position embedding `pos[j % 48]` and concatenate it with `2*bits-1`.
3. Apply the checkpoint's MLP: Linear, GELU, Linear, GELU, Linear, yielding 16 real outputs for eight complex samples. Reshape each adjacent real-output pair as I and Q.
4. Concatenate those samples into the stream, apply the checkpoint's public fixed FIR **once across the whole stream**, and normalize by its mean complex power.
5. [modem5._frames_to_baseband](../modem5.py) applies the 20 ms boundary ramp and renormalizes; each mode then adds its synchronization and shaping described above.

The FIR coefficients and their normalization are generated publicly by [waveform.design_lpf](../waveform.py), and its buffer is also in the checkpoint. Runtime loading replaces the transmitter/receiver parameters with the bundled state dictionary; no runtime-generated secret mapping is selected.

[models3.ContextReceiver](../models3.py) uses the public FIR and RMS normalization, two-channel I/Q convolution layers, symbol downsampling, residual blocks, global context and a two-bit-logit output head. Its numeric weights define soft bit demodulation. [modem4._rx_windows](../modem4.py) flattens those logits in coded-bit order. Positive LLR values favor bit 1. FEC/CRC then reconstruct and validate the public text encoding.

The full tensor values are in the linked public checkpoints, not transcribed as thousands of artificial constellation entries in this document. Matching wire revisions and model identifiers are necessary for compatible TNG44/TNG5 communication. TNG1 needs no checkpoint.

## Design and development evidence

The published source supports a software validation approach to coding, synchronization and waveform recovery. [modem5.cmd_loopback](../modem5.py) generates a waveform, converts it to audio, applies frequency offset, optional sound-card clock mismatch and band-limited Gaussian noise, decodes the result, and compares recovered text with the input. [modem.add_band_noise](../modem.py) defines the noise injection used by that path.

[tng1.channel](../tng1.py) provides AWGN and two-path Watterson-style fading simulation, with Gaussian Doppler gains, delay, frequency offset/drift, level changes and optional impulse noise. These functions show which impairments can be simulated in the public implementation. They are validation helpers, not extra transmitted fields or the radio channel itself.

Source comments describe parameter comparisons for synchronization, clipping and character packing, but many referenced experiment scripts and result logs are absent from this public tree. Those comments do not establish reproducible success rates, channel limits or completed RF validation. No unseen development records or older configurations are used here to define the current protocol.

## Audit of public transformations and secrecy

The active encoding and recovery paths use the public transformations listed below. They provide no secret-key message encryption, private character mapping or message-concealment feature. Character tables, the full Huffman codebook, framing, CRC/FEC, interleavers, whitening seeds/algorithms and modulation mappings are available in this document and the linked source. Anyone can inspect those definitions to analyze the on-air data structure; this openness does not guarantee successful decoding of damaged audio.

| Active public operation | Location | Seed / initialization | Communication purpose |
|---|---|---|---|
| TNG5 reversible data XOR whitening | `tng5.WHITEN / whiten / seg_bits / decode_llr`; `tng5_app.decode_one` | Fixed public seed **4040**, exact 18-byte mask above | Reduce long repeated bit patterns; reversible before CRC verification |
| TNG44 current guard inversion | `link7.GUARD_V2` | Fixed alternating bit mask (packed `0x55`) | Identify current streaming framing signature |
| TNG44 delay interleaver | `link7.interleaver_src` | No RNG; J=12, M=24 | Distribute coded bits over time |
| TNG44 final coded-frame fill | `link7.encode_raw`, `stream_ui.encode_bytes` | Public **0xF111** | Fill unused non-information frame bits |
| TNG5 within-segment permutation | `tng5._PERM / interleave_map` | Public **55** | Separate/distribute coded copies |
| TNG5 guard | `tng5.GUARD5` | Public **5005** | Fixed modem guard waveform |
| TNG5 final coded-frame fill | `tng5.encode_bits`, `tng5_app.frames_from_segs` | Public coded length **n** | Fill unused non-information frame bits |
| TNG1 sync and symbol interleaver | `tng1.layout` | Public **11**, arrays above | Known acquisition tones and error distribution |
| CRC/FEC parity XOR | `framing`, `fec`, `link7`, `tng1`, `conv_sim` | Public polynomial/state rules above | Error detection/correction |
| Model and charset SHA-256 | `registry` | Public files/table contents | Compatibility identification; not payload encryption |

No additional active payload whitening/scrambler was found. Legacy file decoders and internal diagnostic/example noise generation are not the current transmit format. Sorting/cache/UI uses of the name `key` are not cryptographic keys; UI colormap lookup tables are not character mappings. Model training seed metadata is not used to generate a secret runtime on-air mapping.

This document describes communication behavior, not the legal applicability of amateur-radio regulations in any jurisdiction. It makes no claim that CRC authenticates a sender, that FEC provides confidentiality, or that all channels can be decoded.

## Implementation references and limits

The text tables, byte/bit layouts, CRC/FEC rules, permutations, whitening mask, synchronization sequences and modulation rules above cover the major active on-air structure. Exact DSP kernel construction, resampling, floating-point weights and receiver candidate thresholds remain defined by the linked public source and checkpoint files. No private external table is required.

Current live transmission uses only the revisions in the opening table. TNG5 wire revision 1 has a file-only decoder; revisions 2 and 3 are not decoded into text by the current implementation. TNG1 wire revision 1 is supported through its file-only fallback. Legacy WAV fallbacks do not imply live interoperability with older revisions. Receiver acquisition, combining and joining are conditional algorithms; their description is not a promise of successful reception under every noise/fading condition.
