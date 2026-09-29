# TNG PKT

[English](README.md) | [한국어](README.ko.md)

TNG PKT is a Windows amateur radio text-mode application for text communication over HF SSB audio, with modes intended for different noise and fading conditions. It sends and receives text through an SSB audio path and provides three independent communication modes: **TNG44**, **TNG5** and **TNG1**.

**Current public-source candidate: 0.11.2 Beta / Build 1009**  
Author: **6L5TNG**

## Features

- Live text transmission and reception through selectable audio input and output devices.
- A shared mode selector for transmitting and receiving, with adjustable audio center frequency.
- Callsign macros, transmit progress, manual stop and TNG1 **Resend**.
- Live waterfall and spectrum displays, packet timelines and mode-specific analysis panels.
- Dockable panels, layout presets and saved window layouts.
- WAV file decoding and TNG44 transmit WAV export.
- Receive history, session logs and CSV export.
- Korean and English interfaces, first-run setup and CPU/audio diagnostics.
- Optional radio frequency control and PTT through Hamlib or serial control lines.

## Modes

The modes have different speed and robustness goals. Both stations must select the same mode.

| Mode | Intended role |
|---|---|
| **TNG44** | **Standard / faster** — text communication on relatively good HF channels |
| **TNG5** | **Robust** — lower text rate with stronger error protection for weak signals and fading |
| **TNG1** | **Robust+** — slow block-based communication that prioritizes recovery in difficult channels |

The numbers in the mode names are not current measured character rates. Reception depends on the channel, radio/audio setup and receiving PC.

TNG44 supports UTF-8 text, including Hangul. TNG5/TNG1 support a shared 57-character set of uppercase letters, digits, space, newline and selected punctuation. Lowercase is converted to uppercase and tabs to spaces; unsupported characters trigger a warning before transmission and are removed if the user confirms the same text again.

## Open communication

TNG PKT is intended for open amateur-radio communication. It does not provide message encryption or secret-key communication. Its on-air encodings are public; error correction and whitening do not provide secrecy.

For protocol design, transmit/receive processing, framing, error correction, character mappings and on-air encoding, see [Protocol Documentation](docs/PROTOCOL.md).

## Install and run

The release configuration targets **Windows x64**. The packaged application uses CPU inference and does not require a GPU or a separate Python installation.

Use the matching `TNG-PKT-Setup-0.11.2.exe` installer when available with a binary release. Start-menu shortcuts are installed; a desktop shortcut is optional. See [Installation and Windows builds](docs/INSTALL.md) for source execution, packaging and the migration instructions for older Build 1005 installations.

To run this source on Windows with **Python 3.13**, open PowerShell in the source root:

```powershell
python -m venv .venv
& .\.venv\Scripts\python.exe -m pip install -r requirements.txt
& .\.venv\Scripts\python.exe neuromod_app.py
```

On first launch, choose a language, enter your callsign, select the audio input/output connected to your radio and configure a radio connection if needed. The setup also provides CPU/audio diagnostics.

For a contact, both stations select the same mode and compatible audio center frequencies. The default audio center is **1500 Hz**. Use the mode selector and center-frequency control to configure operation before sending text.

Settings, layouts and runtime logs are stored in `%APPDATA%\TNG PKT`. Updates are installed manually; automatic update checking and installation are not implemented. Uninstalling preserves the user-data folder.

## Radio control and Hamlib

Audio carries the modem signal. Optional CAT control provides frequency/mode control and PTT separately; PTT can also use DTR/RTS, or the radio can use VOX.

Hamlib is **not bundled**. Install `rigctld`/`rigctl` separately to use a local or existing TCP Hamlib connection. Capabilities depend on the radio, backend and interface. See [radio setup and connection guidance](docs/INSTALL.md#licenses-and-radio-connection-security).

## Beta scope and limitations

- Live reception runs the selected mode near the selected audio center. It does not receive all three modes simultaneously or search the full passband.
- Switching modes resets receive state; the mode selector is locked during transmission.
- TNG5/TNG1 have restricted character support. Missing or CRC-failed portions can produce partial text.
- WAV export currently uses TNG44. Older WAV compatibility is limited; see [protocol scope and limits](docs/PROTOCOL.md#implementation-references-and-limits).
- The packaged runtime uses CPU PyTorch and does not require a GPU.
- This Beta source does not establish completed RF validation, universal radio compatibility, guaranteed weak-signal performance or superiority over other digital modes.

## License

TNG PKT source code and the bundled `stage3.pt` and `stage3_strong.pt` model weights are licensed under the [MIT License](LICENSE), copyright (c) 2026 6L5TNG.

Third-party libraries and native runtimes retain their own licenses. See [Third-party notices](THIRD_PARTY_NOTICES.md) and [full license texts](THIRD_PARTY_LICENSES.txt), including the corresponding-source and library-replacement requirements for binary distribution.

The project owner confirms ownership and permission to publish and redistribute the original icon2 artwork, as recorded in the third-party notices.
