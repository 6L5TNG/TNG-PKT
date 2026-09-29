[English](USER_GUIDE.md) | [한국어](USER_GUIDE.ko.md)

# TNG PKT User Guide

For **TNG PKT 0.11.2 Beta / Build 1010**. Button and panel names below follow the English interface. This guide describes the controls implemented in this release; radio and interface capabilities still depend on your equipment.

## Contents

1. [Introduction](#1-introduction)
2. [First Launch](#2-first-launch)
3. [Main Window Tour](#3-main-window-tour)
4. [Choosing a Mode](#4-choosing-a-mode)
5. [Understanding the Waterfall and Spectrum](#5-understanding-the-waterfall-and-spectrum)
6. [VFO and Radio Control](#6-vfo-and-radio-control)
7. [Receiving a Message](#7-receiving-a-message)
8. [Transmitting a Message](#8-transmitting-a-message)
9. [PTT and Radio Control](#9-ptt-and-radio-control)
10. [Audio Setup](#10-audio-setup)
11. [Layout and Workspace](#11-layout-and-workspace)
12. [QSO / Operating Area](#12-qso--operating-area)
13. [Analysis Tools](#13-analysis-tools)
14. [Zoom / View Controls](#14-zoom--view-controls)
15. [Settings](#15-settings)
16. [Diagnostics](#16-diagnostics)
17. [Logs and Saved Data](#17-logs-and-saved-data)
18. [Typical Operating Workflow](#18-typical-operating-workflow)
19. [Troubleshooting](#19-troubleshooting)
20. [Amateur-Radio Use](#20-amateur-radio-use)
21. [Further Documentation](#21-further-documentation)

## 1. Introduction

TNG PKT is an amateur-radio text communication program. It sends and decodes modem audio through an SSB radio and a computer audio interface. It offers TNG44, TNG5 and TNG1, with different balances of speed and error protection.

The audio connection carries the message. Optional CAT, or computer control of the radio, reads and changes the radio dial frequency. PTT, or push-to-talk control, switches the radio between receive and transmit. These are separate connections: working CAT does not by itself mean that audio reaches the radio.

## 2. First Launch

Install the Windows release and launch **TNG PKT** from the Start menu. A separate Python installation is unnecessary for the installer version. For installation details and the older Build 1005 migration instructions, see [Installation Documentation](INSTALL.md).

When there is no saved settings file, **Setup** opens before the main window. Use **Next** and **Back** to move through these steps; clicking a step name does not jump ahead.

1. **Language** — choose English or Korean. The initial setup immediately uses that language, and the main window opens in it.
2. **Call Sign** — enter your own callsign in **My Call**. It is stored in uppercase and supplies the `{MYCALL}` macro placeholder. Entering it does not automatically insert identification into every manually typed message.
3. **Audio** — choose **Audio API**, **Input** and **Output**. Input must carry the radio's receive audio into the PC; output must carry the PC's transmit audio to the radio. Watch **Input Level** for activity. **Test Tone** plays a short tone through the selected output; it does not issue a CAT/DTR/RTS PTT command. If your radio uses VOX, that audio can key it.
4. **Radio** — choose **Rig**, **Serial Port**, **Baud Rate**, **PTT Method** and, for serial PTT, **PTT Port**. Use settings that match the radio and interface. **Test CAT** should show `CAT OK` with frequency and mode. **Test PTT** briefly keys CAT/DTR/RTS and releases it automatically; with VOX it explains that audio keys the radio. You can **Skip** this step and configure the full Radio page later.
5. **Diagnostics** — click **Run Diagnostics** to check receive processing and the selected audio devices. Allow about 30 seconds, though the duration varies with the PC. You may **Skip** this step and run it later.
6. **Done** — click **Start** to save setup and open TNG PKT. Closing initial setup before finishing leaves it incomplete and it appears again next time.

For an audio-only station, skip Radio or use **Rig: None** with **VOX**. Set the dial and sideband on the radio itself. CAT is optional for text operation. Later, **Settings → General → Run Setup Again** reopens setup; **Run Diagnostics** repeats the check. Language changes made after the main window has opened take effect in that window after restarting TNG PKT.

## 3. Main Window Tour

The initial **Operate** layout puts the spectrum above the waterfall on the left and VFO, RX and TX on the right. Panels can move, so identify them by their titles if your layout differs. Start here; you do not need all the analysis graphs to make a contact.

| Area | What it tells you | What to do there |
|---|---|---|
| Top toolbar | Current layout and selected modem mode | Choose **Operate**, **Analysis** or **All**; select **TNG44**, **TNG5** or **TNG1**. Use **Panels** to open hidden panels. |
| **VFO** | Radio dial and band on the first row; audio **Offset**, calculated **RF** and **PTT** on the second | Read the radio frequency, tune through CAT when connected, or adjust Offset for a received signal. |
| **FFT Spectrum** | Audio energy at each frequency at the present time | Compare the RX trace with noise and the generated TX trace. Adjust the dBFS range if needed. |
| **Waterfall** | Audio frequencies over time, with synchronization and segment overlays | Locate the signal and click its center in the live waterfall to set Offset. |
| **RX** | Readable received text and current receive status | Read confirmed text, watch segment progress, copy text, or stop the current reception. |
| **TX** | A multiline message editor, progress and remaining time | Type a message and press **TX**; use **Stop** during transmission. |
| **Macros** | Callsign shortcuts, a separate single-line message field and TX Level | Enter **DX Call**, insert a macro, or send from this panel. It is a separate editor from TX. |
| **RX Log** | RX results and TX events in a table | Select an RX row to inspect its packet, open a WAV, enable Live RX, or export the session as CSV. |
| **Meters** and **Channel Quality / Stats** | Input level, estimates and packet/session measurements | Use these when diagnosing audio or comparing receptions. Open them with Panels. |
| Analysis panels | Detailed data for a selected packet or live receiver | Use the Analysis layout or individual panels after selecting an RX Log row. |
| Bottom status bar | Readiness, actions/errors, mode information, audio status and clocks | Read an error here first. Hover truncated information to see the full text. |

Toolbar **Log Folder** opens the current log directory. **Settings** changes station configuration. **About** shows versions and model information. **Help** currently displays `Coming soon`; use this guide and the panel **?** help instead.

Panel title controls appear when you hover the title: **?** gives a short explanation, **⧉** detaches/attaches the panel, and **✕** hides it. A packet panel can keep the last result while new audio arrives; its update age helps distinguish an old result from current reception.

## 4. Choosing a Mode

Both stations must use the **same mode**. The toolbar selection controls both live RX and TX; this release does not receive all three modes simultaneously. Changing mode resets current receive state, and mode changes are locked while transmitting.

| Mode | When to choose it | Practical consequence |
|---|---|---|
| **TNG44** | Ordinary text contacts when the channel is reasonably good; the faster option of the three | Supports UTF-8 text, including Hangul. Try it first when both stations can receive reliably. |
| **TNG5** | Weak signals or fading where stronger protection is useful | Gives up text speed for additional error protection. Allow more time for text to become confirmed. |
| **TNG1** | Difficult conditions where recovery is the priority | The slowest mode, receiving in blocks of about 14.7 seconds, with repeat combining available. Keep initial messages short. |

FEC means forward error correction: extra transmitted information helps the receiver repair errors. It improves the opportunity for recovery but cannot guarantee a message through interference or missing audio. Agree on a mode change with the other station; TNG PKT does not automatically select a more robust mode for you.

TNG5 and TNG1 use a shared 57-character set: uppercase letters, digits, spaces, line breaks and selected punctuation. Lowercase becomes uppercase and tabs become spaces. Unsupported characters, including Hangul, cause an **Invalid chars** warning. Edit the text, or press TX again with exactly the same text to send with those characters removed. TNG44 is the appropriate choice when you need unrestricted UTF-8 text. Mode names are not current measured character rates; detailed specifications belong in [Protocol Documentation](PROTOCOL.md).

## 5. Understanding the Waterfall and Spectrum

### Read the display

The waterfall is a history of the audio arriving from your receiver. Its **horizontal axis is audio frequency, from 0 to 3000 Hz**, not the radio's MHz dial. Its **vertical axis is time**. In live view the newest audio is at the top and older audio moves downward. In WAV file view the top is the beginning of the file and time advances downward.

Brighter colors indicate more energy relative to the display scale. Noise often fills a broad area; a continuing signal leaves a trace at its audio frequencies. A visible or bright trace alone does not establish that it is a decodable TNG PKT transmission.

The FFT Spectrum shows the present distribution of energy over the same audio frequencies, rather than the history. RX is the input spectrum, TX is the generated transmit audio spectrum, and **Peak Hold** retains recent peaks. Its vertical scale is **dBFS**, decibels relative to full-scale digital audio. It is an audio measurement, not RF power measured at the antenna. **Min**, **Max** and **Auto** change the displayed level range; Auto fits once when clicked.

### Set the audio center

The VFO **Offset** field is the modem's audio center frequency. The default is **1500 Hz**. It sets where TNG PKT places TX audio and where live RX looks for the selected mode. A solid center guide and band guides help you compare the signal position with that setting. The guides are visual references, not a guarantee that everything between them will decode.

To tune a received signal:

1. Choose the sender's mode and make sure receive audio is present.
2. Locate the middle of the signal's occupied audio band.
3. Left-click that frequency in the **live Waterfall**, or type a value in **Offset**. The setting rounds to 10 Hz steps.
4. Let a fresh transmission or block arrive and check for CRC-confirmed text.

Click tuning is active only in live waterfall view. It changes audio Offset, not the radio dial. It also resets current reception; avoid clicking during a message you want to finish. Offset is locked during TX. Its allowed range is limited for each mode so the nominal signal band remains within 300–2700 Hz.

### Understand the search range

Seeing a signal anywhere in the 0–3000 Hz display does not mean the receiver is searching there. The live receiver searches near the selected Offset: nominally **±100 Hz for TNG44/TNG5** and **±60 Hz for TNG1**. These are frequency-error search limits, not the occupied bandwidth of the signal. Local refinement/tracking also occurs; do not aim for the limit.

If a signal is well away from the selected center, click its center before the next transmission. If the signal lies at the edge of the radio's audio passband, adjust the radio dial/filter so the entire signal can pass, then set Offset again. Increasing graph zoom cannot recover audio filtered out by the radio.

**Live Waterfall** controls the moving display; **Live RX** in RX Log controls decoding separately. **Segments** shows/hides overlays. **Speed** changes how much history fits on screen, and **Palette** changes colors. These view settings do not select a different modem mode or improve its error protection.

## 6. VFO and Radio Control

VFO means variable frequency oscillator; here the panel provides radio dial control and the relationship between radio frequency and modem audio.

**First row:** the large dial display, the band shortcut selector, and connection status. With CAT connected, the dial follows the radio. Wheel over a digit to change that place value, or click to type a frequency. For example, `14.078` means MHz and `14078000` means Hz. Enter applies the edit; Escape cancels it. A band shortcut sends that saved dial frequency to the radio.

**Second row:** **Offset** in audio Hz, **RF** for the calculated signal center on air, and the momentary **PTT** button. With a USB dial of 14.078000 MHz and Offset 1500 Hz, the RF center is 14.079500 MHz. For LSB/data-L it is dial minus Offset. Check the actual sideband and filter on your radio; there is no main-window radio-mode selector.

When CAT is unavailable, the panel says **Not connected**, the dial/RF show placeholders, and band/dial controls are disabled. Offset still works. Use the radio's own dial display when operating without CAT; the app does not present an unverified manual frequency as measured RF, and RF fields in logs are blank.

Hamlib supplies radio-specific control, and **rigctld** is its background control service. TNG PKT can start a local rigctld or connect to one already running. Hamlib is not bundled with the Windows release. Install it separately if using CAT; details are in [Installation Documentation](INSTALL.md#licenses-and-radio-connection-security).

## 7. Receiving a Message

1. Match the sender's mode, radio sideband and signal position. Enable **Live RX** in RX Log; use **Panels → RX Log** if that panel is hidden.
2. Confirm that the input device carries radio audio. A moving waterfall or Input Level is useful evidence of input, but not proof of decoding.
3. Leave Offset steady while the signal arrives. The receiver looks for synchronization, a known signal pattern used to locate the message in time and frequency. TNG1 uses synchronization within its blocks and may take longer to confirm.
4. Watch the waterfall overlays. **Magenta dashed markers/boxes are tentative, before the first successful CRC.** They indicate a candidate, not successfully received text. A candidate that never confirms may disappear.
5. Wait for CRC validation. CRC is an error-detection check on a received segment or block. When the first part passes, the RX panel opens the message and the tentative overlays change to their normal state colors. **CRC-passed segments are solid green; failed segments are red; waiting parts remain dashed.** Start/end synchronization boxes are separate guides and do not certify every text segment.
6. Read the RX text and the segment count together. Text is added as checked parts become available. **▍** at the end means waiting for the next part; **□** marks a failed/missing part. A readable passage with a gap is a partial message.
7. At the end, inspect the final status and RX Log result. A **Partial** or failure result means you should ask for the missing text again rather than assume the entire message was received.

The first CRC validates that part, not the complete transmission. There may be subsequent good and bad segments. Keep receiving until the message completes or is lost. RX **Stop** requests decoding of the received portion and returns the receiver to waiting; it is different from unchecking Live RX, which disables live decoding.

For TNG1, **Resend** at the transmitting station repeats the previous message from its beginning. The receiver can combine matching repeated blocks and sometimes recover a missing portion later. This is not an automatic request/acknowledgment system; coordinate repeats with the other operator and still check CRC results.

## 8. Transmitting a Message

Before transmitting, make sure the frequency is available and the radio/audio route is the one you intend to use.

1. Select the agreed **TNG44**, **TNG5** or **TNG1** mode.
2. Check the radio dial, sideband and audio Offset. If CAT is connected, compare the VFO dial/RF with the radio. Without CAT, check the radio directly.
3. Verify **Settings → Radio → PTT Method** and **Settings → Audio → Output**. Set **TX Level** for your interface and radio; this controls audio amplitude, not a watt setting. Monitor the radio's own output/ALC and avoid overdriving its input.
4. Type into the multiline **TX** editor, including your callsign where appropriate. Press **TX**. Read any unsupported-character warning before pressing again.
5. The editor becomes read-only during TX. Watch the progress bar, **Remaining**, and the active/sent segment highlighting. Gray text has been sent and red highlights show the part being sent; this is local progress, not confirmation from the other station.
6. On completion TNG PKT stops audio, requests PTT release for CAT/DTR/RTS, and restores the TX controls. With VOX, the radio releases according to its own VOX delay when audio stops. Ordinary live RX is paused during TX and resumes afterward.
7. Wait for the other station's reply to determine whether your message was received.

**Stop** once schedules a clean end after the required current segment/block portion. The TX panel changes the button to **Halt**. Press it again to stop immediately. Near the end, the app may say **Near end, cannot stop** and finish the remaining tail. An immediate halt can leave the receiver with an incomplete message.

TNG1 has **Resend**, enabled after a TNG1 transmission. It sends the last transmitted normalized text, even if you have subsequently edited the editor; use TX to send a new message.

The separate Macros panel can also send: enter text in its **Message** field and click **TX** or press Enter. Its **TX Now** option sends a clicked macro immediately. Leave that option unchecked when you want to review the text first.

**Max TX Time**, in Settings → Radio → TNG Options, limits continuous TX/manual PTT. The default is 300 seconds; exceeding it stops audio and requests PTT release. Long TNG1 messages can reach it. Choose a sensible limit for your equipment and shorten messages when necessary.

## 9. PTT and Radio Control

Configure **Settings → Radio** while idle. CAT controls the radio, and the **PTT Method** determines how it is keyed; DTR/RTS PTT can be used without frequency CAT.

### VOX

Use **VOX** when the radio or interface keys from audio. TNG PKT sends audio but does not assert a control line. Enable and adjust VOX on the radio/interface, select the correct output device, and verify that the beginning of a message is not cut off. **VOX Lead Tone** can add a tone before the modem audio to give VOX time to switch. Test PTT alone does not test a VOX audio path or actually key it with a tone.

### CAT

Use **CAT** when your radio's Hamlib backend supports PTT. Select the real **Rig**, CAT port and serial parameters, and first obtain a successful **Test CAT** result. Then test PTT. **Transmit Audio Source** chooses **Rear/Data** or **Front/Mic** for CAT PTT where supported; choose the connector carrying your audio. Frequency control success and PTT/audio-source support are separate checks.

### DTR

Use **DTR** if your interface keys PTT with the serial DTR control line. Select the PTT **Port**, or the CAT serial port when the interface is designed to share it. The cable/interface must translate that line into radio PTT; selecting DTR does not create an audio connection. Avoid letting another application own that port.

### RTS

Use **RTS** for an interface wired to the RTS line instead. Its setup is like DTR, but you must select the line your hardware actually uses. CAT **Force Control Lines** are serial-port configuration, not a substitute for selecting the intended PTT Method. Leave them at the interface's recommended values.

### Connect Hamlib and test the result

In **TNG Options**, **Start rigctld with TNG** starts a local service using your Rig/serial settings. Supply **rigctld Path** if it cannot be found automatically. **Connect to running rigctld** instead uses **rigctld Address** (host and TCP port; default `127.0.0.1:4532`). Use an existing service's actual model/connection configuration; changing a client setting does not reconfigure that service's serial connection. Use a trusted connection as described in the installation document.

**Poll Interval** controls how often CAT readings update. **Mode: None** leaves radio mode selection to you; **USB** or **Data/Pkt** requests the corresponding radio mode. **Split Operation: None** is the straightforward starting point. **Rig** uses radio split and **Fake It** temporarily moves the dial during TX, keeping generated audio near 1500 Hz; support depends on the radio/backend. Use split only after verifying that operation with your equipment.

**Tx delay** provides a silent interval between keyed PTT and modem audio for CAT/DTR/RTS. The VFO **PTT** button keys only while held and sends no modem audio. **Test PTT** on the Settings page releases automatically after about three seconds; the first-run test is about two seconds. These tests can key the transmitter. The app also requests release on TX end, halt, errors and exit; CAT loss during CAT-controlled TX halts transmission. Confirm release on the radio itself when testing a new interface.

## 10. Audio Setup

Think of the two audio selections from the PC's perspective:

| Selection | Required route | A common wrong choice |
|---|---|---|
| **Input** | Radio receive audio → computer | Laptop microphone: waterfall reacts to room sound, while the radio signal is absent. |
| **Output** | Computer modem audio → radio transmit input | PC speakers/headphones: local sound plays, but the radio has no modulation. |

A radio with USB audio may appear as a USB sound device. An external interface may have a similar generic name. Windows' default device may instead be your headset or onboard sound card; do not rely on its name alone. Follow the connection from the radio and verify both directions.

Choose **Audio API** first, then the matching Input and Output listed for that API. The packaged Windows audio backend supports MME, DirectSound, WDM/KS and WASAPI; ASIO is not provided. If one API cannot open the device, try another available API and reselect both devices.

Use **Settings → Audio**, then **Apply** to save and apply while keeping the dialog open, or **OK** to apply and close. **Cancel** discards ordinary unapplied dialog edits. Audio-device changes are locked during TX or an active message reception; finish/stop that operation before changing devices. The separate Audio dock, when opened through Panels, changes device selections directly and remembers them.

For a normal RX route, **Input Level** moves with radio audio and the live waterfall changes with signals/noise from the receiver. A level near silence suggests a missing route; a level reaching full scale suggests clipping. Change receive level on the radio/interface or in Windows, since the app's TX Level does not control RX gain. For TX, use setup Test Tone or a short message with the intended PTT method and check the radio's modulation/output indication. A successful device-open diagnostic alone does not verify the cable or radio menu routing.

Saved devices are matched by their names and API. If a saved USB device is unavailable, the app may select another/default device. Recheck after unplugging equipment or changing Windows audio configuration; read device names in the bottom status bar and its tooltip.

## 11. Layout and Workspace

A layout is the arrangement of the program's panels, not a different radio or modem configuration. You can hide a panel without disabling the function it represents: hiding RX does not turn off Live RX.

- **Operate** gives a large waterfall and readable RX/TX with VFO. Use it for a contact or a smaller screen.
- **Analysis** emphasizes packet graphs, RX Log and quality measurements. Some TX tools appear as tabs behind other panels; click the tab name to bring one forward.
- **All** combines operating and analysis tools across more columns. It suits a large display; several panels share tabbed areas. Packet Anatomy, Zoom Waterfall and the TNG1 tone grid are opened separately as needed.

Drag a panel's **title bar** to move it, or drag a boundary between panels to change their sizes. Hover a title and use **⧉** to float the panel as a separate window, or attach it again. **✕** hides it; recover it from **Panels**. Very small panels may show internal scrollbars. You are changing the view, not deleting messages or changing the modem.

**Detach Analysis** moves the constellation, eye, features, LLR, synchronization and trend group to a separate analysis window, useful on a second monitor. Click it again or close that analysis window to return the group. Other panels can be floated individually.

To recover an ordinary workspace, return the analysis group if detached and click **Operate**. Floating Packet Anatomy/Zoom/Tone Grid windows can remain visible when applying a preset; close them separately if desired. There is no separate Reset Layout button.

On normal exit, TNG PKT saves window geometry, the preset and detached-analysis information in `layout.ini`. **This build reapplies the selected preset at startup instead of restoring the complete saved main-window dock state.** Do not expect every custom panel position, tab or hidden state to return exactly. Audio and station settings are stored separately.

For routine contacts, start with Operate, then open RX Log if you want a record beside the conversation. For investigating a difficult reception, select Analysis and open Packet Anatomy for the relevant RX row. If an off-screen or damaged layout cannot be recovered through the toolbar, see Troubleshooting for a reversible layout-file reset.

## 12. QSO / Operating Area

A QSO is a radio contact. TNG PKT's operating area is a text exchange workspace with an event log; it does not expose an ADIF contact-entry form or automatic contact confirmation.

**RX** keeps messages in a scrolling view, separated by local date/time and mode. **Follow** returns to the newest text; scrolling up disables following, and scrolling back to the bottom resumes it. **A− / A / A+** decrease, reset or increase text size. **Copy** copies selected text; without a selection it copies the current message's decoded text/gap representation. **Clear** clears the view only, not saved session logs.

**TX** holds the message you compose. Its transmitted progress is local, and RX holds what the other station actually sent. The app does not mirror every outgoing message into the RX conversation view; consult RX Log for TX entries.

**Macros** has **DX Call**, entered by you for the other station. `{MYCALL}` uses Settings → General → My Call; `{DXCALL}` uses that field; `{SNR}` uses the held SNR value. Clicking a macro fills the Macros panel's single-line editor, not the multiline TX editor. With **TX Now** enabled it also transmits immediately. Right-click a macro to edit/delete it; use its add control or Settings → Macros to create one. The app does not infer DX Call reliably from arbitrary received text, so check it before replying.

**RX Log** contains UTC, Result, SNR, Δf, Seg, Drift, Resync, FEC, RF and Message/Reason. Seg shows passed/total segments where available; FEC is the corrected-bit count. RF is recorded when CAT information is available. TX and stop events also appear, but an outgoing row does not prove remote reception. Select an RX row with analysis data to update packet panels and open Packet Anatomy. Hover the message/reason for more detail.

## 13. Analysis Tools

Analysis is optional for ordinary contacts. Use it to answer a practical question such as “Was the start found?”, “Which part failed CRC?” or “Did repeated TNG1 blocks help?” Open **Analysis** or **Panels**, then select a relevant **RX Log** row. A log row without stored analysis may have nothing to show. Empty/unknown fields are not zero-error results.

### Packet Timeline and Channel Quality / Stats

**Packet Timeline** puts frames above segment CRC results. Green means OK, yellow marks FEC corrections, and red means failed/checkable errors. Hover a frame or segment for its details; click to highlight the related frame/segment in the analysis. A frame's inferred correctness is limited by which segment bits can be checked.

**Channel Quality / Stats** shows Hard BER, FEC Fixed / Checked, SNR, EVM after EQ, start-sync correlation and residual drift for the selected packet, plus current session counts. BER describes hard-decision bit errors where the decoded reference permits checking; FEC counts repairs rather than text errors remaining. SNR is an estimate of signal relative to noise. EVM describes deviation from a reconstructed signal after equalization, which compensates for channel distortion. Correlation measures agreement with the known synchronization pattern. Start with **CRC result and segment count**; none of these estimates independently proves correct text.

**Meters** show Input Level, SNR Est, Freq Error and Margin. Frequency error is relative to the selected audio center and confirmed synchronization. Margin subtracts a fixed reference of −6.5 dB from the SNR estimate; it is a display reference, not a universal decode threshold for every mode/channel. **Trend** plots recent packet measurements (up to 60 packets) versus UTC; compare changes over several receptions instead of judging one point.

### IQ Constellation, Eye Diagram and NN Features / Phase Trace

**IQ Constellation** plots signal components for received symbols. **EQ** uses the reconstructed channel correction; **Raw** has frequency/phase correction only. Comparing the same packet in the two views helps inspect distortion; a compact-looking plot alone is not a CRC check.

**Eye Diagram** overlays two-symbol traces. It offers another view of how the received waveform varies in time. The EQ/Raw selection is shared with the constellation. For routine operation you can leave these graphs closed; an experienced operator can compare the corrected and raw traces for one packet.

**NN Features / Phase Trace** has a two-dimensional projection of the neural receiver's features before its bit decision and a selected frame's I/Q trace. The projection is a visualization, not received text or a direct SNR reading. Choose a frame through Packet Timeline to inspect its trace. Colors/accuracy require a checkable reference; unknown portions do not supply one.

### LLR Distribution and Costas Sync Map

LLR is a soft bit-decision confidence value. **LLR Distribution** shows how these decisions are distributed; pink and blue represent known reference bits, while gray indicates unknown reference. Large magnitudes can show stronger decisions, but even a strong wrong decision is possible. Use CRC and checkable-bit information to interpret it.

**Costas Sync Map** shows the synchronization score over time and frequency error. Green contours indicate the detector threshold; a circle marks the peak and a plus marks the detection position. **Live** and **Snapshot** distinguish the current search from a saved detection view. Its palette and **Smooth** change presentation. A synchronization peak locates a candidate; it does not guarantee valid payload text. For TNG5, the packet view uses its own synchronization data when available.

### Packet Anatomy: TNG44 and TNG5

**Packet Anatomy** is the user-facing packet dissector. Select an RX Log row; the app opens the corresponding window when data exists. You do not need to run `dissect.py` or `dissect5.py` separately.

The layered view follows audio through frequency-corrected I/Q, frames/segments, soft decisions, reordered/corrected bits, CRC and bytes/text. **Click a layer** to highlight the matching position across layers and show details in the item tree. Red triangular markers identify error-correction changes. Use **Prev / Next**, the segments-per-page control, or the overview strip to move through a long packet. Horizontal pan/zoom is available in this view; **To Start** returns to the first page/view.

TNG5's layers also show middle synchronization and repeat combining. If a TNG5 result only has the alternate summary data, a segment table is shown instead, with CRC16, characters, frames, mean absolute LLR, correction count, bytes and text. Begin by locating failed CRC segments and the corresponding text gap. Advanced users can trace those segments backward through the bit layers. A correction marker is evidence of a decoder change, not necessarily failure.

### TNG1 analysis

TNG1 uses different panels in the same analysis positions; selecting a TNG1 packet can also switch those views. It has no neural receiver model, so NN-specific views are not an equivalent measure of its decoding.

| Tool | What you can inspect | First thing to look at |
|---|---|---|
| **16-Tone Grid · TNG1** | Tone cells over time; RX above TX, synchronization cells and FEC-changed cells | White borders identify CRC-passed symbols; click a block boundary for its anatomy. |
| **Symbol Confidence · TNG1** | Ratio of the strongest to second-strongest tone cell | Ambiguous cells can explain uncertainty; do not treat the ratio as CRC success. |
| **Sync Accumulation · TNG1** | Synchronization evidence combined over blocks, versus block phase and frequency offset | Candidate position and accumulated blocks; a peak is still a candidate until CRC validates data. |
| **12 Sync Tones · TNG1** | Individual synchronization-tone scores | How the twelve contributions differ, rather than assuming one high tone proves a message. |
| **Packet Anatomy · TNG1** | Block selection, symbol decisions/corrections, decoded bits, CRC and text | The block's CRC result and recovered text. |
| **TX Anatomy · TNG1** | Text encoding, block bits and transmitted tone-symbol structure | Which block is currently being sent. |

A Tone Grid block with no corresponding decoded data may report **no anatomy data**; the live display alone does not create a CRC-passed packet.

### TX Monitor and TX Anatomy

**TX Monitor** shows local transmit stages, current segment, elapsed/remaining time and throughput. **Throughput Data** counts characters in completed segments since data began; **Throughput Total** includes the start overhead. Short-message rates can therefore differ from long-message data rates. Use frame navigation and **Follow TX** to follow the current position.

**TX Anatomy** follows bytes, information bits, coded bits and their transmit order. Dim content has been sent and the red position line shows local progress. TNG1 has the block-specific view described above. These displays explain your outgoing signal; they do not measure how the other station received it.

## 14. Zoom / View Controls

Open **Panels → Zoom Waterfall** for a closer audio-frequency view around the current Offset. It contains a spectrum above a waterfall, with selected-center/band/tone guides and a detected frequency-error line when available.

- **Span** chooses **±150**, **±300** or **±500 Hz** around Offset.
- **Resolution** offers **Time (4 Hz)**, **Balanced (2 Hz)** and **Frequency (1 Hz)**. Finer frequency resolution uses a longer time window; choose Time to inspect brief changes, or Frequency to separate nearby tones.
- **Speed** changes display history; **Segments** toggles overlays.
- **Min / Max / Auto** adjust display level range. Auto fits the recent input once when pressed.

For the normal zoom view, return to **±300 Hz** and **Balanced (2 Hz)**, or close the window and use the full Waterfall. Zoom settings are remembered. They affect the graph, not the receiver's search range or radio dial; use the full live Waterfall or VFO Offset to tune.

RX text has its own size controls, **A− / A / A+**, and **Ctrl+−**, **Ctrl++ / Ctrl+=**, **Ctrl+0** while the RX panel has focus. These change text readability only. Packet Anatomy has its own horizontal graph navigation and To Start control; do not confuse those with audio tuning.

## 15. Settings

Settings contains **General, Radio, Audio, Mode, RX / Display and Macros**. Normally use Apply or OK after editing. Cancel does not undo changes already applied. In particular, **Test CAT/Test PTT apply the Radio settings before testing**, and rerunning setup/diagnostics can save results independently of ordinary dialog edits.

### General

**My Call** supplies your own macro callsign. **Clock** chooses UTC, local time or both for the status bar; RX Log remains UTC and RX separators use local time. **Language** changes on restart. **Run Setup Again** is useful after changing station hardware, and **Run Diagnostics** checks current audio selections. **Log Folder → Browse** chooses a directory for subsequent log sessions; the new path applies after restarting. Existing logs are not moved automatically.

### Radio

Use Rig/serial settings for CAT and PTT Method/Port for keying. Most users can leave serial data/stop bits and handshake at the radio/interface defaults. Wrong baud/port/model settings prevent connection; the wrong control line can prevent PTT. Test the connection before a contact. TNG Options contains rigctld connection/path, Max TX Time and VOX Lead Tone. **Bands** lets you Add/Delete/edit shortcuts or restore **Defaults**; its **Dial (MHz)** values are your shortcuts, not an official list of TNG PKT calling frequencies. Radio controls are locked while transmitting or testing/holding PTT.

### Audio

Set Audio API, Input, Output and TX Level here. The live Input Level helps check the receive route. Changing an API changes the device list; recheck both choices. Device selections are locked during active RX/TX, so end the operation first.

### Mode

**Mode at Start** chooses **Last Mode** or **Always TNG44**. This is a startup preference; select the current contact's mode on the toolbar. It does not enable automatic switching or reception of every mode.

### RX / Display

**RX Font Size** adjusts readable received text, and **Palette** changes the waterfall colors. Neither changes receive sensitivity or CRC behavior.

### Macros

Edit names and text, **Add** a new entry, or **Delete** the selected entry. Use `{MYCALL}`, `{DXCALL}` and `{SNR}` placeholders. Check the expanded text and the chosen mode's character support before sending. Missing callsign fields produce an incomplete reply, so fill them in first.

## 16. Diagnostics

Run Diagnostics from setup or **Settings → General**. It opens the selected input/output devices and benchmarks receive processing for TNG44, TNG5 and TNG1. The results include processing margin on one CPU core, audio sample rate, buffer dropouts and estimated clock error when available. Processing margin compares the audio's duration with processing time; more margin means more time available to keep up.

The actual English UI uses these labels (the Korean UI uses 적합 / 주의 / 부적합):

| Result | Meaning | What to do |
|---|---|---|
| **Pass** (fit) | The measured check met its suitability criterion | Continue setup, then check real radio audio and a short contact. |
| **Warning** (caution) | Reduced CPU margin, buffer dropouts or audio rate/clock concerns | Read the reason; reduce other PC load or review the audio device/API, then run again. |
| **Fail** (unsuitable) | Processing margin below the criterion, excessive clock error, or a device/check error | Correct the reported issue before relying on live reception. |

CPU grading uses one-core margin: at least 3× is Pass, at least 1.5× is Warning, and lower is Fail. Audio may warn about dropouts or a sample rate different from the preferred 48 kHz. The reason matters: **Failed to open device** is a connection/configuration issue, not a measurement of propagation conditions. **No result** or a failed diagnostic run is not Pass.

Results are saved with settings, and can become outdated after device or PC changes. Diagnostics do not measure your RF path, remote reception, antenna, interference or complete station compatibility. A Pass is not a guarantee of protocol performance on air.

## 17. Logs and Saved Data

On Windows, user data normally resides in **`%APPDATA%\TNG PKT`**. Paste that path into File Explorer to open it, or use the toolbar Log Folder for the active log directory.

| Item | Purpose | How to use it |
|---|---|---|
| `settings.json` | Callsign, devices, radio preferences, macros and last diagnostics | Normally change it through Settings; back it up before troubleshooting/resetting. |
| `layout.ini` | Window geometry, preset and analysis-window layout information | Normally managed by the app; see the startup restoration limit in Layout and Workspace. |
| `logs/` | Default directory for saved session and diagnostic/timing records | Keep relevant files when reporting a problem. A custom Log Folder may replace this location. |
| `session_YYYYmmdd_HHMMSSZ.jsonl` | RX/TX events for an application session, one record per line | A saved operating history; the `Z` denotes UTC. It is not a full audio recording. |
| `latency_YYYYMMDD.csv` | Automatically written timing information | Primarily for troubleshooting; ordinary contacts do not require interpreting it. |

**Export CSV**, in RX Log, saves the **current session's** RX/TX records to a path you choose. It includes UTC, direction, result, available signal/error measurements, mode, dial/RF and text. It does not automatically collect all earlier log files. The file uses UTF-8 with a BOM for spreadsheet readability.

**Open WAV** loads an audio file, displays its waterfall and attempts supported mode decoding automatically; that file operation differs from live selected-mode RX. The display switches out of live waterfall view. Re-enable **Live Waterfall** afterward to return to current input. Older file-format compatibility is limited; see Protocol Documentation.

**Save WAV** in Macros creates a **TNG44 transmit WAV from that panel's single-line Message field**. It does not record RX audio, use the multiline TX editor, or export TNG5/TNG1 even if one of those modes is selected. It uses the TNG44 engine's standard waveform rather than the whole active radio/PTT/level setup. This release has no user-facing continuous RX WAV recorder.

Changing Settings → General → Log Folder applies on the next start, leaving existing files where they were. Choose a writable directory. You normally do not need to edit data files yourself, and uninstalling/updating preserves the user-data folder.

## 18. Typical Operating Workflow

This example uses a USB radio/interface and TNG44. Choose an appropriate agreed SSB frequency for your operation; the app's band shortcut values are not frequency recommendations.

1. Connect the radio's USB/serial and audio interface. Turn the radio on and select its intended USB/data-USB audio route.
2. In Settings → Audio select that interface's Input and Output. Check Input Level and the live waterfall for radio audio.
3. In Settings → Radio configure CAT if wanted, and test the intended PTT method. Use Rig None/VOX if operating without CAT.
4. Select **Operate** and **TNG44**. Agree with the other operator on that mode and the RF signal position.
5. Set the radio dial. Start with Offset **1500 Hz** and check VFO/RF when CAT is connected.
6. Enable Live RX. Watch the full waterfall; if the other station's signal is displaced, click its center before the next message.
7. Wait through tentative magenta dashed indicators until CRC-passed text appears. Read the full message and check for □ gaps/Partial.
8. Type a short reply in TX, for example `6L5TNG DE YOURCALL TNX UR MSG K`, replacing the callsigns with the actual station calls.
9. Press TX, watch Remaining and the radio's transmit indication, then confirm it returns to receive when audio ends.
10. Read the other station's reply. For repeated failures, first check the audio route/mode/tuning; if those are correct, agree on TNG5 or TNG1 before both changing mode. Use uppercase supported text for those modes.
11. Enter DX Call and use a reviewed macro if convenient. Leave TX Now off while learning the workflow.
12. Review RX Log after the contact and export CSV if needed. To study a failed reception, select its row and inspect CRC/segment data in Packet Anatomy.

## 19. Troubleshooting

Work from the audio route and mode toward more detailed analysis. Keep a copy of settings and relevant logs before a reset.

### 1. The program opens but nothing is received

Open RX Log and ensure **Live RX** is checked. Wait for model loading to finish. Check the selected mode, Input device and Offset. A running app or moving display does not prove the decoder is enabled.

### 2. The waterfall is empty or frozen

Ensure **Live Waterfall** is checked; Open WAV changes the display to file view. Check Input Level, the physical receive-audio route, Audio API and Input device. A freeze caused by disabling the display is different from silent audio. Read any device-open error in the status bar.

### 3. A signal is visible but does not decode

Match the sender's mode and compatible release/protocol. Center the signal near Offset, inside the search range, and pass the whole band through the radio filter. Check sideband, clipping and interference. Wait for a new start/block if you changed mode or center mid-message.

### 4. Only magenta dashed or pending indications appear

The receiver found a candidate but has not validated payload CRC. Allow time, especially for TNG1 and TNG5, then inspect signal position and audio level. A candidate may disappear without text if it never confirms. Do not interpret it as a completed message.

### 5. RX has □ gaps, ▍ remains, or the result is Partial

Some parts failed or the message is still arriving. Wait for the end; check the final segment count. Ask for the missing part again. In TNG1, ask the sender to use Resend for the same message so matching blocks can combine. Signal loss or an early TX halt can leave incomplete text.

### 6. The other station cannot receive me

Agree on mode and frequency; verify your radio really keys and produces modulated RF. Check Output device, radio audio-input menu, PTT Method and TX Level. Sending progress shows local playback only. Use a shorter supported-text message and confirm the other station's center position.

### 7. No TX audio, or audio comes from the PC speaker

Choose the actual radio/interface **Output** for the selected API. Read **No output device** or **Playback failed** errors. Test the route with setup Test Tone when appropriate. Device enumeration or a Pass does not mean the output cable is connected to the radio.

### 8. PTT does not key, or releases too late

For VOX, enable VOX on the radio/interface and verify audio; Test PTT alone sends no tone. For CAT, obtain CAT OK and verify backend PTT support/audio source. For DTR/RTS, check the correct PTT port and wiring, and close competing port users. Review Tx delay/VOX Lead Tone if the beginning is cut off; VOX release delay is on the radio.

### 9. CAT connection fails

Check Rig model, COM port, baud and serial settings. Install Hamlib if using local rigctld and set rigctld Path if needed. For an existing service, verify host/TCP port and that the service is running. Another program may own the CAT port. Test CAT applies the edited Radio settings before testing.

### 10. The radio frequency display seems wrong or is blank

Blank dial/RF with Not connected is expected without CAT. With CAT, compare the **dial** with the radio, then account for Offset in the **RF center**: USB adds it, LSB subtracts it. Check the radio sideband and Poll Interval. Split/Fake It may move the TX dial temporarily; begin with Split None if unsure.

### 11. A mode change or tuning click interrupts reception

That resets receive state in this release. Finish the message first, or wait for a fresh transmission/block after changing settings. A disabled selector or Offset during TX is intentional; stop/finish transmitting before changing it.

### 12. Text cannot be sent in TNG5/TNG1

Read Invalid chars and remove unsupported characters or use TNG44 for UTF-8/Hangul. Pressing TX again with the same text removes unsupported characters; review the result before doing so. A message reduced to no supported characters cannot be sent.

### 13. The USB interface changed or the wrong audio device is active

Reconnect it, reopen Settings → Audio, choose API first and then reselect both devices. Saved names can be missing and a default device selected instead. End current RX/TX before changing devices if controls are locked.

### 14. Model or resource errors appear

Read the status bar and **About → Copy Version Info**. Reinstall the matching official package if resources are missing/damaged. For source execution, retain `models/stage3.pt`, `models/stage3_strong.pt`, `assets/` and `style.qss` in the documented structure. Do not substitute unrelated models to silence a mismatch; they may be incompatible. TNG1 itself uses no neural model, but the application still initializes its other receiver resources.

### 15. Diagnostics reports Warning/Fail or cannot finish

Read which mode/device and reason failed. Close heavy background applications and retest CPU margin. For audio, recheck the device/API, availability and reported dropouts/rate/clock error. Run again after the configuration changes; old stored results are not a current test.

### 16. Layout is confusing or a panel is missing

Use Panels to reopen it, return Detach Analysis, and choose Operate. Close remaining floating special panels if needed. If geometry is off-screen, close the app, back up/rename `%APPDATA%\TNG PKT\layout.ini`, and restart to create a fresh default layout. This resets the view without deleting station settings/logs.

### 17. Settings are tangled or initial setup does not appear

Use Settings → General → Run Setup Again for the ordinary recovery path. If a full settings reset is necessary, close TNG PKT, back up/rename `settings.json` in the user-data folder, then restart and complete setup. Your station/audio/radio/macro choices must be entered again; keep logs and layout separately. Invalid JSON is automatically backed up as `settings.broken-...json` and defaults used. Do not edit an active settings file while the app is running.

### 18. TX stops before the whole message is sent

Check for Max TX Time, PTT failure or CAT loss in the status bar. Shorten the message or set an appropriate Max TX Time for the equipment. A scheduled Stop finishes only the retained part; an immediate Halt can cut a part. Verify the radio has returned to receive before retrying.

## 20. Amateur-Radio Use

TNG PKT is intended for open amateur-radio text communication. It has no secret-key message encryption. The protocol and on-air encoding are public; error correction and whitening are signal-processing functions, not secrecy. Include appropriate station identification in your messages and operate within your station's privileges. Technical encoding details are in [Protocol Documentation](PROTOCOL.md).

## 21. Further Documentation

- [Protocol Documentation](PROTOCOL.md) — mode specifications, framing, error correction, on-air encoding and compatibility limits.
- [Installation Documentation](INSTALL.md) — Windows installation/update notes, source execution and Hamlib connection guidance.
- [Project overview](../README.md) — release scope and general project information.

For a problem report, use **About → Copy Version Info** to collect application/build, protocol/model and library details. About is in English in both interfaces. Double-click a supported Model ID cell to copy the full model hash; attach relevant logs and describe the mode, audio devices, PTT method and exact observed error.
