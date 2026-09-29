# Third-party notices

This records the 0.11.2 Beta / Build 1009 Windows distribution.
Full collected license and copyright texts are in
[THIRD_PARTY_LICENSES.txt](THIRD_PARTY_LICENSES.txt). TNG PKT code and its two bundled model weights are licensed under the MIT
[LICENSE](LICENSE). Third-party components retain the terms recorded here.

## Direct components

| Component | Inspected version | Use / license and distribution conditions |
|---|---|---|
| Python | 3.13.2 | Embedded interpreter; PSF license stack. Retain its complete license text. |
| PySide6 / Shiboken6 / Qt | 6.8.1 | GUI and bindings; LGPLv3 option for used components. Retain notices and LGPL/GPL texts; allow replacement and debugging of modifications; provide corresponding library source. |
| PyTorch | 2.11.0+cpu | CPU model inference; BSD-3-Clause plus bundled component terms in LICENSE/NOTICE. |
| NumPy | 2.4.6 | Numeric arrays; composite wheel terms, including BSD-3-Clause. Preserve the complete wheel licenses. |
| SciPy | 1.15.3 | Signal processing; BSD-3-Clause plus wheel notices (including OpenBLAS). |
| pyqtgraph | 0.14.0 | Plots; MIT. Retain copyright and permission notice. |
| sounddevice | 0.5.5 | Audio I/O; MIT. PortAudio DLLs have separate terms. |
| soundfile | 0.13.1 | Declared source dependency; not included or used in this Windows bundle. BSD-3-Clause. libsndfile wheel COPYING is LGPL-2.1; retain notices, allow library replacement, and supply corresponding source if bundled. |
| psutil | 7.2.2 | Diagnostics; BSD-3-Clause. |
| pyserial | 3.5 | DTR/RTS PTT; BSD terms, see upstream license linked below. |
| Matplotlib | 3.11.2 | Colormap fallback; Matplotlib license agreement and included font/component notices. |
| PyInstaller | 6.20.0 | Build tool and executable bootloader; GPL-2.0-or-later with the bootloader exception. The exception does not license the application or waive dependency terms. |
| Inno Setup | 6.7.3 | Installer compiler/runtime; Inno Setup License. Preserve existing binary copyright and website notices. |
| Hamlib | Not bundled; user-installed version | External rigctld/rigctl executables only. Library is LGPL; utilities have GPL terms. No Hamlib redistribution is performed by this source/spec. Reassess if bundling it later. |

Permissive components generally permit binary redistribution with the required
notices. LGPL components additionally require the source/replacement obligations
in their actual license texts. This table is not a blanket release clearance.
Unpinned NumPy/Matplotlib requirements can resolve to other versions on a new
build; recheck the installed metadata and notices before publishing that build.
Other runtime notices are retained in the full text file where inspected.

## Qt packaging and replacement

Application imports use QtCore, QtGui and QtWidgets. Plotting hooks can also
collect QtNetwork, QtOpenGL, QtOpenGLWidgets, QtSvg, QtTest and QtPrintSupport.
These modules come from Qt base / Qt SVG and have an LGPLv3 option.
The QtGui packaging hook removes unused Virtual Keyboard and PDF image plugins
before dependency analysis. The spec permits only this audited Qt module set
and rejects unexpected Qt DLLs, including unused GPL-only addons.
The optional Mesa software OpenGL DLL is omitted: this application uses raster
Widgets/GraphicsView plots and does not request a software OpenGL context.

Qt license/copyright and original module attribution texts are preserved in
THIRD_PARTY_LICENSES.txt. The final bundle inventory is checked during packaging.

The onedir layout keeps Qt/PySide/Shiboken shared libraries separately under
`_internal/PySide6` and `_internal/shiboken6`. Users may replace the libraries
with ABI-compatible modified builds. The MIT project license imposes no ban on
reverse engineering/debugging of those modifications. Source execution and the
build spec are provided; no library signing/lockdown is added.
Exact Qt 6.8.1 and PySide 6.8.1 upstream source archives, including build scripts,
are staged outside the GitHub source in `Build_Tools/third_party_sources`.
For a binary release, upload these source archives and their manifest alongside
the installer on the same release page, without charge. No third-party source
patches are made by TNG PKT. The installed Qt reports a shared release build
using MSVC 2022. Distribution of source must accompany distribution of binaries;
staging sources locally alone does not complete a public source offer.
See [LGPLv3 requirements](https://www.gnu.org/licenses/lgpl-3.0.html) and
[Qt's LGPL guidance](https://www.qt.io/development/open-source-lgpl-obligations).

## Native runtime inventory

| File(s) / source | License / required notices / status |
|---|---|
| libportaudio64bit.dll / sounddevice wheel, PortAudio upstream | MIT; complete PortAudio copyright and permission text retained. Current installed version reports 19.7.0-devel. The custom hook includes only this non-ASIO DLL; MME, DirectSound, WDM/KS and WASAPI remain available. |
| libportaudio64bit-asio.dll / same wheel | Unused automatic collection excluded. No ASIO SDK redistribution claim is made. `SD_ENABLE_ASIO` is not a supported option for this installer. |
| libcrypto-3.dll, libssl-3.dll / Python runtime | OpenSSL 3.0.15, Apache-2.0; complete license retained, copyright (c) 1998-2024 The OpenSSL Authors. |
| libcrypto-3-x64.dll, libssl-3-x64.dll / system-resolved Qt backend dependency | Existing bundle version 3.6.4, Apache-2.0; inspect actual collection on any final build and retain copyright (c) 1998-2026 The OpenSSL Authors. |
| sqlite3.dll / Python runtime | SQLite 3.45.3, public domain; official declaration retained. |
| libscipy_openblas*.dll / NumPy and SciPy wheels | OpenBLAS BSD-3-Clause and bundled LAPACK/component terms; complete wheel license texts retained. |
| libiomp5md.dll, libiompstubs5md.dll / CPU PyTorch wheel | Intel OpenMP build 20250910. Intel Developer Tools EULA and third-party notices are retained in full below; these DLLs are not covered by the project MIT license. |
| uv.dll / CPU PyTorch wheel | libuv MIT; complete notice included in PyTorch's bundled LICENSE. |
| libffi-8.dll / Python runtime | libffi MIT; source copyright and permission text required. |
| python3.dll, python313.dll / Python 3.13.2 | PSF license stack; complete Python LICENSE retained. |
| Qt6*.dll, pyside6.abi3.dll, shiboken6.abi3.dll, Qt plugins / PySide6 wheels | LGPLv3 with original module/component notices and source/replacement measures above. |
| opengl32sw.dll / PySide6 wheel | Unused software OpenGL fallback excluded; no redistribution attempted. |
| protoc.exe / torch hook | Unused protobuf development compiler excluded; inference uses the runtime library. |
| msvcp140*.dll, vcruntime140*.dll, ucrtbase.dll, api-ms-win-*.dll / wheel and Microsoft runtime sources | Microsoft license terms, not MIT/BSD. Required runtime DLLs and wheel-specific renamed variants cannot be removed indiscriminately. Retained as dependencies of the packaged Python/Qt/numeric wheels. Preserve the applicable Microsoft and wheel notices. |

## Project model weights and artwork

| File | Registry / SHA-256 | Evidence and rights status |
|---|---|---|
| models/stage3.pt | TNG44 NN 0.1.0 / e73e59a5e2d7dade6907140a6243eb87793b48dcea6570be543a5cf32533d30c | Matches the local development and bundled model. Stage-3 checkpoint configuration and completed 60,000-step log agree with local training code. The code initializes StreamModem and generates random bits through a synthetic channel; no resume message occurs in that log. Evidence supports local training from initialization. |
| models/stage3_strong.pt | TNG5 NN 0.1.0 / 2cbd875d00c3408459d147a6dac28d8a09713c0592b2253b81e1115e59981c4c | Matches local and bundled copies. Checkpoint note and completed 30,000-step log identify initialization from models/stage3.pt; random bits and synthetic fading are used. This is fine-tuning of the project's stage3 model. |

No third-party pretrained weights or restricted training dataset were identified
in the examined code, checkpoint metadata or matching logs. The owner confirms the two models are project-owned and authorizes their
publication and redistribution under the project MIT LICENSE. TNG1 has no neural weight
file; its Huffman charset is project code. Experimental .pt/.onnx files are not
included in this public tree.

The owner confirms icon2.png is their original artwork and that they hold its
publication and redistribution rights. The existing derived icon is retained.

## Hamlib connection security

Application-managed rigctld uses `-T 127.0.0.1` and the application connects to
loopback in run mode. Connect mode preserves the configured remote host.
For a remote rigctld, use a trusted network or an authenticated tunnel/VPN and
restrict access with a firewall. Avoid exposing CAT/PTT control to the Internet;
TNG PKT's current TCP client does not add authentication or encryption.
[rigctld options](https://hamlib.sourceforge.net/html/rigctld.1.html).

Additional upstream license references:
[pyserial](https://github.com/pyserial/pyserial/blob/v3.5/LICENSE.txt),
[Hamlib](https://github.com/Hamlib/Hamlib),
[PortAudio](https://github.com/PortAudio/portaudio/blob/v19.7.0/LICENSE.txt).


OpenMP audit: both libiomp5md.dll and libiompstubs5md.dll match every PE
section of the official intel-openmp 2025.3.0 Windows wheel. Whole-file hashes
differ because certificate table sizes differ. The applicable package license
is Intel End User License Agreement for Developer Tools (August 2024), included
verbatim in THIRD_PARTY_LICENSES.txt with its third-party notices.
These runtime-specific terms remain applicable to the Intel DLLs and are
provided to end users in full; the project MIT license does not replace them.
