# Installation, source execution and Windows builds

## Windows installer

Run `TNG-PKT-Setup-0.11.2.exe`, then launch TNG PKT from the Start menu.
Uninstall the older Build 1005 version before installing this release so that
obsolete Qt plugins are removed. User settings are preserved.
A desktop shortcut is optional. User settings are stored in
`%APPDATA%\TNG PKT` and are preserved when uninstalling.

## Run from source

The verified release configuration is Windows x64, Python 3.13 and CPU PyTorch.
Run these commands in PowerShell from the source root:

```powershell
python -m venv .venv
& .\.venv\Scripts\python.exe -m pip install -r requirements.txt
& .\.venv\Scripts\python.exe -m tngpkt
```

Keep `models/stage3.pt`, `models/stage3_strong.pt`, `assets/` and `style.qss`
alongside the application. The `diag.py` module is required by first-run setup
and the diagnostics dialog.

## Build the executable and installer

Install Python 3.13 x64 and Inno Setup 6. From the source root, run:

```powershell
& .\packaging\build_windows.ps1 -Python python -Iscc 'C:\Program Files (x86)\Inno Setup 6\ISCC.exe'
```

Set `-Iscc` to your Inno Setup compiler location. The build environment is created
in `packaging/build_env`, the executable in `packaging/dist/TNG PKT`, and the
installer in `output/TNG-PKT-Setup-0.11.2.exe`. Use `-Output` to select another
installer output directory.

The script builds this source directly, installs dependencies over the network,
checks for CPU PyTorch and reads the version from `tngpkt/registry.py`.
The PyInstaller spec includes both models, assets and `style.qss`.
The build script does not increment version or build numbers.

## Licenses and radio connection security

The build includes `LICENSE`, `THIRD_PARTY_NOTICES.md` and `THIRD_PARTY_LICENSES.txt` in
the application resource directory (`_internal` in the onedir build). Review
the notices and final DLL inventory before publishing an installer; installers predating
Build 1006 do not contain these packaging/security changes.
Qt/PySide shared libraries remain replaceable with compatible modified builds;
corresponding library source must be made available under their license terms.

When the application starts rigctld, it binds and connects to `127.0.0.1`.
Connecting to an existing remote rigctld remains supported. Use a trusted
network or a secure tunnel/VPN and firewall; this client does not authenticate
or encrypt CAT/PTT commands. Do not expose rigctld directly to the Internet.

Publish matching Qt/PySide source archives from the release source package
on the same GitHub Release as the binary.
The packaged audio backend uses the non-ASIO PortAudio DLL; MME, DirectSound,
WDM/KS and WASAPI are supported. Do not set `SD_ENABLE_ASIO` for this installer.

## Manual updates

Run the new installer over the existing installation, using the same installation
folder. It replaces application files and keeps `%APPDATA%\TNG PKT`, including
settings.json, layout.ini, logs and diagnostic results stored in settings.json.
Automatic GitHub update checking or installation is not implemented.
