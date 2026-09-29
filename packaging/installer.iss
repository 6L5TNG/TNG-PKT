#ifndef AppVersion
  #error AppVersion required: read it from registry.py
#endif
#ifndef DistDir
  #define DistDir "dist"
#endif
[Setup]
AppId={{EEDB8167-3444-407C-B5C9-7C5E5B25C1A4}
AppName=TNG PKT
AppVersion={#AppVersion}
AppPublisher=6L5TNG
DefaultDirName={autopf}\TNG PKT
DefaultGroupName=TNG PKT
PrivilegesRequired=lowest
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
DisableDirPage=no
DisableProgramGroupPage=yes
OutputDir=output
OutputBaseFilename=TNG-PKT-Setup-{#AppVersion}
SetupIconFile=..\assets\tng-pkt.ico
UninstallDisplayIcon={app}\TNG PKT.exe
Compression=lzma2/fast
SolidCompression=yes
WizardStyle=modern
CloseApplications=yes
RestartApplications=no
[Tasks]
Name: desktopicon; Description: "Create a desktop shortcut"; Flags: unchecked
[Files]
Source: "{#DistDir}\TNG PKT\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs
[Icons]
Name: "{autoprograms}\TNG PKT"; Filename: "{app}\TNG PKT.exe"; WorkingDir: "{app}"; IconFilename: "{app}\TNG PKT.exe"; AppUserModelID: "6L5TNG.TNGPKT"
Name: "{autodesktop}\TNG PKT"; Filename: "{app}\TNG PKT.exe"; WorkingDir: "{app}"; IconFilename: "{app}\TNG PKT.exe"; AppUserModelID: "6L5TNG.TNGPKT"; Tasks: desktopicon
[Run]
Filename: "{app}\TNG PKT.exe"; Description: "Launch TNG PKT"; Flags: nowait postinstall skipifsilent
; Deliberately no AppData files or UninstallDelete rules: user settings survive.
