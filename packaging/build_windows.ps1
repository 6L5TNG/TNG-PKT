param(
    [string]$Python = "python",
    [Parameter(Mandatory=$true)][string]$Iscc,
    [string]$Output = "$PSScriptRoot\..\output",
    [string]$WorkRoot = $PSScriptRoot
)
$ErrorActionPreference = 'Stop'
$taskSource = (Resolve-Path -LiteralPath "$PSScriptRoot\..").Path
$taskOutput = [IO.Path]::GetFullPath($Output)
$taskIscc = (Resolve-Path -LiteralPath $Iscc).Path
$taskWork = [IO.Path]::GetFullPath($WorkRoot)
[IO.Directory]::CreateDirectory($taskWork) | Out-Null
$taskEnvPython = "$taskWork\build_env\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $taskEnvPython)) {
    & $Python -m venv "$taskWork\build_env"
    if ($LASTEXITCODE) { throw 'venv failed' }
}
& $taskEnvPython -X utf8 -m pip install -r "$taskSource\requirements.txt" -r "$PSScriptRoot\build-requirements.txt"
if ($LASTEXITCODE) { throw 'Dependency installation failed' }
$taskPreviousSource = $env:TNG_PKT_SOURCE
$taskPreviousBuildRoot = $env:TNG_PKT_BUILD_ROOT
$env:TNG_PKT_BUILD_ROOT = $taskWork
$env:TNG_PKT_SOURCE = $taskSource
Push-Location $taskWork
try {
    & $taskEnvPython -X utf8 -m PyInstaller --noconfirm --distpath dist --workpath build "$PSScriptRoot\TNG-PKT.spec"
    if ($LASTEXITCODE) { throw 'PyInstaller failed' }
    $taskVersion = & $taskEnvPython -X utf8 -c 'import ast,os,pathlib; t=ast.parse((pathlib.Path(os.environ["TNG_PKT_SOURCE"])/"registry.py").read_text(encoding="utf-8")); print(next(n.value.value for n in t.body if isinstance(n,ast.Assign) and any(isinstance(x,ast.Name) and x.id=="APP_VERSION" for x in n.targets)))'
    if ($LASTEXITCODE) { throw 'Version read failed' }
    [IO.Directory]::CreateDirectory($taskOutput) | Out-Null
    & $taskIscc "/DAppVersion=$taskVersion" "/FTNG-PKT-Setup-$taskVersion" "/O$taskOutput" "/DDistDir=$taskWork\dist" "$PSScriptRoot\installer.iss"
    if ($LASTEXITCODE) { throw 'Inno Setup failed' }
} finally {
    Pop-Location
    if ($null -eq $taskPreviousBuildRoot) { Remove-Item Env:\TNG_PKT_BUILD_ROOT -ErrorAction SilentlyContinue }
    else { $env:TNG_PKT_BUILD_ROOT = $taskPreviousBuildRoot }
    if ($null -eq $taskPreviousSource) { Remove-Item Env:\TNG_PKT_SOURCE -ErrorAction SilentlyContinue }
    else { $env:TNG_PKT_SOURCE = $taskPreviousSource }
}
