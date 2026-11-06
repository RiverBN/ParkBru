param(
    [string]$Address
)

$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $projectRoot
$runStamp = Get-Date -Format 'yyyyMMdd-HHmmss'
$logDir = Join-Path $projectRoot 'logs'
New-Item -ItemType Directory -Force -Path $logDir | Out-Null
$setupLog = Join-Path $logDir "showcase-$runStamp-setup.log"
$buildLog = Join-Path $logDir "showcase-$runStamp-build.log"
$previewLog = Join-Path $logDir "showcase-$runStamp-preview.log"
$detectorLog = Join-Path $logDir "showcase-$runStamp-detector.log"

"ParkBru HTTPS showcase run: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss zzz')" |
    Tee-Object -FilePath $setupLog

function Stop-WithMessage([string]$Message) {
    "ERROR: $Message (at $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss zzz'))" |
        Add-Content -Path $setupLog
    Write-Host $Message -ForegroundColor Red
    Write-Host "Setup log: $setupLog" -ForegroundColor Yellow
    if (Test-Path $buildLog) { Write-Host "Build log: $buildLog" -ForegroundColor Yellow }
    exit 1
}

if (-not (Get-Command mkcert -ErrorAction SilentlyContinue)) {
    Stop-WithMessage 'mkcert was not found. Install it first (for example: winget install FiloSottile.mkcert), then reopen PowerShell.'
}
if (-not (Test-Path (Join-Path $projectRoot 'node_modules'))) {
    Stop-WithMessage 'node_modules is missing. Run npm install in the ParkBru folder first.'
}
$pythonLauncher = Get-Command py.exe -ErrorAction SilentlyContinue
if (-not $pythonLauncher) {
    Stop-WithMessage 'The Python launcher (py.exe) was not found. Install Python 3.13 and its Windows launcher first.'
}

if (-not $Address) {
    $candidates = @(
        Get-NetIPAddress -AddressFamily IPv4 -ErrorAction SilentlyContinue |
            Where-Object { $_.IPAddress -notlike '127.*' -and $_.IPAddress -notlike '169.254.*' } |
            Sort-Object InterfaceMetric, SkipAsSource, InterfaceIndex
    )
    if ($candidates.Count -eq 0) {
        Stop-WithMessage 'No usable IPv4 address found. Connect the laptop to the phone hotspot and try again.'
    }

    Write-Host 'Choose the laptop IPv4 address your phone can reach (usually the hotspot adapter):'
    for ($i = 0; $i -lt $candidates.Count; $i++) {
        $adapter = Get-NetAdapter -InterfaceIndex $candidates[$i].InterfaceIndex -ErrorAction SilentlyContinue
        $name = if ($adapter) { $adapter.Name } else { "Interface $($candidates[$i].InterfaceIndex)" }
        Write-Host ("  {0}. {1}  ({2})" -f ($i + 1), $candidates[$i].IPAddress, $name)
    }
    $selection = Read-Host 'Selection number'
    $index = 0
    if (-not [int]::TryParse($selection, [ref]$index) -or $index -lt 1 -or $index -gt $candidates.Count) {
        Stop-WithMessage 'Invalid selection.'
    }
    $Address = $candidates[$index - 1].IPAddress
}

if ($Address -notmatch '^\d{1,3}(\.\d{1,3}){3}$') {
    Stop-WithMessage "'$Address' is not an IPv4 address."
}

$certDir = Join-Path $projectRoot 'certs'
New-Item -ItemType Directory -Force -Path $certDir | Out-Null
$cert = Join-Path $certDir 'parkbru.pem'
$key = Join-Path $certDir 'parkbru-key.pem'

Write-Host "Preparing locally trusted HTTPS certificate for $Address..." -ForegroundColor Cyan
& mkcert -install 2>&1 | Tee-Object -FilePath $setupLog -Append
if ($LASTEXITCODE -ne 0) { Stop-WithMessage 'mkcert could not install its local root certificate.' }
& mkcert -cert-file $cert -key-file $key localhost 127.0.0.1 ::1 $Address 2>&1 |
    Tee-Object -FilePath $setupLog -Append
if ($LASTEXITCODE -ne 0) { Stop-WithMessage 'Certificate generation failed.' }

Write-Host 'Building ParkBru...' -ForegroundColor Cyan
& npm run build 2>&1 | Tee-Object -FilePath $buildLog
if ($LASTEXITCODE -ne 0) { Stop-WithMessage 'Build failed; HTTPS preview was not started.' }

$detectorScript = Join-Path $projectRoot 'scripts\phone-parking-detect.py'
if (-not (Test-Path $detectorScript)) {
    Stop-WithMessage "Detector script not found: $detectorScript"
}

Write-Host 'Starting the parking detector in a separate window...' -ForegroundColor Cyan
$detectorLogger = Join-Path $projectRoot 'scripts\run-detector-logged.ps1'
$powershellExe = (Get-Process -Id $PID).Path
Start-Process -FilePath $powershellExe `
    -ArgumentList @(
        '-NoExit', '-ExecutionPolicy', 'Bypass', '-File', "`"$detectorLogger`"",
        '-ProjectRoot', "`"$projectRoot`"", '-LogPath', "`"$detectorLog`""
    ) `
    -WorkingDirectory $projectRoot

Write-Host ''
Write-Host "ParkBru is starting. Open https://${Address}:4173 on your phone." -ForegroundColor Green
Write-Host 'Keep this window open for the web app. Close the detector window when finished.'
Write-Host "Logs are being saved under: $logDir" -ForegroundColor Cyan
"Starting HTTPS preview on https://${Address}:4173 at $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss zzz')" |
    Tee-Object -FilePath $previewLog
& npm run preview:https -- --port 4173 2>&1 | Tee-Object -FilePath $previewLog -Append
$previewExitCode = $LASTEXITCODE
"HTTPS preview exited with code $previewExitCode at $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss zzz')" |
    Tee-Object -FilePath $previewLog -Append
exit $previewExitCode
