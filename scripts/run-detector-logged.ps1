param(
    [Parameter(Mandatory = $true)][string]$ProjectRoot,
    [Parameter(Mandatory = $true)][string]$LogPath
)

# Keep detector output in its own log while leaving the camera setup interactive.
$ErrorActionPreference = 'Continue'
$detectorScript = Join-Path $ProjectRoot 'scripts\phone-parking-detect.py'
Set-Location $ProjectRoot

"ParkBru detector started: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss zzz')" |
    Tee-Object -FilePath $LogPath -Append
"Command: py -3.13 `"$detectorScript`"" | Tee-Object -FilePath $LogPath -Append

& py -3.13 $detectorScript *>&1 | Tee-Object -FilePath $LogPath -Append
$detectorExitCode = $LASTEXITCODE
"Detector exited with code $detectorExitCode at $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss zzz')" |
    Tee-Object -FilePath $LogPath -Append

if ($detectorExitCode -ne 0) {
    Write-Host "Detector failed. Log: $LogPath" -ForegroundColor Red
    Read-Host 'Press Enter to close this window'
}
