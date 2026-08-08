# Quét và dọn ổ C: — chỉ đụng vào cache/temp có thể tái tạo được.
# KHÔNG đụng: tài liệu cá nhân, Docker image/volume đang dùng, registry, System Restore.
# Chạy trong PowerShell (nên Run as Administrator để dọn được cả Windows Temp).
#
#   powershell -ExecutionPolicy Bypass -File F:\Thyroid\PriCheXy-Net\cleanup_c_drive.ps1          # chỉ quét
#   powershell -ExecutionPolicy Bypass -File F:\Thyroid\PriCheXy-Net\cleanup_c_drive.ps1 -Clean   # quét rồi xoá

param([switch]$Clean)

$ErrorActionPreference = 'SilentlyContinue'

function Get-DirSize($path) {
    if (-not (Test-Path $path)) { return $null }
    $b = (Get-ChildItem $path -Recurse -Force -File | Measure-Object Length -Sum).Sum
    if ($null -eq $b) { return 0 }
    return $b
}
function Fmt($bytes) {
    if ($null -eq $bytes) { return '     n/a' }
    return '{0,8:N2} GB' -f ($bytes / 1GB)
}
function Free-Space { (Get-PSDrive C).Free }

# --- An toàn để xoá sạch nội dung bên trong ---
$targets = [ordered]@{
    'Claude Code temp'      = "$env:LOCALAPPDATA\Temp\claude"
    'User Temp'             = $env:TEMP
    'Windows Temp'          = "$env:WINDIR\Temp"
    'Windows Update cache'  = "$env:WINDIR\SoftwareDistribution\Download"
    'pip cache'             = "$env:LOCALAPPDATA\pip\Cache"
    'npm cache'             = "$env:APPDATA\npm-cache"
    'Chrome cache'          = "$env:LOCALAPPDATA\Google\Chrome\User Data\Default\Cache"
    'Edge cache'            = "$env:LOCALAPPDATA\Microsoft\Edge\User Data\Default\Cache"
    'CrashDumps'            = "$env:LOCALAPPDATA\CrashDumps"
    'Delivery Optimization' = "$env:WINDIR\SoftwareDistribution\DeliveryOptimization"
}

# --- Chỉ BÁO CÁO, không tự xoá (cần bạn tự quyết) ---
$reportOnly = [ordered]@{
    'Docker WSL data (vhdx)' = "$env:LOCALAPPDATA\Docker\wsl"
    'Docker Desktop AppData' = "$env:APPDATA\Docker"
    'WSL / Store packages'   = "$env:LOCALAPPDATA\Packages"
    'NuGet packages'         = "$env:USERPROFILE\.nuget\packages"
    'Conda pkgs'             = "$env:USERPROFILE\.conda\pkgs"
    'Torch hub cache'        = "$env:USERPROFILE\.cache\torch"
    'HuggingFace cache'      = "$env:USERPROFILE\.cache\huggingface"
}

Write-Host ''
Write-Host '=== TRUOC KHI DON ===' -ForegroundColor Cyan
$freeBefore = Free-Space
Write-Host ("C: con trong: {0}" -f (Fmt $freeBefore))

Write-Host ''
Write-Host '--- Se don (cache/temp, tai tao duoc) ---' -ForegroundColor Yellow
foreach ($k in $targets.Keys) {
    Write-Host ('{0,-24} {1}   {2}' -f $k, (Fmt (Get-DirSize $targets[$k])), $targets[$k])
}

Write-Host ''
Write-Host '--- Chi bao cao (KHONG tu xoa) ---' -ForegroundColor Yellow
foreach ($k in $reportOnly.Keys) {
    Write-Host ('{0,-24} {1}   {2}' -f $k, (Fmt (Get-DirSize $reportOnly[$k])), $reportOnly[$k])
}

Write-Host ''
Write-Host '--- Top 15 thu muc lon nhat trong profile ---' -ForegroundColor Yellow
Get-ChildItem $env:USERPROFILE -Directory -Force |
    ForEach-Object { [PSCustomObject]@{ Name = $_.Name; GB = [math]::Round((Get-DirSize $_.FullName) / 1GB, 2) } } |
    Sort-Object GB -Descending | Select-Object -First 15 | Format-Table -AutoSize

Write-Host '--- Docker disk usage ---' -ForegroundColor Yellow
docker system df 2>$null

if (-not $Clean) {
    Write-Host ''
    Write-Host 'Che do QUET (chua xoa gi). Chay lai voi -Clean de thuc su don.' -ForegroundColor Green
    exit 0
}

Write-Host ''
Write-Host '=== DANG DON ===' -ForegroundColor Cyan
foreach ($k in $targets.Keys) {
    $p = $targets[$k]
    if (-not (Test-Path $p)) { continue }
    Write-Host ("  don {0} ..." -f $k)
    Get-ChildItem $p -Force | Remove-Item -Recurse -Force -ErrorAction SilentlyContinue
}

Write-Host '  don Recycle Bin ...'
Clear-RecycleBin -Force -ErrorAction SilentlyContinue

Write-Host '  docker builder prune (build cache) ...'
docker builder prune -f 2>$null
Write-Host '  docker image prune (chi image dangling) ...'
docker image prune -f 2>$null

Write-Host ''
Write-Host '=== SAU KHI DON ===' -ForegroundColor Cyan
$freeAfter = Free-Space
Write-Host ("C: con trong: {0}" -f (Fmt $freeAfter))
Write-Host ("Giai phong duoc: {0}" -f (Fmt ($freeAfter - $freeBefore))) -ForegroundColor Green
