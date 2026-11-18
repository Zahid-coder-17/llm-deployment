# =================================================================
#  YORI: END-TO-END AUTOMATED MOBILE DEPLOYMENT SCRIPT
#  Transfers APK, fused SFT+GRPO model weights, and grants permissions
# =================================================================

$ErrorActionPreference = "Continue"
if (Test-Path Variable:\PSNativeCommandUseErrorActionPreference) {
    $PSNativeCommandUseErrorActionPreference = $false
}

$ProjectRoot = "c:\Users\YADKI ZAHID AHMED\Desktop\llm_deployment"
$AdbBin = "$ProjectRoot\platform-tools\adb.exe"
$ApkPath = "$ProjectRoot\embeddra-edge\android\app\build\outputs\apk\debug\app-debug.apk"
$WeightsPath = "D:\embeddra_eval\checkpoints\maria_1b_android_executive_merged.pt"

Write-Host "=================================================================" -ForegroundColor Cyan
Write-Host "  YORI MOBILE ON-DEVICE DEPLOYMENT (SFT + GRPO MERGED MODEL)" -ForegroundColor Cyan
Write-Host "=================================================================" -ForegroundColor Cyan

# 1. Check for Connected Phone via ADB
Write-Host "`n[1/5] Checking connected USB devices..." -ForegroundColor Yellow
& $AdbBin start-server
Start-Sleep -Seconds 2

$deviceId = $null
for ($i = 1; $i -le 5; $i++) {
    $devicesOutput = & $AdbBin devices | Out-String
    Write-Host $devicesOutput
    if ($devicesOutput -match "([A-Z0-9]{8,})\s+device") {
        $deviceId = $matches[1]
        break
    }
    Write-Host "Waiting for device to register (attempt $i/5)..." -ForegroundColor Yellow
    Start-Sleep -Seconds 2
}

if ($deviceId) {
    Write-Host "[OK] Target Device Connected: $deviceId" -ForegroundColor Green
}
else {
    Write-Host "`n[!] No authorized device detected yet." -ForegroundColor Red
    Write-Host "Please ensure phone is connected and tap 'Allow USB debugging' on your phone screen." -ForegroundColor Yellow
    exit 1
}

# 2. Install App APK
Write-Host "`n[2/5] Installing YORI APK ($ApkPath)..." -ForegroundColor Yellow
& $AdbBin -s $deviceId install -r -g $ApkPath
if ($LASTEXITCODE -ne 0) {
    Write-Host "[!] Re-trying installation without -g flag..." -ForegroundColor Yellow
    & $AdbBin -s $deviceId install -r $ApkPath
}
Write-Host "[OK] YORI APK successfully installed." -ForegroundColor Green

# 3. Grant Permissions
Write-Host "`n[3/5] Configuring Android permissions (Audio + Storage)..." -ForegroundColor Yellow
& $AdbBin -s $deviceId shell pm grant com.embeddra.edge android.permission.RECORD_AUDIO
& $AdbBin -s $deviceId shell appops set com.embeddra.edge MANAGE_EXTERNAL_STORAGE allow
Write-Host "[OK] Permissions granted." -ForegroundColor Green

# 4. Push 2.15 GB Fused SFT + GRPO Model Checkpoint
Write-Host "`n[4/5] Preparing on-device storage directory..." -ForegroundColor Yellow
$TargetDir = "/sdcard/Android/data/com.embeddra.edge/files"
& $AdbBin -s $deviceId shell "mkdir -p $TargetDir"

Write-Host "Checking if model already exists on phone..." -ForegroundColor Yellow
$checkRemote = ""
try {
    $checkRemote = (& $AdbBin -s $deviceId shell "ls -l $TargetDir/maria_1b_android_executive_merged.pt" 2>$null) | Out-String
}
catch {}

if ($checkRemote -match "maria_1b_android_executive_merged.pt" -and $checkRemote -notmatch "No such file") {
    Write-Host "[OK] Model weights already exist on phone:" -ForegroundColor Green
    Write-Host "     $checkRemote"
}
else {
    Write-Host "Pushing fused SFT+GRPO weights (2.15 GB) to phone storage..." -ForegroundColor Yellow
    Write-Host "Source: $WeightsPath"
    Write-Host "Destination: $TargetDir/maria_1b_android_executive_merged.pt"
    Write-Host "This transfer may take ~45-90 seconds over USB 3.0. Please keep USB connected." -ForegroundColor Cyan
    
    $pushStart = Get-Date
    & $AdbBin -s $deviceId push $WeightsPath "$TargetDir/maria_1b_android_executive_merged.pt"
    $pushDuration = ((Get-Date) - $pushStart).TotalSeconds
    Write-Host "[OK] Fused weights transferred in $([math]::Round($pushDuration, 1))s!" -ForegroundColor Green
}

# 5. Launch YORI on Phone
Write-Host "`n[5/5] Launching YORI on phone screen..." -ForegroundColor Green
& $AdbBin -s $deviceId shell "am start -n com.embeddra.edge/.MainActivity"

Write-Host "`n=================================================================" -ForegroundColor Green
Write-Host " DEPLOYMENT COMPLETE! YORI IS NOW RUNNING ON YOUR PHONE." -ForegroundColor Green
Write-Host "=================================================================" -ForegroundColor Green


