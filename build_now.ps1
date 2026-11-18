$ErrorActionPreference = "Stop"
$root = "c:\Users\YADKI ZAHID AHMED\Desktop\llm_deployment"
$env:JAVA_HOME = "$root\jdk\jdk-17.0.10+7"
$env:Path = "$root\jdk\jdk-17.0.10+7\bin;$env:Path"
$env:ANDROID_HOME = "$root\android-sdk"
$env:ANDROID_SDK_ROOT = "$root\android-sdk"
& "$root\embeddra-edge\android\tools\gradle-8.7\bin\gradle.bat" -p "$root\embeddra-edge\android" assembleDebug --no-daemon
