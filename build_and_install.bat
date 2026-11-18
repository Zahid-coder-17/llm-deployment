@echo off
set "JAVA_HOME=C:\Users\YADKI ZAHID AHMED\Desktop\llm_deployment\jdk\jdk-17.0.10+7"
set "PATH=%JAVA_HOME%\bin;%PATH%"
cd /d "C:\Users\YADKI ZAHID AHMED\Desktop\llm_deployment\embeddra-edge\android"
call tools\gradle-8.7\bin\gradle.bat assembleDebug
if %ERRORLEVEL% NEQ 0 exit /b %ERRORLEVEL%
C:\Android\Sdk\platform-tools\adb.exe -s RZCX61AJBKX install -r app\build\outputs\apk\debug\app-debug.apk

