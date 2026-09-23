@echo off
setlocal enabledelayedexpansion
REM Daily brief generation for stock-judgment-bench.
REM
REM Generation is fully resumable and cached, so a killed process costs only
REM the brief in flight. This loops rather than trusting a single invocation:
REM the inner process has been observed dying with STATUS_CONTROL_C_EXIT when
REM launched unattended by Task Scheduler, and the cause was not diagnosable
REM (the TaskScheduler operational log is disabled on this machine).
REM
REM Exit codes from run_briefs:  3 = provider's daily quota spent (stop)
REM                              0 = every brief generated (stop)
REM                          other = died unexpectedly (retry)
REM
REM enabledelayedexpansion matters: inside a parenthesised block cmd expands
REM %VAR% once at parse time, so an earlier version stamped every line with
REM the loop's start time and printed the wrong exit reason. !VAR! defers it.
cd /d "%~dp0.."
set "PY=%~dp0..\venv\Scripts\python.exe"
set "LOG=logs\daily.log"

echo.>> "%LOG%"
echo ===== !DATE! !TIME! START =====>> "%LOG%"

set "REASON=retry ceiling reached"
for /L %%i in (1,1,40) do (
    "%PY%" -m scripts.run_briefs --retry-failed >> "%LOG%" 2>&1
    set "RC=!errorlevel!"
    if "!RC!"=="3" (
        set "REASON=daily quota spent"
        goto :done
    )
    if "!RC!"=="0" (
        set "REASON=all briefs generated"
        goto :done
    )
    echo   attempt %%i exited !RC!, retrying>> "%LOG%"
    timeout /t 10 /nobreak >nul
)
:done
echo ===== !DATE! !TIME! STOP - !REASON! =====>> "%LOG%"
endlocal
