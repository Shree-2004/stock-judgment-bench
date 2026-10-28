# Registers the M1 poller with Windows Task Scheduler: every 15 minutes,
# windowless (pythonw), on battery too, one instance at a time.
#
#   powershell -ExecutionPolicy Bypass -File scripts\register_live_task.ps1
#
# Remove with:  Unregister-ScheduledTask -TaskName sjb-live-ingest -Confirm:$false
#
# Every 15 min because the NSE feed is a rolling window that held only ~6 hours
# on a results-season evening; a slower poll risks coverage gaps, which the
# store records but cannot recover.
$repo = Split-Path -Parent $PSScriptRoot
$pyw  = Join-Path $repo "venv\Scripts\pythonw.exe"
if (-not (Test-Path $pyw)) { throw "pythonw not found at $pyw - create the venv first" }

$action   = New-ScheduledTaskAction -Execute $pyw -Argument "-m scripts.live_ingest" -WorkingDirectory $repo
$trigger  = New-ScheduledTaskTrigger -Once -At (Get-Date).AddMinutes(1) -RepetitionInterval (New-TimeSpan -Minutes 15)
$settings = New-ScheduledTaskSettingsSet -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries `
            -StartWhenAvailable -ExecutionTimeLimit (New-TimeSpan -Minutes 5) -MultipleInstances IgnoreNew

Register-ScheduledTask -TaskName "sjb-live-ingest" -Action $action -Trigger $trigger -Settings $settings `
    -Description "stock-judgment-bench: poll NSE announcements RSS into store/live.sqlite" -Force |
    Select-Object TaskName, State
"Next run: " + (Get-ScheduledTaskInfo -TaskName "sjb-live-ingest").NextRunTime
