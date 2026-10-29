# Registers the M3 prediction round with Windows Task Scheduler: weekdays at
# 18:30 IST, with a second pass at 21:30 that fills in anything the first pass
# lost to provider errors or the daily quota. A pass on a holiday, or one that
# wakes up the next morning, finds no session for "today" and exits without
# writing anything.
#
#   powershell -ExecutionPolicy Bypass -File scripts\register_ledger_task.ps1
#
# Remove with:  Unregister-ScheduledTask -TaskName sjb-ledger -Confirm:$false
#
# Each pass appends the ledger head to ledger/chain.txt and commits that file
# alone, and only on master. It never pushes. Push when you choose; the pushed
# history is the public witness.
$repo = Split-Path -Parent $PSScriptRoot
$pyw  = Join-Path $repo "venv\Scripts\pythonw.exe"
if (-not (Test-Path $pyw)) { throw "pythonw not found at $pyw - create the venv first" }

$days     = "Monday", "Tuesday", "Wednesday", "Thursday", "Friday"
$action   = New-ScheduledTaskAction -Execute $pyw -Argument "-m scripts.live_predict" -WorkingDirectory $repo
$triggers = @(
    (New-ScheduledTaskTrigger -Weekly -DaysOfWeek $days -At "18:30"),
    (New-ScheduledTaskTrigger -Weekly -DaysOfWeek $days -At "21:30")
)
$settings = New-ScheduledTaskSettingsSet -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries `
            -StartWhenAvailable -ExecutionTimeLimit (New-TimeSpan -Minutes 45) -MultipleInstances IgnoreNew

Register-ScheduledTask -TaskName "sjb-ledger" -Action $action -Trigger $triggers -Settings $settings `
    -Description "stock-judgment-bench: daily prediction round into the hash-chained ledger" -Force |
    Select-Object TaskName, State
"Next run: " + (Get-ScheduledTaskInfo -TaskName "sjb-ledger").NextRunTime
