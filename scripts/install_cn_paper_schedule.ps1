# Launch the authorized local CN simulated strategy without a visible terminal.
$ErrorActionPreference = 'Stop'
$paperRoot = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..'))
$paperPython = Join-Path $paperRoot 'paper_futu\.venv\Scripts\pythonw.exe'
$paperTaskName = 'SelfLightGBM-CN-Paper-0920'
if (-not (Test-Path -LiteralPath $paperPython)) { throw 'SDK pythonw.exe is missing' }
$paperExisting = Get-ScheduledTask -TaskName $paperTaskName -ErrorAction SilentlyContinue
if ($paperExisting) {
    if ($paperExisting.Actions.Execute -ne $paperPython -or $paperExisting.Actions.Arguments -ne '-m paper_futu.cn_worker') {
        throw 'A different task already uses this name; refusing to replace it'
    }
}
$paperUser = [System.Security.Principal.WindowsIdentity]::GetCurrent().Name
$paperAction = New-ScheduledTaskAction -Execute $paperPython -Argument '-m paper_futu.cn_worker' -WorkingDirectory $paperRoot
$paperStart = [datetime]::Today.AddHours(9).AddMinutes(20)
if ($paperStart -le [datetime]::Now) { $paperStart = $paperStart.AddDays(1) }
$paperTrigger = New-ScheduledTaskTrigger -Daily -At $paperStart
$paperPrincipal = New-ScheduledTaskPrincipal -UserId $paperUser -LogonType Interactive -RunLevel Limited
$paperSettings = New-ScheduledTaskSettingsSet -StartWhenAvailable -MultipleInstances IgnoreNew -ExecutionTimeLimit (New-TimeSpan -Hours 8) -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries
Register-ScheduledTask -TaskName $paperTaskName -Action $paperAction -Trigger $paperTrigger -Principal $paperPrincipal -Settings $paperSettings -Description 'CSI300 S6 quarterly QREG; Top20, head6 40%, max6 replacements, 5-session rebalance. CN SIMULATE only. Start 09:20, poll 30 seconds, exit after closing updates. Versioned transition in configs/futu_cn_paper.json.' -Force | Select-Object TaskName,State
Get-ScheduledTaskInfo -TaskName $paperTaskName | Select-Object NextRunTime,LastRunTime,LastTaskResult | Format-List
