param([Parameter(Mandatory=$true)][string]$Python)
$ErrorActionPreference = 'Stop'
$observerRoot = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..'))
$observerPython = [System.IO.Path]::GetFullPath($Python)
$observerTaskName = 'SelfLightGBM-S6-Prospective-ReadOnly-1745'
$observerProtocol = Join-Path $observerRoot 'reports\paper_futu\s6_forward_v1\protocol.json'
if (-not (Test-Path -LiteralPath $observerPython)) { throw 'Observer Python is missing' }
if (-not (Test-Path -LiteralPath $observerProtocol)) { throw 'Freeze and validate the forward protocol first' }
$observerFrozen = Get-Content -LiteralPath $observerProtocol -Raw | ConvertFrom-Json
$observerExisting = Get-ScheduledTask -TaskName $observerTaskName -ErrorAction SilentlyContinue
$observerArguments = '-m paper_futu.prospective_s6 --stage record'
if ($observerExisting -and ($observerExisting.Actions.Execute -ne $observerPython -or
    $observerExisting.Actions.Arguments -ne $observerArguments)) {
    throw 'A different task uses this name; refusing to replace it'
}
$observerUser = [System.Security.Principal.WindowsIdentity]::GetCurrent().Name
$observerAction = New-ScheduledTaskAction -Execute $observerPython -Argument $observerArguments -WorkingDirectory $observerRoot
$observerStart = ([datetime]::ParseExact($observerFrozen.first_execution_date, 'yyyy-MM-dd', $null)).AddHours(17).AddMinutes(45)
if ($observerStart -le [datetime]::Now) { $observerStart = [datetime]::Today.AddDays(1).AddHours(17).AddMinutes(45) }
$observerTrigger = New-ScheduledTaskTrigger -Daily -At $observerStart
$observerPrincipal = New-ScheduledTaskPrincipal -UserId $observerUser -LogonType Interactive -RunLevel Limited
$observerSettings = New-ScheduledTaskSettingsSet -StartWhenAvailable -MultipleInstances IgnoreNew -ExecutionTimeLimit (New-TimeSpan -Minutes 10) -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries
Register-ScheduledTask -TaskName $observerTaskName -Action $observerAction -Trigger $observerTrigger -Principal $observerPrincipal -Settings $observerSettings -Description 'Read-only S6 closing observation from existing paper receipts plus official CSI300 total-return index. Never connects to a broker or sends orders.' -Force | Select-Object TaskName,State
Get-ScheduledTaskInfo -TaskName $observerTaskName | Select-Object NextRunTime,LastRunTime,LastTaskResult
