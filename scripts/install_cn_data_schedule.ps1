param([Parameter(Mandatory=$true)][string]$Python)
$ErrorActionPreference = 'Stop'
$dataRoot = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..'))
$dataPython = [System.IO.Path]::GetFullPath($Python)
if (-not (Test-Path -LiteralPath $dataPython)) { throw 'Data Python is missing' }
$dataTaskName = 'SelfLightGBM-CN-PublicData-Close'
$dataArguments = '-X utf8 -m paper_futu.update_cn_data --catch-up'
$dataExisting = Get-ScheduledTask -TaskName $dataTaskName -ErrorAction SilentlyContinue
if ($dataExisting -and ($dataExisting.Actions.Execute -ne $dataPython -or $dataExisting.Actions.Arguments -notin @($dataArguments,'-X utf8 -m paper_futu.update_cn_data'))) {
    throw 'Different task already uses the data updater name'
}
$dataUser = [System.Security.Principal.WindowsIdentity]::GetCurrent().Name
$dataAction = New-ScheduledTaskAction -Execute $dataPython -Argument $dataArguments -WorkingDirectory $dataRoot
$dataTriggers = @('08:50','15:40','16:10','16:40','17:10','18:10') | ForEach-Object {
    New-ScheduledTaskTrigger -Daily -At ([datetime]::ParseExact($_, 'HH:mm', $null))
}
$dataPrincipal = New-ScheduledTaskPrincipal -UserId $dataUser -LogonType Interactive -RunLevel Limited
$dataSettings = New-ScheduledTaskSettingsSet -StartWhenAvailable -MultipleInstances IgnoreNew -ExecutionTimeLimit (New-TimeSpan -Minutes 40) -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries
Register-ScheduledTask -TaskName $dataTaskName -Action $dataAction -Trigger $dataTriggers -Principal $dataPrincipal -Settings $dataSettings -Description 'Public daily market update and signal inference; no OpenD, broker SDK or order submission. Locked and completed-session-only.' -Force | Select-Object TaskName,State
Get-ScheduledTaskInfo -TaskName $dataTaskName | Select-Object NextRunTime,LastRunTime,LastTaskResult

# Logon recovery imports no broker SDK in its data process and NEVER runs a trade executor.
$recoveryName='SelfLightGBM-CN-Login-Recovery'
$recoveryArguments='-X utf8 -m paper_futu.recover_cn_session'
$oldRecovery=Get-ScheduledTask -TaskName $recoveryName -ErrorAction SilentlyContinue
if ($oldRecovery -and ($oldRecovery.Actions.Execute -ne $dataPython -or $oldRecovery.Actions.Arguments -ne $recoveryArguments)) { throw 'Different logon recovery task already exists' }
$recoveryAction=New-ScheduledTaskAction -Execute $dataPython -Argument $recoveryArguments -WorkingDirectory $dataRoot
$logon=New-ScheduledTaskTrigger -AtLogOn -User $dataUser
$logon.Delay='PT30S'
$recoverySettings=New-ScheduledTaskSettingsSet -StartWhenAvailable -MultipleInstances IgnoreNew -ExecutionTimeLimit (New-TimeSpan -Hours 2) -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries
Register-ScheduledTask -TaskName $recoveryName -Action $recoveryAction -Trigger $logon -Principal $dataPrincipal -Settings $recoverySettings -Description 'On Windows logon, catch up completed public daily data and separately query paper account if OpenD is logged in. NO order execution.' -Force | Select-Object TaskName,State

# If OpenD login happens AFTER Windows logon, retry only account observation every ten minutes.
$accountName='SelfLightGBM-CN-Account-ReadOnly'
$sdkPython=Join-Path $dataRoot 'paper_futu\.venv\Scripts\pythonw.exe'
$accountArguments='-X utf8 -m paper_futu.record_cn_account'
$oldAccount=Get-ScheduledTask -TaskName $accountName -ErrorAction SilentlyContinue
if ($oldAccount -and ($oldAccount.Actions.Execute -ne $sdkPython -or $oldAccount.Actions.Arguments -ne $accountArguments)) { throw 'Different account recovery task already exists' }
$accountAction=New-ScheduledTaskAction -Execute $sdkPython -Argument $accountArguments -WorkingDirectory $dataRoot
$accountTrigger=New-ScheduledTaskTrigger -Daily -At ([datetime]::Today.AddHours(9).AddMinutes(20))
$repeatTemplate=New-ScheduledTaskTrigger -Once -At ([datetime]::Today.AddHours(9).AddMinutes(20)) -RepetitionInterval (New-TimeSpan -Minutes 10) -RepetitionDuration (New-TimeSpan -Hours 13 -Minutes 40)
$accountTrigger.Repetition=$repeatTemplate.Repetition
$accountSettings=New-ScheduledTaskSettingsSet -StartWhenAvailable -MultipleInstances IgnoreNew -ExecutionTimeLimit (New-TimeSpan -Minutes 3) -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries
Register-ScheduledTask -TaskName $accountName -Action $accountAction -Trigger $accountTrigger -Principal $dataPrincipal -Settings $accountSettings -Description 'Read-only simulated account observation every ten minutes 09:20-23:00; offline/login waits are non-trading states. Never sends or cancels orders.' -Force | Select-Object TaskName,State
