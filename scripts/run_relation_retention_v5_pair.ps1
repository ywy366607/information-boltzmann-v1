param([string]$Work='F:/cgb_relation_retention_v5_2481_20261119')
$ErrorActionPreference='Stop'
if(Test-Path -LiteralPath "$Work/pipeline_status.json"){
 throw 'Existing experiment directory; preserve its state and choose another work directory'
}
try {
 & "$PSScriptRoot/run_relation_retention_pair.ps1" -Work $Work
 & "$PSScriptRoot/finish_relation_retention_pair.ps1" -Work $Work `
   -ReportStem 'cgb_relation_retention_v5_loop_2481_20261119' `
   -RestorePrefix 'cgb_relation_use_v5_restore'
} catch {
 New-Item -ItemType Directory -Path $Work -Force | Out-Null
 @{status='failed';error=$_.Exception.Message;pid=$PID} | ConvertTo-Json |
   Set-Content -LiteralPath "$Work/continuation_error.json"
 throw
}
