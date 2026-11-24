param([string]$Work='F:/cgb_relation_retention_v5_2481_20261119')
$ErrorActionPreference='Stop'
$repo='D:/information-boltzmann-v1'
$python='D:/conda_envs/vox/python.exe'
$env:OMP_NUM_THREADS='1';$env:MKL_NUM_THREADS='1';$env:PYTHONUTF8='1'
Set-Location -LiteralPath $repo
try {
 @{status='waiting';stage='complete pair and restore';pid=$PID} | ConvertTo-Json |
   Set-Content -LiteralPath "$Work/actor_expectation_status.json"
 $deadline=(Get-Date).AddHours(2)
 while($true){
  if(Test-Path -LiteralPath "$Work/continuation_error.json"){
   $errorState=Get-Content -LiteralPath "$Work/continuation_error.json" -Raw | ConvertFrom-Json
   # An earlier failed shell launch predates the current live pipeline.
   if(-not(Test-Path -LiteralPath "$Work/pipeline_status.json")){throw $errorState.error}
   $pipeline=Get-Content -LiteralPath "$Work/pipeline_status.json" -Raw | ConvertFrom-Json
   if($errorState.pid -eq $pipeline.pid){throw $errorState.error}
  }
  if(Test-Path -LiteralPath "$Work/pipeline_status.json"){
   $pipeline=Get-Content -LiteralPath "$Work/pipeline_status.json" -Raw | ConvertFrom-Json
   if($pipeline.status -eq 'failed'){throw 'Pair failed; inspect its log before actor audit'}
  }
  if(Test-Path -LiteralPath "$Work/finalization_status.json"){
   $state=Get-Content -LiteralPath "$Work/finalization_status.json" -Raw | ConvertFrom-Json
   if($state.status -eq 'failed'){throw $state.error}
   if($state.status -eq 'completed'){break}
  }
  if((Get-Date) -gt $deadline){throw 'Pair finalization timeout; preserve its complete boundaries'}
  Start-Sleep -Seconds 10
 }
 @{status='running';stage='actual actor encoded binding';pid=$PID} | ConvertTo-Json |
   Set-Content -LiteralPath "$Work/actor_expectation_status.json"
 & $python -m scripts.audit_native_policy_boundary --checkpoint "$Work/apply/body/last.pt.gz" `
   --work "$repo/scratch/actor_reference_v5_2489" `
   --output results/published/cgb_actual_actor_reference_v5_2489_20261119.json *> "$Work/actor_reference.log"
 if($LASTEXITCODE -ne 0){throw 'Actual body actor reference failed'}
 & $python -m scripts.summarize_native_actor_expectation --work $Work `
   --actor-report results/published/cgb_actual_actor_reference_v5_2489_20261119.json `
   --output results/published/cgb_actor_expectation_v5_20261119.json *> "$Work/actor_expectation_summary.log"
 if($LASTEXITCODE -ne 0){throw 'Complete-trajectory actor expectation summary failed'}
 @{status='completed';source_cycle=2489;updates=0;world_calls=0} | ConvertTo-Json |
   Set-Content -LiteralPath "$Work/actor_expectation_status.json"
} catch {
 @{status='failed';error=$_.Exception.Message} | ConvertTo-Json |
   Set-Content -LiteralPath "$Work/actor_expectation_status.json"
 throw
}
