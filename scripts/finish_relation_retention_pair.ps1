param([string]$Work='F:/cgb_relation_retention_2481_20261119',
 [string]$ReportStem='cgb_relation_retention_loop_2481_20261119',
 [string]$RestorePrefix='cgb_relation_use_restore')
$ErrorActionPreference='Stop'
Set-Location -LiteralPath 'D:/information-boltzmann-v1'
$python='D:/conda_envs/vox/python.exe'
$env:OMP_NUM_THREADS='1'; $env:MKL_NUM_THREADS='1'; $env:PYTHONUTF8='1'
while($true){
 $state=Get-Content -LiteralPath "$Work/pipeline_status.json" -Raw | ConvertFrom-Json
 if($state.status -eq 'completed'){break}
 if($state.status -eq 'failed'){throw 'Training pair failed; preserve complete boundaries and logs'}
 Start-Sleep -Seconds 10
}
try {
 $report="results/published/$ReportStem.json"
 & $python -m scripts.summarize_native_relation_retention --work $Work --output $report *> "$Work/summary.log"
 if($LASTEXITCODE -ne 0){throw 'Pair summary failed'}
 foreach($arm in @('observe','apply')){
  & $python -m scripts.audit_native_relation_use_restore --work "$Work/$arm" --output "results/published/${RestorePrefix}_${arm}_2489_20261119.json" *> "$Work/${arm}_restore.log"
  if($LASTEXITCODE -ne 0){throw "Restore audit failed: $arm"}
 }
 & $python -m scripts.finish_native_relation_retention --report $report --restore-prefix $RestorePrefix *> "$Work/publication.log"
 if($LASTEXITCODE -ne 0){throw 'Route publication failed'}
 @{status='completed';training='complete';summary='passed';restore='passed';docs='updated'} |
  ConvertTo-Json | Set-Content -LiteralPath "$Work/finalization_status.json"
} catch {
 @{status='failed';error=$_.Exception.Message} | ConvertTo-Json | Set-Content -LiteralPath "$Work/finalization_status.json"
 throw
}
