param([string]$Work='D:/information-boltzmann-v1/scratch/cgb_momentum_value_2489_20261119')
$ErrorActionPreference='Stop'
$repo='D:/information-boltzmann-v1'
$python='D:/conda_envs/vox/python.exe'
$sourceBody='F:/cgb_relation_retention_v5_2481_20261119/apply/body/last.pt.gz'
$env:OMP_NUM_THREADS='1';$env:MKL_NUM_THREADS='1';$env:PYTHONUTF8='1'
Set-Location -LiteralPath $repo
if(Test-Path -LiteralPath "$Work/pipeline_status.json"){throw 'Preserve existing work; choose a new directory'}
New-Item -ItemType Directory -Path $Work -Force | Out-Null
$files=@('scripts/ib_local/epistemic_value_reference.py','scripts/ib_local/causal_relation_program.py',
 'scripts/ib_local/causal_use_value.py','scripts/ib_local/relation_retention_use.py',
 'scripts/ib_local/generative_relations.py','scripts/ib_local/finite_use_value.py',
 'scripts/ib_local/run_mother_continuation.py','scripts/run_native_preview_restore_comparison.py')
$hashes=@{};foreach($f in $files){$hashes[$f]=(Get-FileHash -LiteralPath $f -Algorithm SHA256).Hash}
$hashes|ConvertTo-Json|Set-Content -LiteralPath "$Work/source_hashes.json"
try {
 @{status='running';stage='zero-update admission';pid=$PID}|ConvertTo-Json|
   Set-Content -LiteralPath "$Work/pipeline_status.json"
 & $python -m scripts.audit_native_policy_boundary --checkpoint $sourceBody --work "$Work/admission" `
   --momentum-repair --output results/published/cgb_momentum_value_admission_2489_20261119.json *> "$Work/admission.log"
 if($LASTEXITCODE -ne 0){throw 'Momentum-value actual body admission failed'}
 $admission=Get-Content results/published/cgb_momentum_value_admission_2489_20261119.json -Raw|ConvertFrom-Json
 if(-not $admission.source_unchanged -or $admission.momentum_repair.old_actor_replay_error -gt 0.00003){throw 'Admission provenance/replay mismatch'}
 foreach($arm in @('control','carry')){
  foreach($f in $files){if((Get-FileHash -LiteralPath $f -Algorithm SHA256).Hash -ne $hashes[$f]){throw "Source changed: $f"}}
  @{status='running';stage='full loop';arm=$arm;pid=$PID}|ConvertTo-Json|
    Set-Content -LiteralPath "$Work/pipeline_status.json"
  $argsRun=@('-m','scripts.run_native_preview_restore_comparison','--checkpoint',$sourceBody,
     '--work',"$Work/$arm",'--cycles','4','--world-activations',"$Work/${arm}_activations")
  if($arm -eq 'carry'){$argsRun+='--epistemic-momentum'}else{$argsRun+='--epistemic-value'}
  & $python @argsRun *> "$Work/$arm.log"
  if($LASTEXITCODE -ne 0){throw "Full loop failed: $arm"}
  & $python -m scripts.audit_native_relation_use_restore --work "$Work/$arm" --output "$Work/${arm}_restore.json" *> "$Work/${arm}_restore.log"
  if($LASTEXITCODE -ne 0){throw "Restore failed: $arm"}
 }
 & $python -m scripts.summarize_native_momentum_value --work $Work `
   --output results/published/cgb_momentum_value_loop_2489_20261119.json --publish *> "$Work/summary.log"
 if($LASTEXITCODE -ne 0){throw 'Momentum-value full-loop summary failed'}
 @{status='completed';cycles_per_arm=4;updates_per_arm=16;bytes_per_arm=3840}|ConvertTo-Json|
   Set-Content -LiteralPath "$Work/pipeline_status.json"
} catch {
 @{status='failed';error=$_.Exception.Message;pid=$PID}|ConvertTo-Json|
   Set-Content -LiteralPath "$Work/pipeline_status.json"
 throw
}
