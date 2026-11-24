$ErrorActionPreference='Stop'
Set-Location -LiteralPath 'D:/information-boltzmann-v1'
$work='D:/information-boltzmann-v1/scratch/cgb_self_execution_loop_2493_20261119'
$control='D:/information-boltzmann-v1/scratch/cgb_native_decision_pair_2493_20261119/native'
$source='D:/information-boltzmann-v1/scratch/cgb_momentum_value_2489_20261119/carry/body/last.pt.gz'
$python='D:/conda_envs/vox/python.exe'
$env:OMP_NUM_THREADS='1';$env:MKL_NUM_THREADS='1';$env:PYTHONUTF8='1'
if(Test-Path -LiteralPath $work){throw 'Preserve previous self-execution work'}
$admission=Get-Content 'results/published/cgb_self_execution_admission_2493_20261119.json' -Raw|ConvertFrom-Json
$anchor=Get-Content 'results/published/cgb_word_execution_2493_20261119.json' -Raw|ConvertFrom-Json
if(-not $admission.valid -or -not $anchor.valid -or -not $anchor.checks.native_anchor_exact){throw 'Real operator admission not complete'}
if((Get-FileHash -LiteralPath $source -Algorithm SHA256).Hash.ToLowerInvariant() -ne $admission.source_sha256){throw 'Source changed'}
$reference=Get-Content "$control/result.json" -Raw|ConvertFrom-Json
if($reference.status -ne 'completed' -or $reference.cycle -ne 2497 -or $reference.updates -ne 16){throw 'Historical control incomplete'}
if((Get-PSDrive D).Free -lt 8GB){throw 'Insufficient temporary activation/save workspace'}
New-Item -ItemType Directory -Path $work|Out-Null
@{path=$control;source=$reference.initial_source_sha256;checkpoint=$reference.checkpoint_sha256;role='reused completed schema3 historical word transport'}|ConvertTo-Json|Set-Content "$work/control_reference.json"
$files=@('scripts/ib_local/prospective_self_execution.py','scripts/ib_local/prospective_decision.py',
 'scripts/ib_local/epistemic_value_reference.py','scripts/ib_local/navigation_generative_cycle.py',
 'scripts/ib_local/navigation_mother_boundary.py','scripts/ib_local/run_mother_continuation.py',
 'scripts/run_native_preview_restore_comparison.py')
$hashes=@{};foreach($f in $files){$hashes[$f]=(Get-FileHash -LiteralPath $f -Algorithm SHA256).Hash}
$hashes|ConvertTo-Json|Set-Content "$work/source_hashes.json"
try {
 @{status='running';stage='native full loop';pid=$PID}|ConvertTo-Json|Set-Content "$work/pipeline_status.json"
 & $python -m scripts.run_native_preview_restore_comparison --checkpoint $source --work "$work/native" --cycles 4 `
   --world-activations "$work/native_activations" --native-decision --native-word-execution *> "$work/native.log"
 if($LASTEXITCODE -ne 0){throw "Native loop failed ($LASTEXITCODE)"}
 foreach($f in $files){if((Get-FileHash -LiteralPath $f -Algorithm SHA256).Hash -ne $hashes[$f]){throw "Runtime changed: $f"}}
 @{status='running';stage='complete restore';pid=$PID}|ConvertTo-Json|Set-Content "$work/pipeline_status.json"
 & $python -m scripts.audit_native_relation_use_restore --work "$work/native" --output "$work/native_restore.json" *> "$work/restore.log"
 if($LASTEXITCODE -ne 0){throw 'Complete native restore failed'}
 & $python -m scripts.summarize_native_self_execution --work $work --control $control `
   --output results/published/cgb_native_self_execution_loop_2493_20261119.json *> "$work/summary.log"
 if($LASTEXITCODE -ne 0){throw 'Native self-execution summary failed'}
 @{status='completed';cycles=4;updates=16;external_bytes=3840}|ConvertTo-Json|Set-Content "$work/pipeline_status.json"
}catch{@{status='failed';error=$_.Exception.Message;pid=$PID}|ConvertTo-Json|Set-Content "$work/pipeline_status.json";throw}
