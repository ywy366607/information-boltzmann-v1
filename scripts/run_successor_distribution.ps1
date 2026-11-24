$ErrorActionPreference='Stop'
Set-Location -LiteralPath 'D:/information-boltzmann-v1'
$work='D:/information-boltzmann-v1/scratch/cgb_successor_distribution_2493_20261119'
$control='D:/information-boltzmann-v1/scratch/cgb_self_execution_loop_2493_20261119/native'
$source='D:/information-boltzmann-v1/scratch/cgb_momentum_value_2489_20261119/carry/body/last.pt.gz'
$python='D:/conda_envs/vox/python.exe'
$measurement='D:/information-boltzmann-v1/results/published/cgb_successor_distribution_measurement_2493_20261119.json'
$env:OMP_NUM_THREADS='1';$env:MKL_NUM_THREADS='1';$env:PYTHONUTF8='1'
if(Test-Path -LiteralPath $work){throw 'Preserve existing successor-distribution experiment'}
if((Get-FileHash -LiteralPath $source -Algorithm SHA256).Hash.ToLowerInvariant() -ne '7cccc744d4549eca2662727cba6b4c900609393243b9fb557dee6a2f17498107'){throw 'Source changed'}
if((Get-PSDrive D).Free -lt 8GB){throw 'Insufficient temporary workspace'}
New-Item -ItemType Directory -Path $work|Out-Null
$files=@('scripts/ib_local/successor_distribution.py','scripts/ib_local/epistemic_value_reference.py',
 'scripts/ib_local/navigation_causal_program.py','scripts/ib_local/causal_use_value.py',
 'scripts/ib_local/causal_relation_program.py','scripts/ib_local/prospective_decision.py',
 'scripts/ib_local/prospective_self_execution.py','scripts/ib_local/run_mother_continuation.py',
 'scripts/run_native_preview_restore_comparison.py')
$hashes=@{};foreach($f in $files){$hashes[$f]=(Get-FileHash -LiteralPath $f -Algorithm SHA256).Hash}
$hashes|ConvertTo-Json|Set-Content "$work/source_hashes.json"
function Assert-FrozenRuntime {foreach($f in $files){if((Get-FileHash -LiteralPath $f -Algorithm SHA256).Hash -ne $hashes[$f]){throw "Runtime changed: $f"}}}
try {
 @{status='running';step=3;stage='same-boundary three-transition measurement';pid=$PID}|ConvertTo-Json|Set-Content "$work/pipeline_status.json"
 & $python -m scripts.audit_native_successor_distribution --checkpoint $source --reference $control `
   --work "$work/measurement" --output $measurement *> "$work/measurement.log"
 if($LASTEXITCODE -ne 0){throw "Matched measurement failed ($LASTEXITCODE)"}
 Assert-FrozenRuntime
 @{status='running';step=4;stage='complete distribution learning loop';pid=$PID}|ConvertTo-Json|Set-Content "$work/pipeline_status.json"
 & $python -m scripts.run_native_preview_restore_comparison --checkpoint $source --work "$work/distribution" `
   --cycles 4 --native-decision --native-word-execution --successor-distribution *> "$work/distribution.log"
 if($LASTEXITCODE -ne 0){throw "Distribution loop failed ($LASTEXITCODE)"}
 Assert-FrozenRuntime
 @{status='running';step=4;stage='complete restore';pid=$PID}|ConvertTo-Json|Set-Content "$work/pipeline_status.json"
 & $python -m scripts.audit_native_relation_use_restore --work "$work/distribution" `
   --output "$work/distribution_restore.json" *> "$work/restore.log"
 if($LASTEXITCODE -ne 0){throw 'Distribution full restore failed'}
 & $python -m scripts.summarize_successor_distribution --work $work --control $control --measurement $measurement `
   --output results/published/cgb_successor_distribution_loop_2493_20261119.json *> "$work/summary.log"
 if($LASTEXITCODE -ne 0){throw 'Complete successor-distribution summary failed'}
 @{status='completed';steps=@(1,2,3,4);cycles=4;updates=16;external_bytes=3840}|ConvertTo-Json|Set-Content "$work/pipeline_status.json"
}catch{@{status='failed';error=$_.Exception.Message;pid=$PID}|ConvertTo-Json|Set-Content "$work/pipeline_status.json";throw}
