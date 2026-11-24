$ErrorActionPreference='Stop'
Set-Location -LiteralPath 'D:/information-boltzmann-v1'
$work='D:/information-boltzmann-v1/scratch/cgb_joint_successor_2493_20261119'
$source='D:/information-boltzmann-v1/scratch/cgb_momentum_value_2489_20261119/carry/body/last.pt.gz'
$python='D:/conda_envs/vox/python.exe'
$env:OMP_NUM_THREADS='1';$env:MKL_NUM_THREADS='1';$env:PYTHONUTF8='1'
if(Test-Path -LiteralPath $work){throw 'Preserve existing joint successor experiment'}
if((Get-FileHash -LiteralPath $source -Algorithm SHA256).Hash.ToLowerInvariant() -ne '7cccc744d4549eca2662727cba6b4c900609393243b9fb557dee6a2f17498107'){throw 'Source changed'}
$admission=Get-Content results/published/cgb_successor_address_mass_2493_20261119.json -Raw|ConvertFrom-Json
if(-not $admission.valid -or $admission.status -ne 'completed'){throw 'Geometry admission incomplete'}
if((Get-PSDrive D).Free -lt 8GB){throw 'Insufficient temporary workspace'}
New-Item -ItemType Directory -Path $work|Out-Null
$files=@('scripts/ib_local/joint_successor.py','scripts/ib_local/successor_joint_reference.py',
 'scripts/ib_local/successor_address_geometry.py','scripts/ib_local/successor_distribution.py',
 'scripts/ib_local/epistemic_value_reference.py','scripts/ib_local/causal_relation_program.py',
 'scripts/ib_local/generative_relations.py','scripts/ib_local/navigation_causal_program.py',
 'scripts/ib_local/causal_use_value.py','scripts/ib_local/run_mother_continuation.py',
 'scripts/run_native_preview_restore_comparison.py')
$hashes=@{};foreach($f in $files){$hashes[$f]=(Get-FileHash -LiteralPath $f -Algorithm SHA256).Hash}
$hashes|ConvertTo-Json|Set-Content "$work/source_hashes.json"
try {
 @{status='running';stage='joint full loop';schema=5;pid=$PID}|ConvertTo-Json|Set-Content "$work/pipeline_status.json"
 & $python -m scripts.run_native_preview_restore_comparison --checkpoint $source --work "$work/joint" `
  --cycles 4 --native-decision --native-word-execution --successor-distribution --joint-successor *> "$work/joint.log"
 if($LASTEXITCODE -ne 0){throw "Joint loop failed ($LASTEXITCODE)"}
 foreach($f in $files){if((Get-FileHash -LiteralPath $f -Algorithm SHA256).Hash -ne $hashes[$f]){throw "Runtime changed: $f"}}
 @{status='running';stage='joint full restore';schema=5;pid=$PID}|ConvertTo-Json|Set-Content "$work/pipeline_status.json"
 & $python -m scripts.audit_native_relation_use_restore --work "$work/joint" --output "$work/restore.json" *> "$work/restore.log"
 if($LASTEXITCODE -ne 0){throw 'Joint restore failed'}
 & $python -m scripts.summarize_joint_successor --work $work --output results/published/cgb_joint_successor_loop_2493_20261119.json *> "$work/summary.log"
 if($LASTEXITCODE -ne 0){throw 'Joint summary failed'}
 @{status='completed';schema=5;cycles=4;updates=16;external_bytes=3840}|ConvertTo-Json|Set-Content "$work/pipeline_status.json"
}catch{@{status='failed';error=$_.Exception.Message;schema=5;pid=$PID}|ConvertTo-Json|Set-Content "$work/pipeline_status.json";throw}
