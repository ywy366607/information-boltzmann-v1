param([string]$Work='D:/information-boltzmann-v1/scratch/cgb_native_decision_pair_2493_20261119',
      [switch]$Resume,[int]$WaitPreviousProcess=0)
$ErrorActionPreference='Stop'
Set-Location -LiteralPath 'D:/information-boltzmann-v1'
$python='D:/conda_envs/vox/python.exe'
$source='D:/information-boltzmann-v1/scratch/cgb_momentum_value_2489_20261119/carry/body/last.pt.gz'
$env:OMP_NUM_THREADS='1';$env:MKL_NUM_THREADS='1';$env:PYTHONUTF8='1'
if($WaitPreviousProcess -gt 0 -and (Get-Process -Id $WaitPreviousProcess -ErrorAction SilentlyContinue)){
 Wait-Process -Id $WaitPreviousProcess
}
if((Test-Path -LiteralPath "$Work/pipeline_status.json") -and -not $Resume){throw 'Preserve previous work directory'}
New-Item -ItemType Directory -Path $Work -Force|Out-Null
$files=@('scripts/ib_local/prospective_decision.py','scripts/ib_local/epistemic_value_reference.py',
 'scripts/ib_local/navigation_generative_cycle.py','scripts/ib_local/shared_unknown_bindings.py',
 'scripts/ib_local/causal_relation_program.py','scripts/ib_local/causal_use_value.py',
 'scripts/ib_local/generative_relations.py','scripts/ib_local/conditional_actor_reference.py',
 'scripts/ib_local/run_mother_continuation.py','scripts/run_native_preview_restore_comparison.py')
if($Resume -and (Test-Path -LiteralPath "$Work/source_hashes.json") -and -not (Test-Path -LiteralPath "$Work/control_source_hashes.json")){
 Copy-Item -LiteralPath "$Work/source_hashes.json" -Destination "$Work/control_source_hashes.json"
}
$hashes=@{};foreach($file in $files){$hashes[$file]=(Get-FileHash -LiteralPath $file -Algorithm SHA256).Hash}
$hashes|ConvertTo-Json|Set-Content -LiteralPath "$Work/source_hashes.json"
try {
 @{status='running';stage='raw-port held-out check';pid=$PID}|ConvertTo-Json|Set-Content -LiteralPath "$Work/pipeline_status.json"
 if(-not $Resume){
  & $python -m scripts.audit_native_raw_boundary --checkpoint $source --work "$Work/raw_port" `
    --output results/published/cgb_raw_boundary_port_2493_20261119.json *> "$Work/raw_port.log"
  if($LASTEXITCODE -ne 0){throw 'Raw port admission failed'}
 }
 foreach($arm in @('control','native')){
  foreach($file in $files){if((Get-FileHash -LiteralPath $file -Algorithm SHA256).Hash -ne $hashes[$file]){throw "Runtime source changed: $file"}}
  @{status='running';stage='full loop';arm=$arm;pid=$PID}|ConvertTo-Json|Set-Content -LiteralPath "$Work/pipeline_status.json"
  $runArgs=@('-m','scripts.run_native_preview_restore_comparison','--checkpoint',$source,
    '--work',"$Work/$arm",'--cycles','4','--world-activations',"$Work/${arm}_activations")
  if($arm -eq 'control'){$runArgs+='--epistemic-momentum'}else{$runArgs+='--native-decision'}
  if($Resume -and (Test-Path -LiteralPath "$Work/$arm/result.json")){
   $done=Get-Content -LiteralPath "$Work/$arm/result.json" -Raw|ConvertFrom-Json
   if($done.status -ne 'completed' -or $done.origin_cycle -ne 2493 -or $done.cycle -ne 2497 -or $done.cycles -ne 4){throw 'Only completed exact-budget arms can be reused'}
  }else{
   # An interrupted arm has mutable SQL/state beyond its last saved boundary.
   # Preserve it as evidence and restart this bounded arm from the common source.
   if(Test-Path -LiteralPath "$Work/$arm"){
    $root=[IO.Path]::GetFullPath($Work).TrimEnd('\','/')
    $partial=[IO.Path]::GetFullPath("$Work/$arm")
    if(-not $partial.StartsWith($root+[IO.Path]::DirectorySeparatorChar,[StringComparison]::OrdinalIgnoreCase)){throw 'Partial arm is outside work root'}
    $retired="$Work/interrupted_${arm}_$(Get-Date -Format yyyyMMdd_HHmmss)"
    Move-Item -LiteralPath $partial -Destination $retired
    if(Test-Path -LiteralPath "$Work/$arm.log"){Move-Item -LiteralPath "$Work/$arm.log" -Destination "$retired/runner.log"}
    if(Test-Path -LiteralPath "$Work/${arm}_activations"){Move-Item -LiteralPath "$Work/${arm}_activations" -Destination "$retired/activations"}
   }
   & $python @runArgs *> "$Work/$arm.log"
   if($LASTEXITCODE -ne 0){throw "Full loop failed: $arm (exit $LASTEXITCODE); see $Work/$arm.log"}
  }
  & $python -m scripts.audit_native_relation_use_restore --work "$Work/$arm" `
    --output "$Work/${arm}_restore.json" *> "$Work/${arm}_restore.log"
  if($LASTEXITCODE -ne 0){throw "Complete restore failed: $arm"}
 }
 & $python -m scripts.summarize_native_decision_pair --work $Work `
   --output results/published/cgb_native_decision_loop_2493_20261119.json *> "$Work/summary.log"
 if($LASTEXITCODE -ne 0){throw 'Decision pair summary failed'}
 @{status='completed';cycles_per_arm=4;updates_per_arm=16;bytes_per_arm=3840}|ConvertTo-Json|Set-Content -LiteralPath "$Work/pipeline_status.json"
} catch {
 @{status='failed';error=$_.Exception.Message;pid=$PID}|ConvertTo-Json|Set-Content -LiteralPath "$Work/pipeline_status.json"
 throw
}
