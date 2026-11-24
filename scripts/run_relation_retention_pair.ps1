param([string]$Work='F:/cgb_relation_retention_2481_20261119')
$ErrorActionPreference='Stop'
$repo='D:/information-boltzmann-v1'
$python='D:/conda_envs/vox/python.exe'
$sourceBody='F:/cgb_native_retention_2473_20261118/apply/body/last.pt.gz'
$env:OMP_NUM_THREADS='1'; $env:MKL_NUM_THREADS='1'; $env:PYTHONUTF8='1'
Set-Location -LiteralPath $repo
New-Item -ItemType Directory -Path $Work -Force | Out-Null
$sourceFiles=@('scripts/ib_local/relation_retention_use.py','scripts/ib_local/causal_relation_program.py',
 'scripts/ib_local/epistemic_value_reference.py','scripts/ib_local/shared_unknown_bindings.py',
 'scripts/ib_local/generative_relations.py','scripts/ib_local/joint_relation_update.py',
 'scripts/ib_local/relation_evidence.py','scripts/ib_local/finite_use_value.py',
 'scripts/ib_local/navigation_causal_program.py','scripts/ib_local/run_mother_continuation.py',
 'scripts/ib_local/navigation_settlement.py','scripts/ib_local/navigation_direct_read.py',
 'scripts/run_native_preview_restore_comparison.py')
$hashes=@{}
foreach($item in $sourceFiles){$hashes[$item]=(Get-FileHash -LiteralPath $item -Algorithm SHA256).Hash}
$hashes | ConvertTo-Json | Set-Content -LiteralPath "$Work/source_hashes.json" -Encoding utf8
foreach($arm in @('observe','apply')){
 foreach($part in @(1,2)){
  foreach($item in $sourceFiles){
   if((Get-FileHash -LiteralPath $item -Algorithm SHA256).Hash -ne $hashes[$item]){throw "Runtime source changed: $item"}
  }
  @{status='running';arm=$arm;part=$part;pid=$PID} | ConvertTo-Json | Set-Content -LiteralPath "$Work/pipeline_status.json"
  $arguments=@('-m','scripts.run_native_preview_restore_comparison','--checkpoint',$sourceBody,
    '--work',"$Work/$arm",'--cycles','4','--relation-retention-use',$arm,
    '--world-activations',"$repo/scratch/relation_retention_${arm}_activations")
  if($part -eq 2){$arguments+='--resume-work'}
  & $python @arguments *> "$Work/${arm}_${part}.log"
  if($LASTEXITCODE -ne 0){
   @{status='failed';arm=$arm;part=$part;exit=$LASTEXITCODE} | ConvertTo-Json | Set-Content -LiteralPath "$Work/pipeline_status.json"
   throw "Relation retention arm failed: $arm part $part"
  }
 }
}
@{status='completed';external_per_arm=8;updates_per_arm=32;bytes_per_arm=7680} |
 ConvertTo-Json | Set-Content -LiteralPath "$Work/pipeline_status.json"
