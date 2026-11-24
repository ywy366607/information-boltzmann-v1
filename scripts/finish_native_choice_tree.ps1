param([int]$AuditProcess=4908,
      [string]$Work='D:/information-boltzmann-v1/scratch/cgb_choice_tree_2493_20261119')
$ErrorActionPreference='Stop'
Set-Location -LiteralPath 'D:/information-boltzmann-v1'
$env:OMP_NUM_THREADS='1';$env:MKL_NUM_THREADS='1';$env:PYTHONUTF8='1'
try {
    @{status='waiting';audit_pid=$AuditProcess;pid=$PID}|ConvertTo-Json|
        Set-Content -LiteralPath "$Work/finalizer_status.json"
    if(Get-Process -Id $AuditProcess -ErrorAction SilentlyContinue){
        Wait-Process -Id $AuditProcess
    }
    & 'D:/conda_envs/vox/python.exe' -m scripts.finalize_native_choice_tree --work $Work `
        --report 'D:/information-boltzmann-v1/results/published/cgb_choice_tree_2493_20261119.json'
    if($LASTEXITCODE -ne 0){throw 'Choice-tree finalization failed; original leaf files preserved'}
    @{status='completed';pid=$PID}|ConvertTo-Json|Set-Content -LiteralPath "$Work/finalizer_status.json"
} catch {
    @{status='failed';error=$_.Exception.Message;pid=$PID}|ConvertTo-Json|
        Set-Content -LiteralPath "$Work/finalizer_status.json"
    throw
}
