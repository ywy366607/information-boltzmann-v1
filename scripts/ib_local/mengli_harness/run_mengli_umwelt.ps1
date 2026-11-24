# Canonical entry point: MengLi Umwelt (梦漓环世界) & Membrane Bridge service.
$ErrorActionPreference = 'Stop'
if (-not $env:DEEPSEEK_API_KEY) {
    $savedKey = [Environment]::GetEnvironmentVariable('DEEPSEEK_API_KEY', 'User')
    if ($savedKey) { $env:DEEPSEEK_API_KEY = $savedKey }
    Remove-Variable savedKey
}
$service = Join-Path $PSScriptRoot 'mengli_service.mjs'
& node $service @args
exit $LASTEXITCODE
