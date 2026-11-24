# Compatibility entry point: one local mother-world service and dashboard.
$ErrorActionPreference = 'Stop'
# A long-running desktop host may predate an updated Windows user environment.
# Reuse the existing provider credential without echoing or persisting it.
if (-not $env:DEEPSEEK_API_KEY) {
    $savedKey = [Environment]::GetEnvironmentVariable('DEEPSEEK_API_KEY', 'User')
    if ($savedKey) { $env:DEEPSEEK_API_KEY = $savedKey }
    Remove-Variable savedKey
}
$service = Join-Path $PSScriptRoot 'mengli_service.mjs'
& node $service @args
exit $LASTEXITCODE
