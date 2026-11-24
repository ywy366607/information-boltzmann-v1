$ErrorActionPreference = 'Stop'
$archiveRoot = 'G:\mengli-retired-policy-copies-20261120'
$roots = @(
 'D:\information-boltzmann-v1\scratch\cgb_action_cross_score_2587_20261120_silence',
 'D:\information-boltzmann-v1\scratch\cgb_action_cross_score_2587_20261120_generate',
 'E:\mengli-wood-choice-reuse-2587-20261120',
 'F:\mengli-action-pair-2585-20261120-generate'
)
$active = @(Get-CimInstance Win32_Process | Where-Object {$_.Name -match '^python' -and $_.CommandLine})
$records = @()
New-Item -ItemType Directory -Path $archiveRoot -Force | Out-Null
$archiveRoot = (Resolve-Path -LiteralPath $archiveRoot).Path
foreach ($root in $roots) {
 $resolved = (Resolve-Path -LiteralPath $root).Path
 $result = Get-Content -LiteralPath (Join-Path $resolved 'run\result.json') -Raw | ConvertFrom-Json
 if ($result.status -ne 'completed') { throw "Incomplete branch: $resolved" }
 if (@($active | Where-Object {$_.CommandLine.Replace('/','\').Contains($resolved+'\')}).Count) { throw 'Live branch' }
 $group = Join-Path $archiveRoot (Split-Path $resolved -Leaf)
 New-Item -ItemType Directory -Path $group -Force | Out-Null
 $group = (Resolve-Path -LiteralPath $group).Path
 if (-not $group.StartsWith($archiveRoot+'\',[StringComparison]::OrdinalIgnoreCase)) { throw 'Archive target escaped root' }
 foreach ($relative in @('run\body\last.pt.gz','run\causal')) {
  $source = (Resolve-Path -LiteralPath (Join-Path $resolved $relative)).Path
  if (-not $source.StartsWith($resolved+'\',[StringComparison]::OrdinalIgnoreCase)) { throw 'Source escaped selected branch' }
  $destination = Join-Path $group (Split-Path $source -Leaf)
  if (Test-Path -LiteralPath $destination) { throw 'Archive target already exists' }
  $files = @(if ((Get-Item -LiteralPath $source).PSIsContainer) {Get-ChildItem -LiteralPath $source -File -Recurse} else {Get-Item -LiteralPath $source})
  $batch = @()
  foreach ($file in $files) {
   $suffix = if ($file.FullName -eq $source) {''} else {$file.FullName.Substring($source.Length)}
   $batch += [PSCustomObject]@{source=$file.FullName;destination=$destination+$suffix;bytes=$file.Length;sha256=(Get-FileHash -LiteralPath $file.FullName -Algorithm SHA256).Hash}
  }
  $batch | ConvertTo-Json -Depth 3 | Set-Content -LiteralPath (Join-Path $group ((Split-Path $source -Leaf)+'.manifest.json')) -Encoding utf8
  Move-Item -LiteralPath $source -Destination $destination
  foreach ($entry in $batch) {
   if ((Get-FileHash -LiteralPath $entry.destination -Algorithm SHA256).Hash -ne $entry.sha256) {throw 'Moved file hash mismatch'}
  }
  $records += $batch
 }
}
$records | ConvertTo-Json -Depth 4 | Set-Content -LiteralPath 'D:\information-boltzmann-v1\results\published\cgb_policy_retired_copies_archive_20261120.json' -Encoding utf8
[PSCustomObject]@{archived_bytes=($records|Measure-Object bytes -Sum).Sum;archive=$archiveRoot;drives=@(Get-PSDrive C,D,E,F|Select-Object Name,Free)} | ConvertTo-Json -Depth 3
