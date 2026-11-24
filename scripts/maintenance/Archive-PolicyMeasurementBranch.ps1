param([Parameter(Mandatory=$true)][string]$Root,[Parameter(Mandatory=$true)][string]$Temporary)
$ErrorActionPreference='Stop'
function Get-ArchiveSHA256([string]$Path) {
 $stream=[IO.File]::OpenRead($Path)
 try {
  $hash=[Security.Cryptography.SHA256]::Create()
  try { [BitConverter]::ToString($hash.ComputeHash($stream)).Replace('-','') } finally { $hash.Dispose() }
 } finally { $stream.Dispose() }
}
$rootPath=(Resolve-Path -LiteralPath $Root).Path
$tempPath=(Resolve-Path -LiteralPath $Temporary).Path
if (-not $rootPath.StartsWith('F:\mengli-wood-policy-use-20261120\',[StringComparison]::OrdinalIgnoreCase)) {throw 'Unexpected branch root'}
if (-not $tempPath.StartsWith('C:\Users\85750\AppData\Local\Temp\mengli-wood-policy-use-20261120\branches\',[StringComparison]::OrdinalIgnoreCase)) {throw 'Unexpected temporary root'}
$result=Get-Content -LiteralPath (Join-Path $rootPath 'run\result.json') -Raw | ConvertFrom-Json
if ($result.status -ne 'completed') {throw 'Incomplete branch'}
$active=@(Get-CimInstance Win32_Process | Where-Object {$_.Name -match '^python' -and $_.CommandLine})
if (@($active | Where-Object {$_.CommandLine.Replace('/','\').Contains($rootPath+'\') -or $_.CommandLine.Replace('/','\').Contains($tempPath+'\')}).Count) {throw 'Branch process still active'}
$archive='G:\mengli-retired-policy-use-20261120'
New-Item -ItemType Directory -Path $archive -Force | Out-Null
$archive=(Resolve-Path -LiteralPath $archive).Path
$group=Join-Path $archive (Split-Path $rootPath -Leaf)
New-Item -ItemType Directory -Path $group -Force | Out-Null
$group=(Resolve-Path -LiteralPath $group).Path
if (-not $group.StartsWith($archive+'\',[StringComparison]::OrdinalIgnoreCase)) {throw 'Archive escaped its root'}
$records=@()
$sources=@((Join-Path $rootPath 'run\body'),(Join-Path $rootPath 'run\causal'),$tempPath)
foreach ($source in $sources) {
 $source=(Resolve-Path -LiteralPath $source).Path
 if (-not ($source.StartsWith($rootPath+'\',[StringComparison]::OrdinalIgnoreCase) -or $source -eq $tempPath)) {throw 'Source escaped branch scope'}
 $leaf=if ($source -eq $tempPath) {'temporary'} else {Split-Path $source -Leaf}
 $destination=Join-Path $group $leaf
 if (Test-Path -LiteralPath $destination) {throw 'Archive already exists'}
 $batch=@(Get-ChildItem -LiteralPath $source -Recurse -File | ForEach-Object {
  [PSCustomObject]@{source=$_.FullName;destination=$destination+$_.FullName.Substring($source.Length);bytes=$_.Length;sha256=(Get-ArchiveSHA256 $_.FullName)}
 })
 $batch | ConvertTo-Json -Depth 3 | Set-Content -LiteralPath (Join-Path $group ($leaf+'.manifest.json')) -Encoding utf8
 Move-Item -LiteralPath $source -Destination $destination
 foreach ($entry in $batch) {if ((Get-ArchiveSHA256 $entry.destination) -ne $entry.sha256) {throw 'Archive hash mismatch'}}
 $records += $batch
}
$records | ConvertTo-Json -Depth 3 | Set-Content -LiteralPath (Join-Path $rootPath 'retired_artifacts.json') -Encoding utf8
