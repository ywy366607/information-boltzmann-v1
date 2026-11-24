# Removes only the explicitly reviewed retired intermediate files.
# User authorization: unused project intermediate PT files may be deleted.
# No directory traversal, recursive deletion, or removal of reports/archives.
$ErrorActionPreference = 'Stop'
$repo = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..\..'))
$source = Join-Path $repo 'results\cgb_generative_joint_20261113\guarded_from_3887\body\last.pt.gz'
$guardPath = Join-Path $repo 'results\cgb_generative_joint_20261113\guarded_from_3895_disk_guard\status.json'
$guard = Get-Content -LiteralPath $guardPath -Raw | ConvertFrom-Json
$sourceHash = (Get-FileHash -LiteralPath $source -Algorithm SHA256).Hash.ToLowerInvariant()
if ($sourceHash -ne $guard.source_sha256) {
    throw 'Protected 3895 checkpoint hash does not match the registered source.'
}

$relativeFiles = @(
    'scratch\marked_next_pending_20261108\pending_recovery.pt',
    'scratch\joint_feedback_20261108\latest.pt',
    'scratch\marked_feedback_20261108\latest.pt',
    'results\cgb_water_last_attempt_20261109\scalar_trial_retry\admission_checkpoint.pt',
    'results\cgb_native_s_credit_20261108\local_storage_repair_serial_v2\admission\admission_before_third.pt',
    'results\cgb_generative_joint_20261113\guarded_from_3895\body\last.tmp.gz'
)
$records = @()
foreach ($relative in $relativeFiles) {
    $path = [IO.Path]::GetFullPath((Join-Path $repo $relative))
    if (-not $path.StartsWith($repo + '\', [StringComparison]::OrdinalIgnoreCase) -or $path -eq $source) {
        throw "Target outside the authorized project scope: $path"
    }
    if (-not (Test-Path -LiteralPath $path -PathType Leaf)) { continue }
    # Resolve every component: a link/junction may redirect a literal path.
    $component = Get-Item -LiteralPath $path
    while ($null -ne $component) {
        if (($component.Attributes -band [IO.FileAttributes]::ReparsePoint) -ne 0) {
            throw "Linked path component refused: $($component.FullName)"
        }
        if ($component.FullName -eq $repo) { break }
        if ($component -is [IO.FileInfo]) { $component = $component.Directory }
        else { $component = $component.Parent }
    }
    $item = Get-Item -LiteralPath $path
    $records += [pscustomobject]@{
        path = $path
        bytes = $item.Length
        sha256 = (Get-FileHash -LiteralPath $path -Algorithm SHA256).Hash.ToLowerInvariant()
        reason = 'Retired research intermediate or incomplete startup checkpoint; reports retained'
    }
}
$manifest = Join-Path $repo 'results\cgb_generative_joint_20261113\authorized_checkpoint_cleanup_20261113.json'
$report = [pscustomobject]@{
    authorization = 'User explicitly allowed unused project intermediate PT deletion'
    protected_source = $source
    protected_source_sha256 = $sourceHash
    files = $records
    planned_bytes = ($records | Measure-Object -Property bytes -Sum).Sum
    status = 'planned'
    deleted_files = @()
}
$report | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath $manifest -Encoding UTF8
foreach ($record in $records) {
    # Recheck identity immediately before each irreversible file operation.
    if ((Get-FileHash -LiteralPath $record.path -Algorithm SHA256).Hash.ToLowerInvariant() -ne $record.sha256) {
        throw "Reviewed file changed; cleanup stopped: $($record.path)"
    }
    Remove-Item -LiteralPath $record.path -Force
    $report.deleted_files += $record.path
    $report | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath $manifest -Encoding UTF8
}
$after = (Get-FileHash -LiteralPath $source -Algorithm SHA256).Hash.ToLowerInvariant()
if ($after -ne $sourceHash) { throw 'Protected source changed during cleanup.' }
$report.status = 'completed'
$report | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath $manifest -Encoding UTF8
[pscustomobject]@{
    DeletedFiles = $report.deleted_files.Count
    FreedGiB = [math]::Round($report.planned_bytes / 1GB, 3)
    DFreeGiB = [math]::Round((Get-PSDrive -Name D).Free / 1GB, 3)
    Protected3895Unchanged = $true
} | Format-List
