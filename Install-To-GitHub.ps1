param([string]$Destination = 'D:\GitHub\spam-email-detection')
$ErrorActionPreference = 'Stop'
$sourceRoot = [IO.Path]::GetFullPath($PSScriptRoot).TrimEnd('\')
$targetRoot = [IO.Path]::GetFullPath($Destination).TrimEnd('\')
$expectedTarget = [IO.Path]::GetFullPath('D:\GitHub\spam-email-detection').TrimEnd('\')
if ($targetRoot -ne $expectedTarget) { throw 'This installer is prepared only for D:\GitHub\spam-email-detection.' }
if (!(Test-Path -LiteralPath (Join-Path $targetRoot '.git'))) { throw 'The destination must be your existing Git repository.' }
if ($sourceRoot -eq $targetRoot) { throw 'Run the installer from the extracted integration package.' }
$backupRoot = Join-Path $targetRoot ('.integration-backups\' + (Get-Date -Format 'yyyyMMdd-HHmmss-fff'))
$sourceFiles = @(Get-ChildItem -LiteralPath $sourceRoot -Recurse -File | Where-Object { $_.FullName -notmatch '\\(__pycache__|\.git|\.venv)\\' -and $_.Name -ne 'Install-To-GitHub.ps1' })
foreach ($sourceFile in $sourceFiles) {
    $relativeName = $sourceFile.FullName.Substring($sourceRoot.Length).TrimStart('\')
    $targetFile = [IO.Path]::GetFullPath((Join-Path $targetRoot $relativeName))
    if (!$targetFile.StartsWith($targetRoot + '\', [StringComparison]::OrdinalIgnoreCase)) { throw 'Destination path escaped the repository.' }
    if (Test-Path -LiteralPath $targetFile -PathType Leaf) {
        $backupFile = Join-Path $backupRoot $relativeName
        New-Item -ItemType Directory -Path (Split-Path $backupFile) -Force | Out-Null
        Copy-Item -LiteralPath $targetFile -Destination $backupFile
    }
    New-Item -ItemType Directory -Path (Split-Path $targetFile) -Force | Out-Null
    if ($relativeName -eq '.gitignore' -and (Test-Path -LiteralPath $targetFile)) {
        $existing = Get-Content -LiteralPath $targetFile -Raw
        $newRules = Get-Content -LiteralPath $sourceFile.FullName -Raw
        Set-Content -LiteralPath $targetFile -Value ($existing + "`n# Integrated spam project rules`n" + $newRules) -Encoding utf8
    } else { Copy-Item -LiteralPath $sourceFile.FullName -Destination $targetFile -Force }
}
Write-Host "Combined project installed in $targetRoot"
Write-Host "Replaced files were backed up in $backupRoot"
Write-Host 'No original extra files were deleted, and no Git commit or push was made.'
Write-Host 'Next: read README.md, install requirements, then run python app/app.py.'
