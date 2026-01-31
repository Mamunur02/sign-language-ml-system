Write-Host "=============================="
Write-Host "PROJECT SNAPSHOT (SHORT)"
Write-Host "=============================="
Write-Host ""

# Focused trees (avoid venv/.git explosions)
Write-Host "---- Repository Trees (focused) ----"

$roots = @("src", "data", "tools")
$ignore = @(
    ".git",
    "venv",
    ".venv",
    "__pycache__",
    ".pytest_cache",
    "node_modules"
) -join " "

foreach ($r in $roots) {
    if (Test-Path $r) {
        Write-Host ""
        Write-Host "---- $r/ (tree, depth ~6, ignoring noisy dirs) ----"
        tree $r /A /F /I "$ignore"
    }
}

Write-Host ""
Write-Host "---- Top-level (quick list) ----"
Get-ChildItem -Force | Select-Object Mode, Name
Write-Host ""

Write-Host "---- Git Status ----"
git status
Write-Host ""

Write-Host "---- Git Diff (summary) ----"
git diff --stat
Write-Host ""

Write-Host "---- src/ Files (relative) ----"
if (Test-Path "src") {
    Get-ChildItem src -Recurse -File | ForEach-Object {
        $_.FullName.Replace((Get-Location).Path + "\", "")
    }
} else {
    Write-Host "No src/ directory found."
}
Write-Host ""

# Optional: print only small, high-signal files
$filesToPrint = @(
    "README.md",
    ".gitignore",
    "requirements.txt",
    "pyproject.toml"
)

foreach ($file in $filesToPrint) {
    if (Test-Path $file) {
        Write-Host "---- $file ----"
        Get-Content $file
        Write-Host ""
    }
}
