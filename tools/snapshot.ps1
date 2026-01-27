Write-Host "=============================="
Write-Host "PROJECT SNAPSHOT (SHORT)"
Write-Host "=============================="
Write-Host ""

# Helper: safe tree that doesn't explode
Write-Host "---- Repository Tree (depth 3, filtered) ----"
tree . /A | Select-String -NotMatch "\\data\\b|\\.git\\b|__pycache__|\\.venv\\b|\\bvenv\\b|node_modules|\\.pytest_cache"
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
