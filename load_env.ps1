$envFile = Join-Path $PSScriptRoot ".env"

if (Test-Path $envFile) {
    Get-Content $envFile | ForEach-Object {
        if ($_ -match '^([^#][^=]*)=(.*)$') {
            [System.Environment]::SetEnvironmentVariable($matches[1].Trim(), $matches[2].Trim(), "Process")
            Write-Host "Set $($matches[1].Trim())"
        }
    }
    Write-Host "`nEnvironment loaded. Now run:"
    Write-Host "  python -m modal run modal_train.py sft"
} else {
    Write-Host "Error: .env file not found at $envFile"
}
