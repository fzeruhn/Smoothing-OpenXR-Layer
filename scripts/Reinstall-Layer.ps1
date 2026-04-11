$RegistryPath = "HKLM:\Software\Khronos\OpenXR\1\ApiLayers\Implicit"

function Resolve-JsonPath {
    $candidates = @(
        (Join-Path $PSScriptRoot "openxr-api-layer.json"),
        (Join-Path $PSScriptRoot "..\bin\x64\Debug\openxr-api-layer.json"),
        (Join-Path $PSScriptRoot "..\bin\x64\Release\openxr-api-layer.json")
    )

    foreach ($candidate in $candidates) {
        if (Test-Path $candidate) {
            return [System.IO.Path]::GetFullPath($candidate)
        }
    }

    throw "Could not locate openxr-api-layer.json next to script or in ..\bin\x64\Debug|Release\."
}

function Assert-Elevated {
    $identity = [Security.Principal.WindowsIdentity]::GetCurrent()
    $principal = New-Object Security.Principal.WindowsPrincipal($identity)
    return $principal.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)
}

if (-not (Assert-Elevated)) {
    Start-Process -FilePath powershell.exe -Verb RunAs -Wait -ArgumentList @(
        "-ExecutionPolicy", "Bypass",
        "-File", """$PSCommandPath"""
    )
    exit $LASTEXITCODE
}

$JsonPath = Resolve-JsonPath

if (-not (Test-Path $RegistryPath)) {
    New-Item -Path $RegistryPath -Force | Out-Null
}

Remove-ItemProperty -Path $RegistryPath -Name $JsonPath -Force -ErrorAction SilentlyContinue
New-ItemProperty -Path $RegistryPath -Name $JsonPath -PropertyType DWord -Value 0 -Force | Out-Null

Write-Host "Layer reinstalled:"
Write-Host "  $JsonPath"
