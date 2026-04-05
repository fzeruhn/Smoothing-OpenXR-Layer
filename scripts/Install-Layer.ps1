$RegistryPath = "HKLM:\Software\Khronos\OpenXR\1\ApiLayers\Implicit"
$JsonPath = Join-Path "$PSScriptRoot" "openxr-api-layer.json"
if (-not (Test-Path $JsonPath)) {
	$JsonPath = Join-Path "$PSScriptRoot" "..\bin\x64\Release\openxr-api-layer.json"
}
if (-not (Test-Path $JsonPath)) {
	throw "Could not locate openxr-api-layer.json next to script or in ..\bin\x64\Release\. Build Release first."
}
$JsonPath = [System.IO.Path]::GetFullPath($JsonPath)
Start-Process -FilePath powershell.exe -Verb RunAs -Wait -ArgumentList @"
	& {
		If (-not (Test-Path $RegistryPath)) {
			New-Item -Path $RegistryPath -Force | Out-Null
		}
		New-ItemProperty -Path $RegistryPath -Name '$jsonPath' -PropertyType DWord -Value 0 -Force | Out-Null
	}
"@
