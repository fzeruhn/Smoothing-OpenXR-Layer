$JsonPath = Join-Path "$PSScriptRoot" "openxr-api-layer.json"
if (-not (Test-Path $JsonPath)) {
	$JsonPath = Join-Path "$PSScriptRoot" "..\bin\x64\Release\openxr-api-layer.json"
}
if (-not (Test-Path $JsonPath)) {
	throw "Could not locate openxr-api-layer.json next to script or in ..\bin\x64\Release\."
}
$JsonPath = [System.IO.Path]::GetFullPath($JsonPath)
Start-Process -FilePath powershell.exe -Verb RunAs -Wait -ArgumentList @"
	& {
		Remove-ItemProperty -Path HKLM:\Software\Khronos\OpenXR\1\ApiLayers\Implicit -Name '$jsonPath' -Force | Out-Null
	}
"@
