$JsonPath = Join-Path "$PSScriptRoot" "openxr-api-layer-32.json"
if (-not (Test-Path $JsonPath)) {
	$JsonPath = Join-Path "$PSScriptRoot" "..\bin\Win32\Release\openxr-api-layer-32.json"
}
if (-not (Test-Path $JsonPath)) {
	throw "Could not locate openxr-api-layer-32.json next to script or in ..\bin\Win32\Release\."
}
$JsonPath = [System.IO.Path]::GetFullPath($JsonPath)
Start-Process -FilePath powershell.exe -Verb RunAs -Wait -ArgumentList @"
	& {
		Remove-ItemProperty -Path HKLM:\Software\WOW6432Node\Khronos\OpenXR\1\ApiLayers\Implicit -Name '$jsonPath' -Force | Out-Null
	}
"@
