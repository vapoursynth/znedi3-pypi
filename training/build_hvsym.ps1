# Build the znedi3 VapourSynth plugin with the hvsym changes, using clang-cl + ninja in the VS18 env.
$ErrorActionPreference = 'Stop'
$vcvars = 'C:\Program Files\Microsoft Visual Studio\18\Community\VC\Auxiliary\Build\vcvars64.bat'
$llvm   = 'C:\Program Files\Microsoft Visual Studio\18\Community\VC\Tools\Llvm\x64\bin'
$ninja  = 'C:\Program Files\Microsoft Visual Studio\18\Community\Common7\IDE\CommonExtensions\Microsoft\CMake\Ninja'
$pydir  = 'C:\vapoursynth-portable'
$py     = "$pydir\python.exe"
$proj   = 'C:\Libraries\znedi3-pypi'
$build  = "$proj\build_hvsym"

# import vcvars64 into this process
cmd /c "`"$vcvars`" >nul 2>&1 && set" | ForEach-Object {
  if ($_ -match '^([^=]+)=(.*)$') { [System.Environment]::SetEnvironmentVariable($matches[1], $matches[2]) }
}
$env:PATH = "$pydir;$llvm;$ninja;$env:PATH"
$env:CC  = 'clang-cl'
$env:CXX = 'clang-cl'

Write-Output "clang-cl: $((Get-Command clang-cl).Source)"
Write-Output "ninja:    $((Get-Command ninja).Source)"
Write-Output "python:   $((Get-Command python).Source)"

Set-Location $proj
if (Test-Path $build) { Remove-Item -Recurse -Force $build }

Write-Output "=== meson setup ==="
& $py -m mesonbuild.mesonmain setup $build --buildtype=release -Db_lto=false -Db_ndebug=true
if ($LASTEXITCODE -ne 0) { Write-Output "SETUP FAILED $LASTEXITCODE"; exit 1 }

Write-Output "=== meson compile ==="
& $py -m mesonbuild.mesonmain compile -C $build
if ($LASTEXITCODE -ne 0) { Write-Output "COMPILE FAILED $LASTEXITCODE"; exit 1 }

Write-Output "=== artifacts ==="
Get-ChildItem $build -Filter '*.dll' | Select-Object FullName, Length
Write-Output "BUILD OK"
