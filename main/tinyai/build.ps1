<#
.SYNOPSIS
  Builds tinyai (C + NASM x86-64 assembly) with MSYS2 gcc and NASM.

.DESCRIPTION
  No make/cmake needed.  Produces, in .\build :
    tinyai.exe         the training demo
    tinyai.dll         the same C + assembly as a DLL, for the Lisp front end
    test_kernels.exe   assembly kernels vs. C reference (guard pages, ABI registers)
    test_gradcheck.exe back-propagation vs. finite differences
    bench.exe          kernel + end-to-end benchmarks

.PARAMETER Clean   delete .\build first
.PARAMETER Test    run the two C test programs after building (fails the script if either fails)
.PARAMETER Lisp    also run the Common Lisp test suite (needs SBCL; skipped with a note if absent)
.PARAMETER Run     run the demo after building

.EXAMPLE
  .\build.ps1 -Test -Lisp -Run
#>
[CmdletBinding()]
param(
    [switch]$Clean,
    [switch]$Test,
    [switch]$Lisp,
    [switch]$Run
)

$ErrorActionPreference = 'Stop'

# --- toolchain ---------------------------------------------------------------
foreach ($dir in 'C:\msys64\ucrt64\bin', 'C:\msys64\mingw64\bin') {
    if (-not (Get-Command gcc -ErrorAction SilentlyContinue) -and (Test-Path $dir)) {
        $env:PATH = "$dir;$env:PATH"
    }
}
foreach ($tool in 'gcc', 'nasm') {
    if (-not (Get-Command $tool -ErrorAction SilentlyContinue)) {
        throw "'$tool' not found. Install MSYS2 (https://www.msys2.org), then in a UCRT64 shell: pacman -S mingw-w64-ucrt-x86_64-gcc mingw-w64-ucrt-x86_64-nasm"
    }
}

function Invoke-Tool {
    param([string]$Exe, [string[]]$Arguments)
    Write-Host ("  {0} {1}" -f $Exe, ($Arguments -join ' ')) -ForegroundColor DarkGray
    & $Exe @Arguments
    if ($LASTEXITCODE -ne 0) { throw "$Exe failed with exit code $LASTEXITCODE" }
}

# Everything below uses paths relative to the project root, so the space in
# "C:\User 0\..." can never end up unquoted on a command line.
Push-Location $PSScriptRoot
try {
    if ($Clean -and (Test-Path build)) { Remove-Item build -Recurse -Force }
    New-Item -ItemType Directory -Force build | Out-Null

    # Plain -O2 on purpose (no -march=native): the "scalar" fallback must stay scalar.
    $cflags = @('-std=gnu11', '-O2', '-Wall', '-Wextra', '-Wpedantic')

    Write-Host "[1/3] assemble" -ForegroundColor Cyan
    Invoke-Tool nasm @('-f', 'win64', '-w+all', 'asm\kernels.asm', '-o', 'build\kernels_asm.obj')
    Invoke-Tool nasm @('-f', 'win64', '-w+all', 'asm\abi_check.asm', '-o', 'build\abi_check.obj')   # tests only

    Write-Host "[2/3] compile" -ForegroundColor Cyan
    $core = 'kernels', 'mlp', 'data', 'train'
    foreach ($n in $core + 'main', 'test_kernels', 'test_gradcheck', 'bench') {
        Invoke-Tool gcc ($cflags + @('-c', "src\$n.c", '-o', "build\$n.o"))
    }
    # The foreign-language API is compiled with -DTAI_BUILD_DLL so its functions get exported.
    Invoke-Tool gcc ($cflags + @('-DTAI_BUILD_DLL', '-c', 'src\tinyai_api.c', '-o', 'build\tinyai_api.o'))
    # The benchmark's "what would the compiler do on its own?" reference: same loops,
    # but compiled with every optimisation gcc offers.  Never used by the program itself.
    Invoke-Tool gcc (@('-std=gnu11', '-O3', '-march=native', '-ffast-math', '-funroll-loops',
                       '-Wall', '-Wextra', '-c', 'src\bench_cnative.c', '-o', 'build\bench_cnative.o'))

    Write-Host "[3/3] link (static)" -ForegroundColor Cyan
    $coreObjs = @($core | ForEach-Object { "build\$_.o" }) + 'build\kernels_asm.obj'
    Invoke-Tool gcc (@('-static', '-o', 'build\tinyai.exe',         'build\main.o')           + $coreObjs + '-lm')
    Invoke-Tool gcc (@('-static', '-o', 'build\test_kernels.exe',   'build\test_kernels.o', 'build\abi_check.obj') + $coreObjs + '-lm')
    Invoke-Tool gcc (@('-static', '-o', 'build\test_gradcheck.exe', 'build\test_gradcheck.o') + $coreObjs + '-lm')
    Invoke-Tool gcc (@('-static', '-o', 'build\bench.exe',          'build\bench.o', 'build\bench_cnative.o') + $coreObjs + '-lm')
    # The same C + assembly as a DLL, for the Lisp front end (and any other FFI).
    Invoke-Tool gcc (@('-shared', '-static-libgcc', '-o', 'build\tinyai.dll', 'build\tinyai_api.o') + $coreObjs + '-lm')

    Write-Host "build OK -> build\*.exe, build\tinyai.dll" -ForegroundColor Green

    if ($Test) {
        # (| Out-Host keeps the program's output in order with the headings)
        Write-Host "`n== test_kernels ==" -ForegroundColor Cyan
        & .\build\test_kernels.exe | Out-Host
        if ($LASTEXITCODE -ne 0) { throw "test_kernels failed (exit code $LASTEXITCODE; -1073741819 = access violation: a kernel touched memory outside its buffer)" }
        Write-Host "`n== test_gradcheck ==" -ForegroundColor Cyan
        & .\build\test_gradcheck.exe | Out-Host
        if ($LASTEXITCODE -ne 0) { throw "test_gradcheck failed (exit code $LASTEXITCODE)" }
    }
    if ($Lisp) {
        if (-not (Get-Command sbcl -ErrorAction SilentlyContinue)) {
            Write-Host "`n== lisp tests == SKIPPED: sbcl not found (MSYS2: pacman -S mingw-w64-ucrt-x86_64-sbcl)" -ForegroundColor Yellow
        } else {
            Write-Host "`n== lisp tests ==" -ForegroundColor Cyan
            & sbcl --noinform --non-interactive --no-sysinit --no-userinit --load lisp\test.lisp | Out-Host
            if ($LASTEXITCODE -ne 0) { throw "lisp tests failed (exit code $LASTEXITCODE)" }
        }
    }
    if ($Run) {
        Write-Host ""
        & .\build\tinyai.exe
    }
}
finally {
    Pop-Location
}
