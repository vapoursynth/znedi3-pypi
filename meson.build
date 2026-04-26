project('ZNEDI3', 'cpp',
    default_options: ['buildtype=release', 'warning_level=2', 'b_lto=true', 'b_ndebug=if-release', 'cpp_std=c++14'],
    license: 'GPL-2.0',
    license_files: 'gpl2.txt',
    meson_version: '>=1.2.3',
    version: '3.0',
)

graphengine_sources = [
	'graphengine/graphengine/cpuinfo.h',
	'graphengine/graphengine/node.h',
	'graphengine/graphengine/state.h',
	'graphengine/graphengine/x86/cpuinfo_x86.h',
	'graphengine/include/graphengine/filter.h',
	'graphengine/include/graphengine/graph.h',
	'graphengine/include/graphengine/namespace.h',
	'graphengine/include/graphengine/types.h',
	'graphengine/graphengine/cpuinfo.cpp',
	'graphengine/graphengine/graph.cpp',
	'graphengine/graphengine/node.cpp',
	'graphengine/graphengine/x86/cpuinfo_x86.cpp'
]

znedi3_sources = [
	'znedi3/align.h',
	'znedi3/alloc.h',
	'znedi3/ccdep.h',
	'znedi3/cpuinfo.h',
	'znedi3/kernel.h',
	'znedi3/weights.h',
	'znedi3/x86/kernel_x86.h',
	'znedi3/x86/kernel_avx_common.h',
	'znedi3/x86/kernel_sse_commmon.h',
	'znedi3/x86/cpuinfo_x86.h',
	'znedi3/znedi3.h',
	'znedi3/znedi3_impl.h',
	'znedi3/kernel.cpp',
	'znedi3/weights.cpp',
	'znedi3/x86/cpuinfo_x86.cpp',
	'znedi3/x86/kernel_x86.cpp',
	'znedi3/znedi3.cpp',
	'znedi3/znedi3_impl.cpp'
]

vsznedi3_sources = [
	'vsznedi3/vsznedi3.cpp',
]

vsxx_sources = [
	'vsxx/vapoursynth/VapourSynth4.h',
	'vsxx/vapoursynth/VSConstants4.h',
	'vsxx/vapoursynth/VSHelper4.h',
	'vsxx/VapourSynth4++.hpp',
	'vsxx/vsxx4_pluginmain.h',
    'vsxx/vsxx4_pluginmain.cpp'
]

testapp_sources = [
    'testapp/aligned_malloc.h',
	'testapp/argparse.h',
	'testapp/mmap.h',
	'testapp/timer.h',
	'testapp/win32_bitmap.h',
	'testapp/argparse.cpp',
	'testapp/main.cpp',
	'testapp/mmap.cpp',
	'testapp/win32_bitmap.cpp'
]

sources = graphengine_sources + vsxx_sources + znedi3_sources + vsznedi3_sources

incdir = include_directories(
    run_command(
        find_program('python', 'python3'),
        '-c',
        'import vapoursynth as vs; print(vs.get_include())',
        check: true,
    ).stdout().strip(),
    'graphengine/include',
    'znedi3',
    'vsxx',
    'vsxx/vapoursynth'
)

cxx = meson.get_compiler('cpp')

add_project_arguments(
    cxx.get_supported_arguments(
        '-DGRAPHENGINE_IMPL_NAMESPACE=znedi3',
    ),
    language: 'cpp',
)

libs = []

if host_machine.cpu_family() in ['x86', 'x86_64']    
    add_project_arguments(
        cxx.get_supported_arguments(
            '-D_CRT_SECURE_NO_WARNINGS',
            '-DZNEDI3_X86',
            '-DZNEDI3_X86_AVX512',
        ),
        language: 'cpp',
    )
    
    libs += static_library('kernel_avx', 'znedi3/x86/kernel_avx.cpp',
        include_directories: incdir,
        cpp_args: ['-mavx', '-mtune=sandybridge']
    )
    
    libs += static_library('kernel_avx2', 'znedi3/x86/kernel_avx2.cpp',
        include_directories: incdir,
        cpp_args: ['-mavx2', '-mfma', '-mtune=haswell']
    )
    
    libs += static_library('kernel_f16c', 'znedi3/x86/kernel_f16c.cpp',
        include_directories: incdir,
        cpp_args: ['-mavx', '-mf16c', '-mtune=ivybridge']
    )
    
    libs += static_library('kernel_sse', 'znedi3/x86/kernel_sse.cpp',
        include_directories: incdir,
        cpp_args: ['-msse']
    )
    
    libs += static_library('kernel_sse2', 'znedi3/x86/kernel_sse2.cpp',
        include_directories: incdir,
        cpp_args: ['-msse2']
    )
    
    libs += static_library('kernel_avx512', 'znedi3/x86/kernel_avx512.cpp',
        include_directories: incdir,
        cpp_args: ['-mavx512f', '-mfma', '-mtune=skylake-avx512']
    )
endif

py = import('python').find_installation(pure: false)

shared_module('znedi3',
    sources,
    gnu_symbol_visibility: 'hidden',
    include_directories: incdir,
    link_with: libs,
    install: true,
    install_dir: py.get_install_dir() / 'vapoursynth/plugins',
    name_prefix: '',
)

install_data(
    'nnedi3_weights.bin',
    install_dir: py.get_install_dir() / 'vapoursynth/plugins',
)