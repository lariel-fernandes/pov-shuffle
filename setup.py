import itertools
import os
import typing
from pathlib import Path
from typing import NamedTuple, ForwardRef

import cutlass_library
import torch.utils.cpp_extension
from setuptools import setup


class EnvVars(NamedTuple):
    """Environment variables for build configuration."""

    POVS_CUDA_DEBUG_MODE:   bool      = "false"  # Build with debug flags
    POVS_CUDA_CUDA_ARCH:    list[str] = "native" # Comma-separated list of CUDA architectures (e.g.: 75,86 or native)
    POVS_CUDA_PBLOCK_SIZES: list[int] = "32"     # Comma-separated list of physical block sizes, in number of instances
    POVS_CUDA_VBLOCK_SIZES: list[int] = "3"      # Comma-separated list of virtual block sizes, in number of physical blocks
    POVS_CUDA_DTYPES:       list[str] = "float"  # Comma-separated list of shuffled data types (options: int, long, half, float, double)


def load_env_vars() -> EnvVars:
    """Load environment variables for build configuration."""
    env_vars = EnvVars()

    for key, _type in EnvVars.__annotations__.items():
        _type = eval(_type.__forward_arg__) if isinstance(_type, ForwardRef) else _type
        val = os.environ.get(key) or EnvVars._field_defaults[key]

        if typing.get_origin(_type) == list:
            val = val.split(",")
            if typing.get_args(_type) == (int,):
                val = [int(x) for x in val]

        elif _type == bool:
            val = val.lower() in ("1", "true", "yes")

        env_vars = env_vars._replace(**{key: val})

    return env_vars


def get_cuda_arch_flags(archs: list[str]) -> list[str]:
    """Convert architecture list to nvcc flags."""
    return ["-arch=native"] if archs == ["native"] else [
        f"-gencode=arch=compute_{arch_num},code=sm_{arch_num}"
        for arch_num in [arch.split("_")[-1] for arch in archs]
    ]


def find_sources(path: str, source_file_types: set[str]) -> list[str]:
    """Find all source files recursively"""
    return [
        str(os.path.join(root, file))
        for root, dirs, files in os.walk(path)
        for file in files
        if any(file.endswith(x) for x in source_file_types)
        and not "test" in file
    ]


def find_cuda_home() -> str | None:
    """Find CUDA Toolkit installation directory."""
    for path in [
        os.environ.get("CUDA_HOME"),
        os.environ.get("CUDA_PATH"),
        "/usr/local/cuda",
        "/usr/lib/cuda",
        "/opt/cuda",
    ]:
        if path and Path(path).is_dir():
            return path


def get_macros(opts: EnvVars) -> list[tuple[str, str | None]]:
    """Generate macros from build options."""
    return [
        ("PBLOCK_SIZE_CASES(lambda)", " ".join([
            f"case {x}: {{ constexpr int PBLOCK_SIZE = {x}; return lambda(); }};"
            for x in opts.POVS_CUDA_PBLOCK_SIZES
        ])),
        ("VBLOCK_SIZE_CASES(lambda)", " ".join([
            f"case {x}: {{ constexpr int VBLOCK_SIZE = {x}; return lambda(); }};"
            for x in opts.POVS_CUDA_VBLOCK_SIZES
        ])),
    ]


def get_generated_files(opts: EnvVars) -> list[tuple[str, str]]:
    """Get codegen files for build options."""
    return [
        ("povs_cuda_template_instances.gen.inc", "\n".join([
            "INSTANTIATE_POVS_CUDA_%s(%s, %s)" % values
            for values in itertools.product(
                map(str.upper, opts.POVS_CUDA_DTYPES),
                opts.POVS_CUDA_PBLOCK_SIZES,
                opts.POVS_CUDA_VBLOCK_SIZES,
            )
        ])),
    ]


assert cutlass_library.__file__
assert (cuda_home := find_cuda_home())

build_options = load_env_vars()
os.environ["CUDA_HOME"] = torch.utils.cpp_extension.CUDA_HOME = cuda_home


class BuildExtension(torch.utils.cpp_extension.BuildExtension):
    """Custom build extension command."""

    def run(self):
        gen_path = Path(self.build_temp) / "codegen"
        gen_path.mkdir(parents=True, exist_ok=True)

        macros = get_macros(build_options)

        for file_name, content in get_generated_files(build_options):
            (gen_path / file_name).write_text(content)

        for ext in self.extensions:
            ext.define_macros += macros
            ext.include_dirs.append(str(gen_path))

        super().run()


setup(
    cmdclass={
        "build_ext": BuildExtension,
    },
    ext_modules=[
        torch.utils.cpp_extension.CUDAExtension(
            name="povs._cuda",
            sources=find_sources(
                path="src/povs/__cuda",
                source_file_types={".cpp", ".cu"},
            ),
            include_dirs=[
                str(Path(cutlass_library.__file__).parent / "source" / "include"),
            ],
            extra_compile_args={
                "cxx": ["-std=c++20"] +
                       (["-g", "-O0"] if build_options.POVS_CUDA_DEBUG_MODE else ["-O3"]),
                "nvcc": ["-std=c++20"] +
                        get_cuda_arch_flags(build_options.POVS_CUDA_CUDA_ARCH) +
                        (["-g", "-G", "-O0"] if build_options.POVS_CUDA_DEBUG_MODE else ["-O3", "--use_fast_math"]),
            },
            runtime_library_dirs=[
                *torch.utils.cpp_extension.library_paths(),
            ],
        ),
    ],
)
