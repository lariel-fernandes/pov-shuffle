import itertools
import json
import os
import textwrap
import typing
import warnings
from pathlib import Path
from typing import ForwardRef, NamedTuple

import cutlass_library
import torch.utils.cpp_extension
from setuptools import setup


class EnvVars(NamedTuple):
    """Environment variables for build configuration."""

    POVS_CUDA_DEBUG_MODE:                 bool                            = "false"  # Build with debug flags
    POVS_CUDA_CUDA_ARCH:                  list[str]                       = "native" # Comma-separated list of CUDA architectures (e.g.: 75,86 or native)

    POVS_CUDA_INSTANTIATIONS:             list[tuple[int, int, int, str]] = ""       # Add support for these dtype-specific parameter sets; Semicolon-separated tuples of VBlockSize,PBlockSize,InstanceSize,DType (e.g.: "2,16,1,c10::Half;4,32,2,float")
    POVS_CUDA_INSTANTIATIONS_ALL_TYPES:   list[tuple[int, int, int]]      = ""       # Add support for these parameter sets with all supported dtypes; Semicolon-separated tuples of VBlockSize,PBlockSize,InstanceSize (e.g.: "2,16,1;4,32,2")

    POVS_CUDA_CARTESIAN_INSTANCING:       bool                            = "true"   # If true, add support for all combinations of the parameters below with all supported types
    POVS_CUDA_VBLOCK_SIZES:               list[int]                       = "2"      # Comma-separated list of virtual block sizes, in number of physical blocks
    POVS_CUDA_PBLOCK_SIZES:               list[int]                       = "16"     # Comma-separated list of physical block sizes, in number of instances
    POVS_CUDA_INSTANCE_SIZES:             list[int]                       = "1"      # Comma-separated list of shuffled instance sizes



def load_env_vars() -> EnvVars:
    """Load environment variables for build configuration."""
    env_vars = EnvVars()

    for key, _type in EnvVars.__annotations__.items():
        _type = eval(_type.__forward_arg__) if isinstance(_type, ForwardRef) else _type
        val = os.environ.get(key) or EnvVars._field_defaults[key]

        if typing.get_origin(_type) == list:
            item_type = typing.get_args(_type)[0]
            if typing.get_origin(item_type) is tuple:
                if not val:
                    val = []
                else:
                    tuple_types = typing.get_args(item_type)
                    val = [
                        tuple(t(x.strip()) for t, x in zip(tuple_types, item.split(",")))
                        for item in val.split(";")
                    ]
            else:
                val = val.split(",")
                if typing.get_args(_type) == (int,):
                    val = [int(x) for x in val]

        elif _type == bool:
            if val.lower().strip() in ("1", "true", "yes", "on", "enable", "enabled", "positive", "+"):
                val = True
            elif val.lower().strip() in ("0", "false", "no", "off", "disable", "disabled", "negative", "-"):
                val = False
            else:
                raise ValueError(f"Invalid boolean value for {key}: {val}")

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
        and not file.startswith("test_")
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


_DTYPE_TO_ATEN = {
    "c10::Half": "at::kHalf",
    "int":       "at::kInt",
    "long":      "at::kLong",
    "float":     "at::kFloat",
    "double":    "at::kDouble",
}
_ALL_DTYPES = list(_DTYPE_TO_ATEN.keys())


def get_all_instances(opts: EnvVars) -> list[tuple[int, int, int, str]]:
    """Compute all (vblock, pblock, instance, dtype) combinations to instantiate and dispatch."""
    seen: set = set()
    result: list = []

    def add(item):
        if item not in seen:
            seen.add(item); result.append(item)

    for args in opts.POVS_CUDA_INSTANTIATIONS:
        add(args)

    for sizes in opts.POVS_CUDA_INSTANTIATIONS_ALL_TYPES:
        for dtype in _ALL_DTYPES:
            add((*sizes, dtype))

    if opts.POVS_CUDA_CARTESIAN_INSTANCING:
        for sizes in itertools.product(opts.POVS_CUDA_VBLOCK_SIZES, opts.POVS_CUDA_PBLOCK_SIZES, opts.POVS_CUDA_INSTANCE_SIZES):
            for dtype in _ALL_DTYPES:
                add((*sizes, dtype))

    return result


def get_generated_files(opts: EnvVars) -> list[tuple[str, str]]:
    """Get codegen files for build options."""
    instances = get_all_instances(opts)

    template_instantiations = "\n".join(
        "INSTANTIATE_POVS_CUDA(%s, %s, %s, %s)" % vals
        for vals in instances
    )

    cases = textwrap.indent(
        "\n".join(
            f"else if (vblock_val == {vblock} && pblock_val == {pblock} && instance_val == {instance} && scalar_type_val == {_DTYPE_TO_ATEN[dtype]}) "
            f"{{ constexpr int VBLOCK_SIZE = {vblock}, PBLOCK_SIZE = {pblock}, INSTANCE_SIZE = {instance}; using scalar_t = {dtype}; fn; }} \\"
            for vblock, pblock, instance, dtype in instances
        ),
        prefix=2*4*" ",  # 2 indentation levels of width 4
    )
    dispatch_header = textwrap.dedent("""
        #pragma once
        #define POVS_CUDA_DISPATCH(vblock_val, pblock_val, instance_val, scalar_type_val, fn) \\
            do {{ \\
                if (false) {{}} \\
                {cases}
                else {{ TORCH_CHECK(false, "Unsupported povs_cuda combination:", " vblock=", vblock_val, " pblock=", pblock_val, " instance=", instance_val, " scalar_type=", at::toString(scalar_type_val)); }} \\
            }} while (0)
    """.strip()).format(cases=cases)

    return [
        ("povs_cuda_template_instances.gen.inc", template_instantiations),
        ("povs_cuda_dispatch.gen.h", dispatch_header),
    ]


build_options = load_env_vars()

if not cutlass_library.__file__:
    warnings.warn("CUTLASS lib not found")

if (cuda_home := find_cuda_home()) is None:
    warnings.warn("CUDA home not found")
else:
    os.environ["CUDA_HOME"] = torch.utils.cpp_extension.CUDA_HOME = cuda_home


class BuildExtension(torch.utils.cpp_extension.BuildExtension):
    """Custom build extension command."""

    def run(self):
        gen_path = Path(self.build_temp) / "codegen"
        gen_path.mkdir(parents=True, exist_ok=True)

        for file_name, content in get_generated_files(build_options):
            (gen_path / file_name).write_text(content)

        for ext in self.extensions:
            ext.include_dirs.append(str(gen_path))

        Path("src/povs/build_params.json").write_text(json.dumps({
            "debug_mode":               build_options.POVS_CUDA_DEBUG_MODE,
            "cuda_arch":                build_options.POVS_CUDA_CUDA_ARCH,
            "vblock_sizes":             build_options.POVS_CUDA_VBLOCK_SIZES,
            "pblock_sizes":             build_options.POVS_CUDA_PBLOCK_SIZES,
            "instance_sizes":           build_options.POVS_CUDA_INSTANCE_SIZES,
            "cartesian_instancing":     build_options.POVS_CUDA_CARTESIAN_INSTANCING,
            "instantiations":           [list(t) for t in build_options.POVS_CUDA_INSTANTIATIONS] if build_options.POVS_CUDA_INSTANTIATIONS else None,
            "instantiations_all_types": [list(t) for t in build_options.POVS_CUDA_INSTANTIATIONS_ALL_TYPES] if build_options.POVS_CUDA_INSTANTIATIONS_ALL_TYPES else None,
        }))

        super().run()


setup(
    package_data={"povs": ["build_params.json"]},
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
