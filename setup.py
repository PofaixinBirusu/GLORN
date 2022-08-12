from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension


setup(
    name='glorn',
    version='1.0.0',
    ext_modules=[
        CUDAExtension(
            name='glorn.ext',
            sources=[
                'glorn/extensions/extra/cloud/cloud.cpp',
                'glorn/extensions/cpu/grid_subsampling/grid_subsampling.cpp',
                'glorn/extensions/cpu/grid_subsampling/grid_subsampling_cpu.cpp',
                'glorn/extensions/cpu/radius_neighbors/radius_neighbors.cpp',
                'glorn/extensions/cpu/radius_neighbors/radius_neighbors_cpu.cpp',
                'glorn/extensions/pybind.cpp',
            ],
        ),
    ],
    cmdclass={'build_ext': BuildExtension},
)
