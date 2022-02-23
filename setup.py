from setuptools import setup, find_packages

if __name__ == "__main__":
    setup(
        name='cpm_kernels',
        version='1.0.10',
        packages=find_packages(),
        description='CPM CUDA kernels',
        long_description=open("./README.md", 'r').read(),
        long_description_content_type="text/markdown",
        keywords="CPM, cuda, AI",
        classifiers=[
            "Development Status :: 3 - Alpha",
            "Environment :: GPU :: NVIDIA CUDA :: 10.1",
            "Environment :: GPU :: NVIDIA CUDA :: 10.2",
            "Environment :: GPU :: NVIDIA CUDA :: 11.0",
            "Environment :: GPU :: NVIDIA CUDA :: 11.1",
            "Environment :: GPU :: NVIDIA CUDA :: 11.2",
            "Environment :: GPU :: NVIDIA CUDA :: 11.3",
            "Environment :: GPU :: NVIDIA CUDA :: 11.4",
            "Environment :: GPU :: NVIDIA CUDA :: 11.5",
            "Intended Audience :: Developers",
            "Intended Audience :: Science/Research",
            "License :: OSI Approved :: Apache Software License",
            "Natural Language :: English",
            "Operating System :: OS Independent",
            "Programming Language :: Python :: 3",
            "Topic :: Scientific/Engineering :: Artificial Intelligence",
            "Topic :: Software Development :: Libraries :: Python Modules",

        ],
        license='Apache 2.0',
        include_package_data=True,
        package_data={
            'cpm_kernels.kernels': ['cuda/*.fatbin']
        }
    )
