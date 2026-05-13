import os
import sys

from setuptools import setup, find_packages, Extension


def _impute_extension():
    try:
        import numpy
    except ImportError:  # pragma: no cover - build time
        raise RuntimeError(
            "numpy must be installed before building the cfstats C extension")

    use_openmp = os.environ.get("OPENMP", "1") != "0"
    extra_compile_args = []
    extra_link_args = []
    if use_openmp:
        if sys.platform == "darwin":
            # Apple clang needs libomp (`brew install libomp`) and the
            # -Xpreprocessor flag to enable OpenMP.
            extra_compile_args += ["-Xpreprocessor", "-fopenmp"]
            extra_link_args += ["-lomp"]
            # Help the compiler find libomp installed via Homebrew.
            for prefix in ("/opt/homebrew/opt/libomp", "/usr/local/opt/libomp"):
                if os.path.isdir(prefix):
                    extra_compile_args += [f"-I{prefix}/include"]
                    extra_link_args += [f"-L{prefix}/lib"]
                    break
        else:
            extra_compile_args += ["-fopenmp"]
            extra_link_args += ["-fopenmp"]

    return Extension(
        "cfstats.impute._hmm",
        sources=["cfstats/impute/_hmm.c"],
        include_dirs=[numpy.get_include()],
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args,
    )


setup(
    name='cfstats', author="Jasper Linthorst", author_email="jasper.linthorst@gmail.com",
    version='0.1',
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        "numpy==2.0.0",
        "scikit-learn==1.7.2",
        "pandas==3.0.1",
        "biopython==1.86",
        "matplotlib==3.10.8",
        "seaborn==0.13.2",
        "scipy==1.17.1",
        "pysam==0.23.3",
        "joblib==1.5.3",
        "gffutils==0.13",
        "python-glmnet>=2.6.1",
        "statsmodels==0.14.6",
        "clickhouse-connect==0.13.0",
        "dash==4.0.0",
        "plotly==6.5.2",
        "huggingface_hub>=0.20.0",
    ],
    ext_modules=[_impute_extension()],
    entry_points={
        'console_scripts': [
            'cfstats=cfstats.__main__:main',
        ],
    },
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
    ],
    python_requires='>=3.6',
)
