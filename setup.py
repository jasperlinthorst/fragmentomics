
from setuptools import setup, find_packages

setup(
    name='cfstats', author="Jasper Linthorst", author_email="jasper.linthorst@gmail.com",
    version='0.1',
    packages=find_packages(),
    install_requires = ["numpy","scikit-learn","pandas","biopython","matplotlib","seaborn","scipy","scanpy","anndata","gffutils"],#,"glmnet"],
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

