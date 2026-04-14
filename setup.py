
from setuptools import setup, find_packages

#install_requires = ["numpy","scikit-learn","pandas","biopython","matplotlib","seaborn","scipy","scanpy","anndata","gffutils","pysam","pyglmnet"],

setup(
    name='cfstats', author="Jasper Linthorst", author_email="jasper.linthorst@gmail.com",
    version='0.1',
    packages=find_packages(),
    package_data={'cfstats': ['models/*.joblib', 'models/*.pickle']},
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
        "python-glmnet==2.2.2.post2",
        "statsmodels==0.14.6",
        "clickhouse-connect==0.13.0",
        "dash==4.0.0",
        "plotly==6.5.2",
    ],
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

