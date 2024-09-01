from setuptools import find_packages, setup

setup(
    name='my_ml_project',
    version='0.1.0',
    description='An implementation from scratch of decision trees',
    author='Fadi Benzaima',
    packages=find_packages(),
    install_requires=[
        'pandas',
        'numpy',
        'scikit-learn',
        'matplotlib',
        'seaborn',
        'jupyterlab',
    ],
)