from setuptools import setup, find_packages

install_requirements = [
    "setuptools",
    "numpy",
    "tensorflow",
    "sympy",
    "matplotlib",
    "strawberryfields>=0.15",
    "pennylane",
    "pennylane-sf",
    "lightgbm",
    "openml",
    "pandas",
    "scikit-learn",
    "tqdm",
    "gym",
    "box2d-py",
    "imageio"
]
setup(name='prevision-quantum-nn',
      version='1.0.1',
      description='Prevision Automating Quantum Neural Networks Applications',
      author='prevision.io',
      author_email='prevision@prevision.io',
      license='MIT',
      packages=find_packages(),
      install_requires=install_requirements,
      zip_safe=False,
      python_requires = ">=3.6.8"
)
