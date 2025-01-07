from setuptools import setup, find_packages

setup(
    name="bike_share_pred",
    version="0.0.1",
    author="Prathmesh",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
)