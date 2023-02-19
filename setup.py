import setuptools

with open('requirements.txt') as f:
    required = f.read().splitlines()

setuptools.setup(
    name="MFNLC-MA-Drone",
    version="0.0.1",
    install_requires=required,
    packages=setuptools.find_packages()
)
