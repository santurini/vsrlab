from setuptools import setup

setup(
    name="vsrlab",
    version="0.0.1",
    packages=['vsrlab.core', 'vsrlab.optical_flow', 'vsrlab.vsr'],
    package_dir={'vsrlab': 'src'},
    description='VSR playground',
    author="Arturo Ghinassi",
    author_email='ghinassiarturo8@gmail.com',
)
