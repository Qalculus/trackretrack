from setuptools import setup

setup(
    name='trackretrack',
    version=0.9,
    description='Track and retrack image features in videos',
    author='Hannes Ovrén',
    author_email='hannes.ovren@liu.se',
    license='MIT',
    packages=['trackretrack'],

    entry_points={
        'console_scripts': [
            'trackretrack=trackretrack'
        ]
    }
)
