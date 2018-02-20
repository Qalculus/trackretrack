from setuptools import setup

setup(
    name='trackretrack',
    version=0.9,
    description='Track and retrack image features in videos',
    author='Hannes Ovrén',
    author_email='hannes.ovren@liu.se',
    license='MIT',
    py_modules=['trackretrack', 'anms'],

    entry_points={
        'console_scripts': [
            'trackretrack=trackretrack'
        ]
    }
)
