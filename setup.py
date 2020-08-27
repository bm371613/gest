from setuptools import setup

setup(
    name='gest',
    version='0.0',
    packages=['gest'],
    requirements=[
        'keras',
        'numpy',
        'opencv-python',
        'tensorflow',
    ],
    extras_require={'examples': ['pynput']},
)
