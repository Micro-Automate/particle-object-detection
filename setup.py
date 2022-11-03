from setuptools import setup

setup(
    name='miso',
    version='1.0.0',
    packages=['miso',
              'miso.object_detection.dataset',
              'miso.object_detection.dataset.cvat',
              'miso.object_detection.engine',
              'miso.shared'],
    url='',
    license='',
    author='Ross Marchant',
    author_email='ross.g.marchant@gmail.com',
    description='',
    install_requires=[
        'lxml',
        'pycocotools',
        'click',
        'scipy',
        'tqdm',
        'scikit-image'
    ]
)
