from distutils.core import setup

setup(
    name='kztk',
    version='0.1',
    description='Pre-trained models for out-of-the-box deep learning.',
    long_description=open('README.md').read(),
    website='https://github.com/k15z/kztk',
    author_email='kevz@mit.edu',
    keywords='nlp deep learning',
    license='MIT License',
    packages=['kztk','kztk.toxic','kztk.plagiarist'],
    install_requires=[
        'nltk'
    ],
    include_package_data=True
)
