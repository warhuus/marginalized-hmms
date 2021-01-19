from setuptools import setup

setup(
      name='mhmm',
      version='1.0'
      author='Christopher Warhuus',
      author_email='christopher.warhuus@gmail.com',
      packages=setuptools.find_packages(),
      install_requires=['torch',
                        'torchvision',
                        'numpy',
                        'matplotlib']
      )
