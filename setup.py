
from setuptools import setup

setup(name='irl_methods',
      version='0.1.0',
      description='High-quality implementations of various algorithms for '
                  'Inverse Reinforcement Learning',
      url='https://github.com/aaronsnoswell/irl_methods',
      author='Aaron Snoswell',
      author_email='aaron.snoswell@uqconnect.edu.au',
      license='MIT',
      packages=[
            'irl_methods',
            'irl_methods.utils',
            'irl_methods.thirdparty',
            'irl_methods.mdp'
      ],
      install_requires=['numpy', 'cvxopt', 'gym'],
      zip_safe=False)
