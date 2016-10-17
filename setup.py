import os
import sys

from setuptools import find_packages, setup


def setup_package():
    src_path = os.path.dirname(os.path.abspath(sys.argv[0]))
    old_path = os.getcwd()
    os.chdir(src_path)
    sys.path.insert(0, src_path)

    needs_pytest = {'pytest', 'test', 'ptr'}.intersection(sys.argv)
    pytest_runner = ['pytest-runner'] if needs_pytest else []

    setup_requires = ['cffi>=1.6', 'limix_math>=0.3'] + pytest_runner
    install_requires = ['hcache', 'limix_math>=0.3',
                        'lim>=0.1', 'pytest']
    tests_require = install_requires

    metadata = dict(
        name='limix_qep',
        maintainer="Limix Developers",
        version='1.0.1.dev1',
        maintainer_email="horta@ebi.ac.uk",
        packages=find_packages(),
        license="MIT",
        url='https://github.com/Horta/limix-qep',
        install_requires=install_requires,
        setup_requires=setup_requires,
        tests_require=tests_require,
        zip_safe=False,
        include_package_data=True,
        cffi_modules=['liknorm_build.py:liknorm']
    )

    try:
        from distutils.command.bdist_conda import CondaDistribution
    except ImportError:
        pass
    else:
        metadata['distclass'] = CondaDistribution
        metadata['conda_buildnum'] = 1
        metadata['conda_features'] = ['mkl']

    try:
        setup(**metadata)
    finally:
        del sys.path[0]
        os.chdir(old_path)

if __name__ == '__main__':
    setup_package()
