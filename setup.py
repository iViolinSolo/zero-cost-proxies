'''
Author: ViolinSolo
Date: 2023-04-07 16:02:54
LastEditTime: 2023-04-07 19:03:59
LastEditors: ViolinSolo
Description: package setup file.
FilePath: /zero-cost-proxies/setup.py
'''

# Note: To use the 'upload' functionality of this file, you must:
#   $ pip install twine

import io
import os
import sys
from shutil import rmtree

from setuptools import setup, find_packages, Command


# Package meta-data.
NAME = 'alethiometer'
DESCRIPTION = 'ZC proxies calculation repo, altered from foresight package.'
URL = 'https://github.com/iViolinSolo/zero-cost-proxies'
EMAIL = 'i.violinsolo@gmail.com'
AUTHOR = 'ViolinSolo'
REQUIRES_PYTHON = '>=3.6.0'  # ">=3.0, !=3.0.*, !=3.1.*, !=3.2.*, !=3.3*",  # Python版本依赖
VERSION = None


try:
    import torch
    import torchvision
except ImportError:
    print('PyTorch not found! please install torch/torchvision before proceeding to install the alethiometer package.')
    exit(1)


# # 导入静态文件
# file_data = [
#     # ("sci-util/static", ["sci-util/static/icon.svg", "sci-util/static/config.json"]),
# ]

# 第三方依赖
# What packages are required for this module to be executed?
REQUIRED = [
    # 'git-python',
    'h5py>=2.10.0',
    # 'jupyter>=1.0.0',
    # 'matplotlib>=3.2.1',
    # 'nas-bench-201==2.0',
    'numpy>=1.18.4',
    # 'prettytable>=2.0.0',
    # 'pytorch-ignite>=0.3.0',
    # 'pytorchcv>=0.0.58',
    # 'scikit-learn>=0.23.2',
    # 'scipy>=1.4.1',
    # 'tqdm>=4.46.0'
    'Pillow>=4',
]

# What packages are optional?
EXTRAS = {
    # 'fancy feature': ['django'],
}

# The rest you shouldn't have to touch too much :)
# ------------------------------------------------
# Except, perhaps the License and Trove Classifiers!
# If you do change the License, remember to change the Trove Classifier for that!

here = os.path.abspath(os.path.dirname(__file__))

# Import the README and use it as the long-description.
# Note: this will only work if 'README.md' is present in your MANIFEST.in file!
try:
    with io.open(os.path.join(here, 'README.md'), encoding='utf-8') as f:
        long_description = '\n' + f.read()
except FileNotFoundError:
    long_description = DESCRIPTION

# Load the package's __version__.py module as a dictionary.
about = {}
if not VERSION:
    project_slug = NAME.lower().replace("-", "_").replace(" ", "_")
    with open(os.path.join(here, project_slug, '__version__.py')) as f:
        exec(f.read(), about)
else:
    about['__version__'] = VERSION


class UploadCommand(Command):
    """Support setup.py upload."""

    description = 'Build and publish the package.'
    user_options = []

    @staticmethod
    def status(s):
        """Prints things in bold."""
        print('\033[1m{0}\033[0m'.format(s))

    def initialize_options(self):
        pass

    def finalize_options(self):
        pass

    def run(self):
        try:
            self.status('Removing previous builds…')
            rmtree(os.path.join(here, 'dist'))
        except OSError:
            pass

        self.status('Building Source and Wheel (universal) distribution…')
        os.system('{0} setup.py sdist bdist_wheel --universal'.format(sys.executable))

        self.status('Uploading the package to PyPI via Twine…')
        os.system('twine upload dist/*')

        self.status('Pushing git tags…')
        os.system('git tag v{0}'.format(about['__version__']))
        os.system('git push --tags')

        sys.exit()


# Where the magic happens:
setup(
    name=NAME,  # 包名称
    version=about['__version__'],  # 包版本
    description=DESCRIPTION,  # 包详细描述
    long_description=long_description,  # 长描述，通常是readme，打包到PiPy需要
    long_description_content_type='text/markdown',
    author=AUTHOR,  # 作者名称
    author_email=EMAIL,  # 作者邮箱
    python_requires=REQUIRES_PYTHON,  # Python版本依赖
    url=URL,  # 项目官网
    packages=find_packages(exclude=["tests", "*.tests", "*.tests.*", "tests.*"]),  # 项目需要的包
    # If your package is a single module, use this instead of 'packages':
    # py_modules=['mypackage'],
    # entry_points={
    #     'console_scripts': ['mycli=mymodule:cli'],
    # },
    install_requires=REQUIRED,  # 第三方库依赖
    extras_require=EXTRAS,
    # data_files=file_data,  # 打包时需要打包的数据文件，如图片，配置文件等
    include_package_data=True,  # 是否需要导入静态数据文件
    license='Apache Software License',
    classifiers=[  # 程序的所属分类列表
        # Trove classifiers
        # Full list: https://pypi.python.org/pypi?%3Aaction=list_classifiers
        'License :: OSI Approved :: Apache Software License',
        # 'Development Status :: 5 - Production/Stable',
        # 'Intended Audience :: Developers',
        # 'Natural Language :: English',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3',
        # 'Programming Language :: Python :: 3.4',
        # 'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        # 'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: Implementation :: CPython',
        'Programming Language :: Python :: Implementation :: PyPy'
    ],
    # $ setup.py publish support.
    cmdclass={
        'upload': UploadCommand,
    },
    # zip_safe=False,  # 此项需要，否则卸载时报windows error

)



# # 可以先验证setup.py的正确性
# python3 setup.py check
# # 如果没有问题，就可以使用下方的命令正式打包
# python3 setup.py sdist
# # --wheel-dir: 为打包存储的路径
# # 空格后为需要打包的工程路径
# pip3 wheel --wheel-dir=D:\\work\\base_package\\dist D:\\work\\base_package
# pip3 wheel --wheel-dir=./dist/ .

# # 注册包
# twine register dist/smart.whl
# # 上传包  enter your pypi account with password
# twine upload dist/*
# twine upload dist/sci_util-1.2.0-py3-none-any.whl

# 最新的做法就是直接upload
# python3 setup.py upload