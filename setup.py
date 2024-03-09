#!/usr/bin/env python

"""The setup script."""

from setuptools import setup, find_packages

with open('README.rst') as readme_file:
    readme = readme_file.read()

with open('HISTORY.rst') as history_file:
    history = history_file.read()

requirements = ['Click>=7.0', ]

test_requirements = [ ]

setup(
    author="Salim Benhamadi",
    author_email='salim.benhamadi@outlook.com',
    python_requires='>=3.6',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Natural Language :: English',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
    ],
    description="TabularOCR is a Python library that provides an easy-to-use Optical Character Recognition (OCR) solution for extracting tables from images and PDFs. It offers flexible output options, allowing you to export the extracted data in CSV, XLSX, or other spreadsheet formats.",
    entry_points={
        'console_scripts': [
            'tabularocr=tabularocr.cli:main',
        ],
    },
    install_requires=requirements,
    license="MIT license",
    long_description=readme + '\n\n' + history,
    include_package_data=True,
    keywords='TabularOCR',
    name='TabularOCR',
    packages=find_packages(include=['TabularOCR', 'TabularOCR.*']),
    test_suite='tests',
    tests_require=test_requirements,
    url='https://github.com/salim-benhamadi/tabularocr',
    version='0.1.0',
    zip_safe=False,
)
