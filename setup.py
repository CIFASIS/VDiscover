#!/usr/bin/python2
from setuptools import setup

setup(
    name='VDiscover',
    version='0.1',
    packages=['vdiscover'],
    include_package_data=True,
    license='GPL3',
    description='A tool to predict the existence of vulnerabilities in testcases',
    long_description="",
    url='http://vdiscover.org/',
    author='G.Grieco',
    author_email='gg@cifasis-conicet.gov.ar',
    scripts=[
        'fextractor',
        'vpredictor',
        'tcreator',
        'tseeder',
        'vd'],
    install_requires=[
        "python-ptrace",
        "scikit-learn"],
)
