from setuptools import setup, find_packages,setup
from typing import List

def get_requirements()->List[str]:
    requirements_lst:List[str]=[]
    try:
        with open('requirements.txt') as f:
            # Read lines from files
            lines = f.readlines()
            #process the lines to remove whitespace and newlines
            requirements = []
            for line in lines:
                requirement=line.strip()
                if requirement and not requirement.startswith('#') and requirement != '-e .':
                    requirements_lst.append(requirement)

    except FileNotFoundError:
        print("requirements.txt file not found. Please make sure it exists in the same directory as setup.py.")
    return requirements_lst

setup(
    name='network-security',
    version='0.0.1',
    author='Aryan Rahate',
    author_email='rnrahate.ar@gmail.com',
    packages=find_packages(),
    install_requires=get_requirements()
)