from setuptools import find_packages, setup

setup(
    name='mlops_homework_server',
    packages=find_packages(),
    version='0.1.0',
    description='Homework 2 MADE 2022 MLOps',
    author='tupiznak',
    license='',
    entry_points={
        "console_scripts": [
            'run_server=mlops_homework_server.server.run:main',
            'run_req_generator=mlops_homework_server.server.request_generator:main'
        ]
    },
)
