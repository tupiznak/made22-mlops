from setuptools import find_packages, setup

setup(
    name='mlops_homework',
    packages=find_packages(),
    version='0.1.0',
    description='Homework MADE 2022 MLOps',
    author='tupiznak',
    license='',
    entry_points={
        "console_scripts": [
            "mlops_homework_download_dataset = mlops_homework.data.load_dataset:main",
            "mlops_homework_process_dataset = mlops_homework.features.build_features:main",
            "mlops_homework_train_baseline_model = mlops_homework.models.train_baseline_model:main",
        ]
    },
)
