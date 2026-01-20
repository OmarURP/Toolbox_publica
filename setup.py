from setuptools import setup, find_packages

setup(
    name="ImtaTURB",
    version="0.1.0",
    description="Librería de operaciones para análisis de turbulencia",
    author="Tu Nombre",  # Cambia esto
    author_email="tu.email@example.com",  # Cambia esto
    packages=find_packages(),
    install_requires=[
        "numpy",
        "pandas",
        "matplotlib",
        "scipy",
        # Agrega aquí otras dependencias que necesite tu módulo
    ],
    python_requires=">=3.6",
)
