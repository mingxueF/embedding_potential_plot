import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="embedding_potential-mingxueF", # Replace with your own username
    version="0.0.1",
    author="minguxe",
    author_email="fmingxue@gmail.com",
    description="A small embedding_potential_plot package",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/mingxueF/embedding_potential_plot/tree/master",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: unige License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3',
)
