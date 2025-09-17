import io
import os
import setuptools
import re


def get_version():
    version_file = os.path.join(os.path.dirname(__file__), "souko", "__init__.py")
    with open(version_file, "r", encoding="utf-8") as f:
        content = f.read()
    match = re.search(r'^__version__\s*=\s*[\'"]([^\'"]*)[\'"]', content, re.MULTILINE)
    if match:
        return match.group(1)
    raise RuntimeError("Cannot find __version__ in souko/__init__.py")


def read(*paths, **kwargs):
    """Read the contents of a text file safely.
    >>> read("project_name", "VERSION")
    '0.1.0'
    >>> read("README.md")
    ...
    """

    content = ""
    with io.open(
        os.path.join(os.path.dirname(__file__), *paths),
        encoding=kwargs.get("encoding", "utf8"),
    ) as open_file:
        content = open_file.read().strip()
    return content


def read_requirements(path):
    return [
        line.strip()
        for line in read(path).split("\n")
        if not line.startswith(('"', "#", "-", "git+"))
    ]


setuptools.setup(
    name="souko",
    version=get_version(),
    author="Simon Kojima",
    description="Python Library for Managing EEG Data",
    long_description=read("README.md"),
    long_description_content_type="text/markdown",
    url="https://github.com/simonkojima/souko",
    packages=setuptools.find_packages(),
    install_requires=["numpy", "mne>=1.8", "moabb"],
    license="MIT",
)
