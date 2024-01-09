# Platform

Platform to run Bayesian Networks and Machine Learning Classifiers experiments.

# Platform

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Platform to run Bayesian Networks and Machine Learning Classifiers experiments.

## 0. Setup

Before compiling Platform.

### Miniconda

To be able to run Python Classifiers such as STree, ODTE, SVC, etc. it is needed to install Miniconda. To do so, download the installer from [Miniconda](https://docs.conda.io/en/latest/miniconda.html) and run it. It is recommended to install it in the home folder.

In Linux sometimes the library libstdc++ is mistaken from the miniconda installation and produces the next message when running the b_xxxx executables:

```bash
libstdc++.so.6: version `GLIBCXX_3.4.32' not found (required by b_xxxx)
```

The solution is to erase the libstdc++ library from the miniconda installation:

### MPI

In Linux just install openmpi & openmpi-devel packages. Only if cmake can't find openmpi installation (like in Oracle Linux) set the following variable:

```bash
export MPI_HOME="/usr/lib64/openmpi"
```

In Mac OS X, install mpich with brew and if cmake doesn't find it, edit mpicxx wrapper to remove the ",-commons,use_dylibs" from final_ldflags

```bash
vi /opt/homebrew/bin/mpicx
```

### boost library

[Getting Started](<https://www.boost.org/doc/libs/1_83_0/more/getting_started/index.html>)

The best option is install the packages that the Linux distribution have in its repository. If this is the case:

```bash
sudo dnf install boost-devel
```

If this is not possible and the compressed packaged is installed, the following environment variable has to be set pointing to the folder where it was unzipped to:

```bash
export BOOST_ROOT=/path/to/library/
```

In some cases, it is needed to build the library, to do so:

```bash
cd /path/to/library
mkdir own
./bootstrap.sh --prefix=/path/to/library/own
./b2 install
export BOOST_ROOT=/path/to/library/own/
```

Don't forget to add the export BOOST_ROOT statement to .bashrc or wherever it is meant to be.

### libxlswriter

```bash
cd lib/libxlsxwriter
make
make install DESTDIR=/home/rmontanana/Code PREFIX=
```

### Release

```bash
make release
```

### Debug & Tests

```bash
make debug
```

## 1. Introduction
