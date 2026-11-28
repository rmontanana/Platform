#!/usr/bin/env python3
"""Print the Conan conf that pins CMake's FindPython to one interpreter.

Usage: python_conf.py <python>

FindPython takes the headers from the interpreter given in Python3_EXECUTABLE,
but it looks for libpython in CMAKE_PREFIX_PATH first. An active conda
environment exports that variable, so a libpython of the same minor version
living there would be linked instead of the interpreter's own. Naming the
library and the include directory explicitly in the Conan toolchain removes the
search. Prints nothing when the interpreter has no regular libpython file (a
macOS framework build, for instance), leaving FindPython to its defaults.
"""
import json
import subprocess
import sys

FACTS = (
    "import json, os, sysconfig;"
    "print(json.dumps([os.path.join(sysconfig.get_config_var('LIBDIR') or '',"
    " sysconfig.get_config_var('LDLIBRARY') or ''), sysconfig.get_path('include')]))"
)


def main():
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    import os
    library, include = json.loads(subprocess.check_output([sys.argv[1], "-c", FACTS],
                                                         universal_newlines=True))
    if not os.path.isfile(library) or not os.path.isfile(os.path.join(include, "Python.h")):
        return
    for value in (library, include):
        if any(char in value for char in "'\"\\\n"):
            sys.exit("Unsupported Python path: %r" % value)
    print("tools.cmake.cmaketoolchain:extra_variables="
          "{'Python3_LIBRARY': '%s', 'Python3_INCLUDE_DIR': '%s'}" % (library, include))


if __name__ == "__main__":
    main()
