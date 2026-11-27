import os
from conan import ConanFile
from conan.tools.cmake import CMakeToolchain, CMakeDeps, cmake_layout


class PlatformConan(ConanFile):
    name = "platform"
    version = "1.1.6"
    
    # Binary configuration
    settings = "os", "compiler", "build_type", "arch"
    
    # Sources are located in the same place as this recipe, copy them to the recipe
    exports_sources = "CMakeLists.txt", "src/*", "tests/*", "config/*", "cmake/*"
    
    def requirements(self):
        # Core dependencies from vcpkg.json
        self.requires("argparse/3.2")
        self.requires("libtorch/2.7.1")
        self.requires("nlohmann_json/3.11.3")
        self.requires("folding/2.0.0")
        self.requires("fimdlp/3.0.1")
        self.requires("arff-files/2.0.0")
        self.requires("bayesnet/1.5.0")
        # pyclassifiers/1.1.1 is built on bayesnet/1.5.0. Its python_abi and
        # boost_version options name the external CPython and Boost it is built
        # and used with; the Makefile detects and passes them.
        self.requires("pyclassifiers/1.1.1")
        self.requires("libxlsxwriter/1.2.2")
        
    def build_requirements(self):
        self.tool_requires("cmake/[>=3.30]")
        self.test_requires("catch2/3.8.1")
    
    def layout(self):
        cmake_layout(self)
    
    def generate(self):
        deps = CMakeDeps(self)
        deps.generate()
        tc = CMakeToolchain(self)
        # Same interpreter for pyclassifiers and for Platform.
        python = os.environ.get("PYCLASSIFIERS_PYTHON")
        if python:
            tc.variables["Python3_EXECUTABLE"] = python
        tc.generate()
    
    def configure(self):
        # C++20 requirement
        self.settings.compiler.cppstd = "20"
