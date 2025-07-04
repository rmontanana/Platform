from conan import ConanFile
from conan.tools.cmake import CMakeToolchain, CMakeDeps, cmake_layout


class PlatformConan(ConanFile):
    name = "platform"
    version = "1.1.0"
    
    # Binary configuration
    settings = "os", "compiler", "build_type", "arch"
    
    # Sources are located in the same place as this recipe, copy them to the recipe
    exports_sources = "CMakeLists.txt", "src/*", "tests/*", "config/*", "cmake/*"
    
    def requirements(self):
        # Core dependencies from vcpkg.json
        self.requires("argparse/3.2")
        self.requires("libtorch/2.7.0")
        self.requires("nlohmann_json/3.11.3")
        self.requires("folding/1.1.1")
        self.requires("fimdlp/2.1.0")
        self.requires("arff-files/1.2.0")
        self.requires("bayesnet/1.2.0")
        self.requires("pyclassifiers/1.0.3")
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
        tc.generate()
    
    def configure(self):
        # C++20 requirement
        self.settings.compiler.cppstd = "20"
