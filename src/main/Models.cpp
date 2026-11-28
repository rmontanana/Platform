#include <cstdlib>
#include <filesystem>
#include "config_platform.h"
#include "Models.h"
namespace platform {
    // Idea from: https://www.codeproject.com/Articles/567242/AplusC-2b-2bplusObjectplusFactory
    Models* Models::factory = nullptr;
    // pyclassifiers embeds the interpreter named by PYCLASSIFIERS_PYTHON and, without it,
    // CPython's default one, which does not have the Python classifiers installed.
    // Unless the user selects another, use the interpreter Platform was built with,
    // so that the installed executables need no environment to run.
    static void selectPython()
    {
        const std::string python{ platform_python_executable };
        std::error_code error;
        if (!python.empty() && std::filesystem::exists(python, error))
            setenv("PYCLASSIFIERS_PYTHON", python.c_str(), 0);
    }
    Models* Models::instance()
    {
        //manages singleton
        if (factory == nullptr) {
            selectPython();
            factory = new Models();
        }
        return factory;
    }
    void Models::registerFactoryFunction(const std::string& name,
        std::function<bayesnet::BaseClassifier* (void)> classFactoryFunction)
    {
        // register the class factory function
        functionRegistry[name] = classFactoryFunction;
    }
    bayesnet::Smoothing_t Models::smoothing(const bayesnet::BaseClassifier& clf, bayesnet::Smoothing_t requested)
    {
        return dynamic_cast<const pywrap::PyClassifier*>(&clf) != nullptr ? bayesnet::Smoothing_t::NONE : requested;
    }
    std::shared_ptr<bayesnet::BaseClassifier> Models::create(const std::string& name)
    {
        bayesnet::BaseClassifier* instance = nullptr;

        // find name in the registry and call factory method.
        auto it = functionRegistry.find(name);
        if (it != functionRegistry.end())
            instance = it->second();
        // wrap instance in a shared ptr and return
        if (instance != nullptr)
            return std::unique_ptr<bayesnet::BaseClassifier>(instance);
        else
            throw std::runtime_error("Model not found: " + name);
    }
    std::vector<std::string> Models::getNames()
    {
        std::vector<std::string> names;
        std::transform(functionRegistry.begin(), functionRegistry.end(), std::back_inserter(names),
            [](const std::pair<const std::string, std::function<bayesnet::BaseClassifier* (void)>>& pair) { return pair.first; });
        return names;
    }
    std::string Models::toString()
    {
        std::string result = "";
        std::string sep = "";
        for (const auto& pair : functionRegistry) {
            result += sep + pair.first;
            sep = ", ";
        }
        return "{" + result + "}";
    }
    Registrar::Registrar(const std::string& name, std::function<bayesnet::BaseClassifier* (void)> classFactoryFunction)
    {
        // register the class factory function 
        Models::instance()->registerFactoryFunction(name, classFactoryFunction);
    }
}