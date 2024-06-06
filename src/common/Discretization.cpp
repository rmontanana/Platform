#include "Discretization.h"

namespace platform {
    // Idea from: https://www.codeproject.com/Articles/567242/AplusC-2b-2bplusObjectplusFactory
    Discretization* Discretization::factory = nullptr;
    Discretization* Discretization::instance()
    {
        //manages singleton
        if (factory == nullptr)
            factory = new Discretization();
        return factory;
    }
    void Discretization::registerFactoryFunction(const std::string& name,
        function<mdlp::Discretizer* (void)> classFactoryFunction)
    {
        // register the class factory function
        functionRegistry[name] = classFactoryFunction;
    }
    std::shared_ptr<mdlp::Discretizer> Discretization::create(const std::string& name)
    {
        mdlp::Discretizer* instance = nullptr;

        // find name in the registry and call factory method.
        auto it = functionRegistry.find(name);
        if (it != functionRegistry.end())
            instance = it->second();
        // wrap instance in a shared ptr and return
        if (instance != nullptr)
            return std::unique_ptr<mdlp::Discretizer>(instance);
        else
            throw std::runtime_error("Discretizer not found: " + name);
    }
    std::vector<std::string> Discretization::getNames()
    {
        std::vector<std::string> names;
        transform(functionRegistry.begin(), functionRegistry.end(), back_inserter(names),
            [](const pair<std::string, function<mdlp::Discretizer* (void)>>& pair) { return pair.first; });
        return names;
    }
    std::string Discretization::toString()
    {
        std::string result = "";
        std::string sep = "";
        for (const auto& pair : functionRegistry) {
            result += sep + pair.first;
            sep = ", ";
        }
        return "{" + result + "}";
    }
    RegistrarDiscretization::RegistrarDiscretization(const std::string& name, function<mdlp::Discretizer* (void)> classFactoryFunction)
    {
        // register the class factory function 
        Discretization::instance()->registerFactoryFunction(name, classFactoryFunction);
    }
}