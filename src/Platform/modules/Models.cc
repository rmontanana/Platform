#include "Models.h"
namespace platform {
    // Idea from: https://www.codeproject.com/Articles/567242/AplusC-2b-2bplusObjectplusFactory
    Models* Models::factory = nullptr;;
    Models* Models::instance()
    {
        //manages singleton
        if (factory == nullptr)
            factory = new Models();
        return factory;
    }
    void Models::registerFactoryFunction(const std::string& name,
        function<bayesnet::BaseClassifier* (void)> classFactoryFunction)
    {
        // register the class factory function
        functionRegistry[name] = classFactoryFunction;
    }
    shared_ptr<bayesnet::BaseClassifier> Models::create(const std::string& name)
    {
        bayesnet::BaseClassifier* instance = nullptr;

        // find name in the registry and call factory method.
        auto it = functionRegistry.find(name);
        if (it != functionRegistry.end())
            instance = it->second();
        // wrap instance in a shared ptr and return
        if (instance != nullptr)
            return unique_ptr<bayesnet::BaseClassifier>(instance);
        else
            return nullptr;
    }
    std::vector<std::string> Models::getNames()
    {
        std::vector<std::string> names;
        transform(functionRegistry.begin(), functionRegistry.end(), back_inserter(names),
            [](const pair<std::string, function<bayesnet::BaseClassifier* (void)>>& pair) { return pair.first; });
        return names;
    }
    std::string Models::tostring()
    {
        std::string result = "";
        for (const auto& pair : functionRegistry) {
            result += pair.first + ", ";
        }
        return "{" + result.substr(0, result.size() - 2) + "}";
    }
    Registrar::Registrar(const std::string& name, function<bayesnet::BaseClassifier* (void)> classFactoryFunction)
    {
        // register the class factory function 
        Models::instance()->registerFactoryFunction(name, classFactoryFunction);
    }
}