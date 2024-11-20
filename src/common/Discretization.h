#ifndef DISCRETIZATION_H
#define DISCRETIZATION_H
#include <map>
#include <memory>
#include <string>
#include <functional>
#include <vector>
#include <Discretizer.h>
#include <BinDisc.h>
#include <fimdlp/CPPFImdlp.h>
namespace platform {
    class Discretization {
    public:
        Discretization(Discretization&) = delete;
        void operator=(const Discretization&) = delete;
        // Idea from: https://www.codeproject.com/Articles/567242/AplusC-2b-2bplusObjectplusFactory
        static Discretization* instance();
        std::shared_ptr<mdlp::Discretizer> create(const std::string& name);
        void registerFactoryFunction(const std::string& name,
            function<mdlp::Discretizer* (void)> classFactoryFunction);
        std::vector<string> getNames();
        std::string toString();
    private:
        map<std::string, function<mdlp::Discretizer* (void)>> functionRegistry;
        static Discretization* factory; //singleton
        Discretization() {};
    };
    class RegistrarDiscretization {
    public:
        RegistrarDiscretization(const std::string& className, function<mdlp::Discretizer* (void)> classFactoryFunction);
    };
}
#endif