#ifndef MODELS_H
#define MODELS_H
#include <map>
#include "BaseClassifier.h"
#include "AODE.h"
#include "TAN.h"
#include "KDB.h"
#include "SPODE.h"
#include "TANLd.h"
#include "KDBLd.h"
#include "SPODELd.h"
#include "AODELd.h"
#include "BoostAODE.h"
#include "STree.h"
#include "ODTE.h"
#include "SVC.h"
#include "RandomForest.h"
namespace platform {
    class Models {
    private:
        map<std::string, function<bayesnet::BaseClassifier* (void)>> functionRegistry;
        static Models* factory; //singleton
        Models() {};
    public:
        Models(Models&) = delete;
        void operator=(const Models&) = delete;
        // Idea from: https://www.codeproject.com/Articles/567242/AplusC-2b-2bplusObjectplusFactory
        static Models* instance();
        shared_ptr<bayesnet::BaseClassifier> create(const std::string& name);
        void registerFactoryFunction(const std::string& name,
            function<bayesnet::BaseClassifier* (void)> classFactoryFunction);
        std::vector<string> getNames();
        std::string tostring();

    };
    class Registrar {
    public:
        Registrar(const std::string& className, function<bayesnet::BaseClassifier* (void)> classFactoryFunction);
    };
}
#endif