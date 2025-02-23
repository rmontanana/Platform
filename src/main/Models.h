#ifndef MODELS_H
#define MODELS_H
#include <map>
#include <bayesnet/BaseClassifier.h>
#include <bayesnet/ensembles/AODE.h>
#include <bayesnet/ensembles/A2DE.h>
#include <bayesnet/ensembles/AODELd.h>
#include <bayesnet/ensembles/BoostAODE.h>
#include <bayesnet/ensembles/BoostA2DE.h>
#include <bayesnet/classifiers/TAN.h>
#include <bayesnet/classifiers/KDB.h>
#include <bayesnet/classifiers/SPODE.h>
#include <bayesnet/classifiers/SPnDE.h>
#include <bayesnet/classifiers/TANLd.h>
#include <bayesnet/classifiers/KDBLd.h>
#include <bayesnet/classifiers/SPODELd.h>
#include <bayesnet/classifiers/SPODELd.h>
#include <pyclassifiers/STree.h>
#include <pyclassifiers/ODTE.h>
#include <pyclassifiers/SVC.h>
#include <pyclassifiers/XGBoost.h>
#include <pyclassifiers/RandomForest.h>
#include "../experimental_clfs/XA1DE.h"
#include "../experimental_clfs/XBAODE.h"
namespace platform {
    class Models {
    public:
        Models(Models&) = delete;
        void operator=(const Models&) = delete;
        // Idea from: https://www.codeproject.com/Articles/567242/AplusC-2b-2bplusObjectplusFactory
        static Models* instance();
        shared_ptr<bayesnet::BaseClassifier> create(const std::string& name);
        void registerFactoryFunction(const std::string& name,
            function<bayesnet::BaseClassifier* (void)> classFactoryFunction);
        std::vector<string> getNames();
        std::string toString();
    private:
        map<std::string, function<bayesnet::BaseClassifier* (void)>> functionRegistry;
        static Models* factory; //singleton
        Models() {};
    };
    class Registrar {
    public:
        Registrar(const std::string& className, function<bayesnet::BaseClassifier* (void)> classFactoryFunction);
    };
}
#endif