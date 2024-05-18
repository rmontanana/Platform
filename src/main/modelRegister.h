#ifndef MODELREGISTER_H
#define MODELREGISTER_H

static platform::Registrar registrarT("TAN",
    [](void) -> bayesnet::BaseClassifier* { return new bayesnet::TAN();});
static platform::Registrar registrarTLD("TANLd",
    [](void) -> bayesnet::BaseClassifier* { return new bayesnet::TANLd();});
static platform::Registrar registrarS("SPODE",
    [](void) -> bayesnet::BaseClassifier* { return new bayesnet::SPODE(2);});
static platform::Registrar registrarSn("SPnDE",
    [](void) -> bayesnet::BaseClassifier* { return new bayesnet::SPnDE({ 0, 1 });});
static platform::Registrar registrarSLD("SPODELd",
    [](void) -> bayesnet::BaseClassifier* { return new bayesnet::SPODELd(2);});
static platform::Registrar registrarK("KDB",
    [](void) -> bayesnet::BaseClassifier* { return new bayesnet::KDB(2);});
static platform::Registrar registrarKLD("KDBLd",
    [](void) -> bayesnet::BaseClassifier* { return new bayesnet::KDBLd(2);});
static platform::Registrar registrarA("AODE",
    [](void) -> bayesnet::BaseClassifier* { return new bayesnet::AODE();});
static platform::Registrar registrarA2("A2DE",
    [](void) -> bayesnet::BaseClassifier* { return new bayesnet::A2DE();});
static platform::Registrar registrarALD("AODELd",
    [](void) -> bayesnet::BaseClassifier* { return new bayesnet::AODELd();});
static platform::Registrar registrarBA("BoostAODE",
    [](void) -> bayesnet::BaseClassifier* { return new bayesnet::BoostAODE();});
static platform::Registrar registrarBA2("BoostA2DE",
    [](void) -> bayesnet::BaseClassifier* { return new bayesnet::BoostA2DE();});
static platform::Registrar registrarSt("STree",
    [](void) -> bayesnet::BaseClassifier* { return new pywrap::STree();});
static platform::Registrar registrarOdte("Odte",
    [](void) -> bayesnet::BaseClassifier* { return new pywrap::ODTE();});
static platform::Registrar registrarSvc("SVC",
    [](void) -> bayesnet::BaseClassifier* { return new pywrap::SVC();});
static platform::Registrar registrarRaF("RandomForest",
    [](void) -> bayesnet::BaseClassifier* { return new pywrap::RandomForest();});
static platform::Registrar registrarXGB("XGBoost",
    [](void) -> bayesnet::BaseClassifier* { return new pywrap::XGBoost();});

#endif