#ifndef MODELREGISTER_H
#define MODELREGISTER_H
namespace platform {
    static Registrar registrarT("TAN",
        [](void) -> bayesnet::BaseClassifier* { return new bayesnet::TAN();});
    static Registrar registrarTLD("TANLd",
        [](void) -> bayesnet::BaseClassifier* { return new bayesnet::TANLd();});
    static Registrar registrarS("SPODE",
        [](void) -> bayesnet::BaseClassifier* { return new bayesnet::SPODE(2);});
    static Registrar registrarSn("SPnDE",
        [](void) -> bayesnet::BaseClassifier* { return new bayesnet::SPnDE({ 0, 1 });});
    static Registrar registrarSLD("SPODELd",
        [](void) -> bayesnet::BaseClassifier* { return new bayesnet::SPODELd(2);});
    static Registrar registrarK("KDB",
        [](void) -> bayesnet::BaseClassifier* { return new bayesnet::KDB(2);});
    static Registrar registrarKLD("KDBLd",
        [](void) -> bayesnet::BaseClassifier* { return new bayesnet::KDBLd(2);});
    static Registrar registrarA("AODE",
        [](void) -> bayesnet::BaseClassifier* { return new bayesnet::AODE();});
    static Registrar registrarA2("A2DE",
        [](void) -> bayesnet::BaseClassifier* { return new bayesnet::A2DE();});
    static Registrar registrarALD("AODELd",
        [](void) -> bayesnet::BaseClassifier* { return new bayesnet::AODELd();});
    static Registrar registrarBA("BoostAODE",
        [](void) -> bayesnet::BaseClassifier* { return new bayesnet::BoostAODE();});
    static Registrar registrarBA2("BoostA2DE",
        [](void) -> bayesnet::BaseClassifier* { return new bayesnet::BoostA2DE();});
    static Registrar registrarSt("STree",
        [](void) -> bayesnet::BaseClassifier* { return new pywrap::STree();});
    static Registrar registrarOdte("Odte",
        [](void) -> bayesnet::BaseClassifier* { return new pywrap::ODTE();});
    static Registrar registrarSvc("SVC",
        [](void) -> bayesnet::BaseClassifier* { return new pywrap::SVC();});
    static Registrar registrarRaF("RandomForest",
        [](void) -> bayesnet::BaseClassifier* { return new pywrap::RandomForest();});
    static Registrar registrarXGB("XGBoost",
        [](void) -> bayesnet::BaseClassifier* { return new pywrap::XGBoost();});
    static Registrar registrarXSPODE("XSPODE",
        [](void) -> bayesnet::BaseClassifier* { return new bayesnet::XSpode(0);});
    static Registrar registrarXSP2DE("XSP2DE",
        [](void) -> bayesnet::BaseClassifier* { return new bayesnet::XSp2de(0, 1);});
    static Registrar registrarXBAODE("XBAODE",
        [](void) -> bayesnet::BaseClassifier* { return new bayesnet::XBAODE();});
    static Registrar registrarXBA2DE("XBA2DE",
        [](void) -> bayesnet::BaseClassifier* { return new bayesnet::XBA2DE();});
    static Registrar registrarXA1DE("XA1DE",
            [](void) -> bayesnet::BaseClassifier* { return new XA1DE();});
}
#endif
