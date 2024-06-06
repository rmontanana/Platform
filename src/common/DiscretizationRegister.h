#ifndef DISCRETIZATIONREGISTER_H
#define DISCRETIZATIONREGISTER_H
#include <common/Discretization.h>
static platform::RegistrarDiscretization registrarM("mdlp",
    [](void) -> mdlp::Discretizer* { return new mdlp::CPPFImdlp();});
static platform::RegistrarDiscretization registrarBU("BinUniform",
    [](void) -> mdlp::Discretizer* { return new mdlp::BinDisc(3, mdlp::strategy_t::UNIFORM);});
static platform::RegistrarDiscretization registrarBQ("BinQuantile",
    [](void) -> mdlp::Discretizer* { return new mdlp::BinDisc(3, mdlp::strategy_t::QUANTILE);});
#endif