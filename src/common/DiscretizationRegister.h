#ifndef DISCRETIZATIONREGISTER_H
#define DISCRETIZATIONREGISTER_H
#include <common/Discretization.h>
static platform::RegistrarDiscretization registrarM("mdlp",
    [](void) -> mdlp::Discretizer* { return new mdlp::CPPFImdlp();});
static platform::RegistrarDiscretization registrarBU3("bin3u",
    [](void) -> mdlp::Discretizer* { return new mdlp::BinDisc(3, mdlp::strategy_t::UNIFORM);});
static platform::RegistrarDiscretization registrarBQ3("bin3q",
    [](void) -> mdlp::Discretizer* { return new mdlp::BinDisc(3, mdlp::strategy_t::QUANTILE);});
static platform::RegistrarDiscretization registrarBU4("bin4u",
    [](void) -> mdlp::Discretizer* { return new mdlp::BinDisc(4, mdlp::strategy_t::UNIFORM);});
static platform::RegistrarDiscretization registrarBQ4("bin4q",
    [](void) -> mdlp::Discretizer* { return new mdlp::BinDisc(4, mdlp::strategy_t::QUANTILE);});
#endif