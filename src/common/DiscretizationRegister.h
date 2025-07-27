#ifndef DISCRETIZATIONREGISTER_H
#define DISCRETIZATIONREGISTER_H
#include <common/Discretization.h>
#include <limits>
static platform::RegistrarDiscretization registrarM("mdlp",
    [](void) -> mdlp::Discretizer* { return new mdlp::CPPFImdlp();});
static platform::RegistrarDiscretization registrarM3("mdlp3",
    [](void) -> mdlp::Discretizer* { return new mdlp::CPPFImdlp(3, numeric_limits<int>::max(), 3);});
static platform::RegistrarDiscretization registrarM4("mdlp4",
    [](void) -> mdlp::Discretizer* { return new mdlp::CPPFImdlp(3, numeric_limits<int>::max(), 4);});
static platform::RegistrarDiscretization registrarM5("mdlp5",
    [](void) -> mdlp::Discretizer* { return new mdlp::CPPFImdlp(3, numeric_limits<int>::max(), 5);});
static platform::RegistrarDiscretization registrarBU3("bin3u",
    [](void) -> mdlp::Discretizer* { return new mdlp::BinDisc(3, mdlp::strategy_t::UNIFORM);});
static platform::RegistrarDiscretization registrarBQ3("bin3q",
    [](void) -> mdlp::Discretizer* { return new mdlp::BinDisc(3, mdlp::strategy_t::QUANTILE);});
static platform::RegistrarDiscretization registrarBU4("bin4u",
    [](void) -> mdlp::Discretizer* { return new mdlp::BinDisc(4, mdlp::strategy_t::UNIFORM);});
static platform::RegistrarDiscretization registrarBQ4("bin4q",
    [](void) -> mdlp::Discretizer* { return new mdlp::BinDisc(4, mdlp::strategy_t::QUANTILE);});
static platform::RegistrarDiscretization registrarBU5("bin5u",
    [](void) -> mdlp::Discretizer* { return new mdlp::BinDisc(5, mdlp::strategy_t::UNIFORM);});
static platform::RegistrarDiscretization registrarBQ5("bin5q",
    [](void) -> mdlp::Discretizer* { return new mdlp::BinDisc(5, mdlp::strategy_t::QUANTILE);});
static platform::RegistrarDiscretization registrarBU6("bin6u",
    [](void) -> mdlp::Discretizer* { return new mdlp::BinDisc(6, mdlp::strategy_t::UNIFORM);});
static platform::RegistrarDiscretization registrarBQ6("bin6q",
    [](void) -> mdlp::Discretizer* { return new mdlp::BinDisc(6, mdlp::strategy_t::QUANTILE);});
static platform::RegistrarDiscretization registrarBU7("bin7u",
    [](void) -> mdlp::Discretizer* { return new mdlp::BinDisc(7, mdlp::strategy_t::UNIFORM);});
static platform::RegistrarDiscretization registrarBQ7("bin7q",
    [](void) -> mdlp::Discretizer* { return new mdlp::BinDisc(7, mdlp::strategy_t::QUANTILE);});
static platform::RegistrarDiscretization registrarBU8("bin8u",
    [](void) -> mdlp::Discretizer* { return new mdlp::BinDisc(8, mdlp::strategy_t::UNIFORM);});
static platform::RegistrarDiscretization registrarBQ8("bin8q",
    [](void) -> mdlp::Discretizer* { return new mdlp::BinDisc(8, mdlp::strategy_t::QUANTILE);});
static platform::RegistrarDiscretization registrarBU9("bin9u",
    [](void) -> mdlp::Discretizer* { return new mdlp::BinDisc(9, mdlp::strategy_t::UNIFORM);});
static platform::RegistrarDiscretization registrarBQ9("bin9q",
    [](void) -> mdlp::Discretizer* { return new mdlp::BinDisc(9, mdlp::strategy_t::QUANTILE);});
static platform::RegistrarDiscretization registrarBU10("bin10u",
    [](void) -> mdlp::Discretizer* { return new mdlp::BinDisc(10, mdlp::strategy_t::UNIFORM);});
static platform::RegistrarDiscretization registrarBQ10("bin10q",
    [](void) -> mdlp::Discretizer* { return new mdlp::BinDisc(10, mdlp::strategy_t::QUANTILE);});
#endif