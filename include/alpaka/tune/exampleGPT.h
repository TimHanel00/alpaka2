//
// Created by tim on 01.03.25.
//

#ifndef EXAMPLEGPT_H
#define EXAMPLEGPT_H
#include "alpaka/KernelBundle.hpp"
#include "alpaka/core/DemangleTypeNames.hpp"
#include "tuner.hpp"

#include <toml.hpp>

#include <chrono>
#include <fstream>
#include <iostream>
#include <map>
#include <memory>
#include <string>
#include <tuple>
#include <typeinfo>
#include <utility>
#include <variant>
#include <vector>

int main() {
    Tuner<int, double>::loadConfig("tuning_data.toml");

    auto h = static_cast<std::size_t>(2);
    auto f = static_cast<std::uint32_t>(3);

    auto kernelFn = [](auto acc, auto stride, auto rotation)
    { std::cout << "Kernel executing with stride: " << stride << " and rotation: " << rotation << std::endl; };
    auto & tunerInstance=alpaka::TunerWrapper::init<float,double,alpaka::tune::strategy::initialValues>();
    tunerInstance.loadConfig("tuning_data.toml");

    alpaka::KernelBundle kernel(kernelFn, Tuneable("stride", h), Tuneable("rotation", f));
    {
        auto event=tunerInstance.createTimeEvent(kernel);
        Tuner<int, double>::setThreadBlockTune(kernel);
        Tuner<int, double>::setThreadTune(kernel);
        auto parameters = Tuner<int, double>::tune(kernel);
    }

    Tuner<int, double>::write("tuning_data.toml");
}

#endif //EXAMPLEGPT_H
