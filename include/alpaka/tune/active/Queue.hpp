//
// Created by tim on 11.06.25.
//

#ifndef QUEUE_H
#define QUEUE_H
#include <vector>
#include <functional>
#include <optional>
namespace alpaka::tune{
template<typename T_Configs>
struct ConfigQueue {
    std::vector<std::reference_wrapper<T_Configs>> configs;
    std::size_t roundRobinIndex = 0;

    bool empty() {
        for (auto& cfg : configs) {
            if (!cfg.get().fullFlag)
                return false;
        }
        return true;
    }

    void push_back(T_Configs& config) {
        configs.push_back(std::ref(config));
    }

    std::optional<std::reference_wrapper<T_Configs>> getFirst() {
        while (!configs.empty()) {
            if (!configs.front().get().fullFlag) {
                return configs.front();
            }
            configs.erase(configs.begin());
            if (roundRobinIndex > 0) roundRobinIndex--;
        }
        return std::nullopt;
    }

    std::optional<std::reference_wrapper<T_Configs>> getLast() {
        while (!configs.empty()) {
            if (!configs.back().get().fullFlag) {
                return configs.back();
            }
            configs.pop_back();
            if (roundRobinIndex >= configs.size() && roundRobinIndex > 0) {
                roundRobinIndex--;
            }
        }
        return std::nullopt;
    }

    std::optional<std::reference_wrapper<T_Configs>> getRoundRobin() {
        if (configs.empty()) return std::nullopt;

        std::size_t attempts = 0;
        while (attempts < configs.size()) {
            std::size_t idx = roundRobinIndex % configs.size();
            auto& cfg = configs[idx];

            if (!cfg.get().fullFlag) {
                roundRobinIndex = (idx + 1) % configs.size();
                return cfg;
            } else {
                configs.erase(configs.begin() + idx);

                if (idx < roundRobinIndex && roundRobinIndex > 0) roundRobinIndex--;
                if (configs.empty()) return std::nullopt;
            }
            attempts++;
        }
        return std::nullopt;
    }
};
}
#endif //QUEUE_H
