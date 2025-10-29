//
// Created by tim on 11.06.25.
//

#ifndef QUEUE_H
#define QUEUE_H
#include <alpaka/tune/utils/Random.hpp>

#include <functional>
#include <optional>
#include <queue>
#include <vector>

namespace alpaka::tune::core::peripherals
{

#define MAXQUEUESIZE 1

    template<typename T_Configs>
    struct ConfigQueue
    {
        using OptionalRef = std::optional<std::reference_wrapper<T_Configs>>;
        static constexpr uint32_t maxQueueSize = MAXQUEUESIZE;

        std::vector<OptionalRef> configs;
        std::vector<uint32_t> consecutiveRuns;
        std::queue<uint32_t> freeSlots;

        uint32_t validCount = 0;
        uint32_t maxConsecutiveRuns = 3;
        std::optional<uint32_t> lastIndex = std::nullopt;

        ConfigQueue()
        {
            configs.resize(maxQueueSize);
            consecutiveRuns.resize(maxQueueSize, 0u);
            for(uint32_t i = 0; i < maxQueueSize; ++i)
                freeSlots.push(i);
        }

        [[nodiscard]] bool empty() const noexcept
        {
            return (validCount == 0);
        }

        [[nodiscard]] bool full() const noexcept
        {
            return validCount >= maxQueueSize;
        }

        [[nodiscard]] uint32_t size() const noexcept
        {
            return validCount;
        }

        void push_back(T_Configs& config)
        {
            // if the strategy returns a already retired config we reset its state or otherwise it will never be
            // used by the Queue, if multiple entries of the same config are in the queue the
            // retirement of one will mean the implicit retirement of the rest
            if(config.state == config::ConfigState::Retired)
            {
                config.state = config::ConfigState::Initialized;
            }
            if(!freeSlots.empty())
            {
                uint32_t idx = freeSlots.front();
                freeSlots.pop();
                configs[idx] = std::ref(config);
                consecutiveRuns[idx] = 0u;
                ++validCount;
            }
            else
            {
                configs.emplace_back(std::ref(config));
                consecutiveRuns.emplace_back(0u);
                ++validCount;
            }
        }

        std::optional<std::reference_wrapper<T_Configs>> get()
        {
            if(validCount == 0)
                return std::nullopt;

            // Try to reuse the last returned config
            if(lastIndex.has_value())
            {
                uint32_t idx = lastIndex.value();
                auto& opt = configs[idx];
                bool configRetired = (opt->get().state == config::ConfigState::Retired);
                if(opt && !configRetired)
                {
                    if(consecutiveRuns[idx] < maxConsecutiveRuns)
                    {
                        ++consecutiveRuns[idx];
                        return opt.value();
                    }
                    // reset counter after max consecutive runs
                    consecutiveRuns[idx] = 0u;
                }
                else if(opt && configRetired)
                {
                    opt.reset();
                    freeSlots.push(idx);
                    --validCount;
                    consecutiveRuns[idx] = 0u;
                    lastIndex.reset();
                }
            }

            // Pick a random new one
            auto& rng = alpaka::tune::strategy::RNG::get();
            std::uniform_int_distribution<uint32_t> dist(0u, static_cast<uint32_t>(configs.size() - 1));

            for(uint32_t attempts = 0; attempts < configs.size(); ++attempts)
            {
                uint32_t idx = dist(rng);
                auto& opt = configs[idx];
                if(!opt)
                    continue;

                T_Configs& cfg = opt->get();

                if(cfg.state == config::ConfigState::Retired)
                {
                    opt.reset();
                    freeSlots.push(idx);
                    --validCount;
                    consecutiveRuns[idx] = 0u;
                    continue;
                }

                lastIndex = idx;
                consecutiveRuns[idx] = 1u;
                return cfg;
            }

            // Nothing valid found
            lastIndex.reset();
            return std::nullopt;
        }
    };
} // namespace alpaka::tune::core::peripherals
#endif // QUEUE_H
