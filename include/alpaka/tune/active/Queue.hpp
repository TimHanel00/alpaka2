//
// Created by tim on 11.06.25.
//

#ifndef QUEUE_H
#define QUEUE_H
#include <functional>
#include <optional>
#include <vector>

namespace alpaka::tune
{
#define maxQueueSize 40

    template<typename T_Configs>
    struct ConfigQueue
    {
        using OptionalRef = std::optional<std::reference_wrapper<T_Configs>>;

        std::vector<OptionalRef> configs;
        std::queue<std::size_t> freeSlots;

        std::size_t currentIndex = 0;
        std::size_t consecutiveCount = 0;
        std::size_t maxConsecutiveRuns = 5;
        std::size_t validCount = 0;

        ConfigQueue()
        {
            configs.resize(maxQueueSize);
            for(std::size_t i = 0; i < maxQueueSize; ++i)
                freeSlots.push(i);
        }

        bool empty() const
        {
            return validCount == 0;
        }

        bool full() const
        {
            return validCount >= maxQueueSize;
        }

        std::size_t size() const
        {
            return validCount;
        }

        void push_back(T_Configs& config)
        {
            if(!freeSlots.empty())
            {
                std::size_t idx = freeSlots.front();
                freeSlots.pop();
                configs[idx] = std::ref(config);
                ++validCount;
            }
            else
            {
                // fallback if user extends maxQueueSize on purpose
                configs.emplace_back(std::ref(config));
                ++validCount;
            }
        }

        std::optional<std::reference_wrapper<T_Configs>> get()
        {
            if(validCount == 0)
            {
#ifdef Debug
                std::cout << "[ConfigQueue::get] No valid configs remaining (validCount == 0).\n";
#endif
                return std::nullopt;
            }

#ifdef Debug
            std::cout << "[ConfigQueue::get] Starting get() at index: " << currentIndex
                      << ", consecutiveCount: " << consecutiveCount << ", validCount: " << validCount
                      << ", totalSlots: " << configs.size() << "\n";
#endif

            for(std::size_t attempts = 0; attempts < configs.size(); ++attempts)
            {
                auto& opt = configs[currentIndex];
                if(!opt.has_value())
                {
#ifdef Debug
                    std::cout << "[ConfigQueue::get] Slot at index " << currentIndex << " is empty. Skipping.\n";
#endif
                    currentIndex = (currentIndex + 1) % configs.size();
                    continue;
                }

                T_Configs& cfg = opt.value().get();

                if(cfg.fullFlag)
                {
#ifdef Debug
                    std::cout << "[ConfigQueue::get] Slot at index " << currentIndex
                              << " is fullFlag. Removing. Config: " << cfg.toString() << "\n";
#endif
                    opt.reset();
                    freeSlots.push(currentIndex);
                    --validCount;
                    currentIndex = (currentIndex + 1) % configs.size();
                    consecutiveCount = 0;
                    continue;
                }

                if(consecutiveCount < maxConsecutiveRuns)
                {
#ifdef Debug
                    std::cout << "[ConfigQueue::get] Returning config at index " << currentIndex
                              << " (consecutiveCount = " << consecutiveCount + 1 << ")\n";
#endif
                    ++consecutiveCount;
                    return cfg;
                }

                // Move to next config
                currentIndex = (currentIndex + 1) % configs.size();
                consecutiveCount = 0;

                auto& nextOpt = configs[currentIndex];
                if(nextOpt.has_value())
                {
#ifdef Debug
                    std::cout << "[ConfigQueue::get] Switching to next config at index " << currentIndex
                              << ". Marking as WarmUp.\n";
#endif
                    nextOpt.value().get().state = ConfigState::WarmUp;
                    return nextOpt;
                }
#ifdef Debug
                else
                {
                    std::cout << "[ConfigQueue::get] Next slot at index " << currentIndex
                              << " is empty after switch. Continuing.\n";
                }
#endif
            }

#ifdef Debug
            std::cout << "[ConfigQueue::get] No valid configs found after full iteration.\n";
#endif
            return std::nullopt;
        }
    };

    struct configQueueManager
    {
#define maxNumberOfEntries 10
#define maxSameConfigsInARow 1
    };
} // namespace alpaka::tune
#endif // QUEUE_H
