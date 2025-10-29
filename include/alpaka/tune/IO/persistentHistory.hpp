//
// Created by tim on 19.03.25.
//

#ifndef TUNINGHISTORY_H
#define TUNINGHISTORY_H

#pragma once
#include <nlohmann/json.hpp>

#include <algorithm>
#include <array>
#include <atomic>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <ios>
#include <mutex>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <utility>
#include <vector>

namespace alpaka::tune::IO::detail
{
    template<typename T_Descriptor, typename T_Config, typename... ModelArgs>
    inline nlohmann::json make_descriptor_json(
        KernelTuningMetadata<T_Config, T_Descriptor> const& metaData,
        KernelTuningModel<ModelArgs...> const& model)
    {
        nlohmann::json params = nlohmann::json::array();

        // Iterate over the descriptor tuple
        alpaka::tune::utils::for_each(
            metaData.descriptor,
            [&](auto const& elem)
            {
                params.push_back(
                    {{"name", elem.m_name},
                     {"id", static_cast<std::size_t>(elem.ID)},
                     {"kind", static_cast<std::size_t>(elem.kind)}});
            });

        nlohmann::json j;
        j["parameters"] = std::move(params);

        j["model"] = {{"numDims", model.numDims}, {"numValues", nlohmann::json::array()}};

        for(std::size_t i = 0; i < model.numDims; ++i)
        {
            auto val = static_cast<std::uint64_t>(model.m_numValues[i]);
            j["model"]["numValues"].push_back(val);
            std::cout << val;
            if(i + 1 < model.numDims)
                std::cout << ", ";
        }
        return j;
    }

    // prepare Metadata for JSON top level key
    template<typename T_Config, typename T_Descriptor>
    inline nlohmann::json make_metadata_json(KernelTuningMetadata<T_Config, T_Descriptor> const& md)
    {
        return nlohmann::json{
            {"device", md.device},
            {"executor", md.executor},
            {"kernel", md.kernel},
            {"targetMetric", md.targetMetric},
            {"specifiers", md.specifiers}};
    }

    inline bool equal_strings(std::string const& a, std::string const& b)
    {
        return a == b;
    }

    inline nlohmann::json read_json_file(std::filesystem::path const& p)
    {
        if(!std::filesystem::exists(p))
            return nlohmann::json::object();
        std::ifstream ifs(p, std::ios::in | std::ios::binary);
        if(!ifs)
            return nlohmann::json::object();
        nlohmann::json j;
        try
        {
            ifs >> j;
        }
        catch(...)
        {
            return nlohmann::json::object();
        }
        if(!j.is_object())
            return nlohmann::json::object();
        return j;
    }

    inline bool atomically_write(std::filesystem::path const& p, nlohmann::json const& j)
    {
        auto tmp = p;
        tmp += ".tmp";
        {
            std::ofstream ofs(tmp, std::ios::out | std::ios::binary | std::ios::trunc);
            if(!ofs)
                return false;
            ofs << std::setw(2) << j;
        }
        std::error_code ec;
        std::filesystem::rename(tmp, p, ec);
        if(ec)
        {
            std::filesystem::remove(p, ec);
            std::filesystem::rename(tmp, p, ec);
        }
        return !ec;
    }

    template<typename T_Config>
    inline nlohmann::json config_to_json(config::ConfigRecord<T_Config> const& cfg)
    {
        nlohmann::json j;
        j["stamp"] = (cfg.state == config::ConfigState::Invalid) ? static_cast<std::int64_t>(-1)
                                                                 : static_cast<std::int64_t>(cfg.stamp);
        j["indices"] = nlohmann::json::array();
        for(auto v : cfg.m_config)
            j["indices"].push_back(static_cast<typename T_Config::value_type>(v));
        j["measurements"]
            = std::vector(cfg.getMeasurements().history.begin(), cfg.getMeasurements().history.end()); // doubles
        return j;
    }

    inline bool json_has_required_context_shape(nlohmann::json const& ctx)
    {
        return ctx.is_object() && ctx.contains("metadata") && ctx.contains("descriptor") && ctx["metadata"].is_object()
               && ctx["descriptor"].is_object() && ctx.contains("configs") && ctx["configs"].is_array()
               && ctx["descriptor"].contains("model") && ctx["descriptor"]["model"].is_object()
               && ctx["descriptor"]["model"].contains("numDims") && ctx["descriptor"]["model"].contains("numValues");
    }

    inline std::string build_context_concat_string(nlohmann::json const& metadata, nlohmann::json descriptorFull)
    {
        if(descriptorFull.contains("model") && descriptorFull["model"].is_object())
        {
            auto& m = descriptorFull["model"];
            if(m.contains("numValues"))
                m.erase("numValues");
        }

        std::ostringstream oss;
        oss << "device=" << metadata.value("device", "") << "|executor=" << metadata.value("executor", "")
            << "|kernel=" << metadata.value("kernel", "") << "|targetMetric=" << metadata.value("targetMetric", "");

        auto specs = metadata.value("specifiers", std::vector<std::string>{});
        if(!specs.empty())
        {
            oss << "|specifiers=["
                << std::accumulate(
                       std::next(specs.begin()),
                       specs.end(),
                       specs.front(),
                       [](auto a, auto const& b) { return std::move(a) + "," + b; })
                << "]";
        }

        auto formatParameter = [](auto const& p)
        {
            return p.value("name", "") + "|" + std::to_string(p.value("id", 0)) + "|"
                   + std::to_string(p.value("kind", 0));
        };
        // Descriptor: parameters in order (name|id|kind-int)
        auto const& params = descriptorFull["parameters"];
        if(!params.empty())
        {
            oss << "|params["
                << std::accumulate(
                       std::next(params.begin()),
                       params.end(),
                       formatParameter(params.front()),
                       [&](auto acc, auto const& p) { return std::move(acc) + ";" + formatParameter(p); })
                << "]";
        }

        // Model.numDims (model lives inside descriptor)
        auto const& mdl = descriptorFull["model"];
        oss << "|numDims=" << mdl.value("numDims", 0);

        return oss.str();
    }

    inline std::string compute_context_id(nlohmann::json const& metadata, nlohmann::json const& descriptorFull)
    {
        auto concat = build_context_concat_string(metadata, descriptorFull);
        return std::to_string(std::hash<std::string>{}(concat));
    }
} // namespace alpaka::tune::IO::detail

namespace alpaka::tune::IO
{
    struct PersistentHistory
    {
    public:
        static PersistentHistory& get(std::string const& filename)
        {
            static std::mutex s_instancesMx;
            std::scoped_lock lk(s_instancesMx);
            static std::unordered_map<std::string, PersistentHistory> instances;
            auto [it, inserted] = instances.try_emplace(filename, filename);
            ++it->second.nr_StakeHolders; // read
            ++it->second.nr_StakeHolders; // write
            return it->second;
        }

        PersistentHistory() = default;

        explicit PersistentHistory(std::string fn) : m_filename(std::move(fn))
        {
        }

        // match: strict metadata, descriptor order & triples, model.numDims strict, specifiers set-equal
        template<typename... T_TunableTuple, typename T_Config, typename T_Descriptor>
        bool match(
            KernelTuningModel<T_TunableTuple...> const& model,
            KernelTuningMetadata<T_Config, T_Descriptor> const& metadata,
            nlohmann::json const& ctxJson) const noexcept
        {
            using namespace detail;
            if(!json_has_required_context_shape(ctxJson))
                return false;

            auto const& jm = ctxJson["metadata"];
            auto const& jd = ctxJson["descriptor"];
            if(!equal_strings(jm.value("device", ""), metadata.device))
                return false;
            if(!equal_strings(jm.value("executor", ""), metadata.executor))
                return false;
            if(!equal_strings(jm.value("kernel", ""), metadata.kernel))
                return false;
            if(!equal_strings(jm.value("targetMetric", ""), metadata.targetMetric))
                return false;

            auto specJ = jm.value("specifiers", std::vector<std::string>{});
            if(specJ != metadata.specifiers)
                return false;

            // descriptor parameters strict order: (name,id,kind-int)
            auto const& jparams = jd["parameters"];
            if(!jparams.is_array())
                return false;

            // Build a temp JSON from runtime descriptor to compare structurally
            nlohmann::json jd_rt = make_descriptor_json(metadata, model);
            auto const& jparams_rt = jd_rt["parameters"];
            if(jparams.size() != jparams_rt.size())
                return false;

            for(std::size_t i = 0; i < jparams.size(); ++i)
            {
                auto const& a = jparams[i];
                auto const& b = jparams_rt[i];
                if(a.value("name", "") != b.value("name", ""))
                    return false;
                if(a.value("id", -1) != b.value("id", -1))
                    return false;
                if(a.value("kind", -1) != b.value("kind", -1))
                    return false;
            }
            auto const nd_file = static_cast<std::size_t>(jd["model"].value("numDims", 0));
            if(nd_file != model.numDims)
                return false;

            return true;
        }

        // READ: returns number of configs loaded (after filtering)
        template<typename T_MetricInterface, typename... ModelArgs, typename T_Config, typename T_ConfigDescriptor>
        std::size_t read(
            KernelTuningModel<ModelArgs...> const& model,
            alpaka::tune::IO::ActiveHistory<T_Config>& history,
            KernelTuningMetadata<T_Config, T_ConfigDescriptor> const& metadata,
            core::peripherals::EnvironmentState<T_Config>& state) noexcept
        {
            using namespace detail;
            static constexpr auto numDimsV = KernelTuningModel<ModelArgs...>::numDims;
            using value_type = typename T_Config::value_type;
            std::scoped_lock lk(m_mx);


            auto root = read_json_file(m_filename);
            if(!root.is_object())
            {
                return 0;
            }

            auto md = make_metadata_json(metadata);

            auto descFull = make_descriptor_json(metadata, model);

            auto myId = compute_context_id(md, descFull);


            std::vector<std::string> candidateKeys;

            // Fast path: top-level key present and its contextId matches ours
            if(root.contains(myId))
            {
                auto const& ctx = root.at(myId);
                if(ctx.is_object() && ctx.value("contextId", std::string{}) == myId
                   && json_has_required_context_shape(ctx) && match(model, metadata, ctx))
                {
                    candidateKeys.push_back(myId);
                }
            }

            // Fallback: scan all keys (support older writers or collisions)
            if(candidateKeys.empty())
            {
                for(auto it = root.begin(); it != root.end(); ++it)
                {
                    if(!it.value().is_object())
                        continue;
                    auto const& ctx = it.value();
                    if(!json_has_required_context_shape(ctx))
                        continue;

                    if(ctx.value("contextId", std::string{}) == myId && match(model, metadata, ctx))
                    {
                        candidateKeys.push_back(it.key());
                    }
                }
            }

            if(candidateKeys.empty())
                return 0;

            auto const& ctx = root.at(candidateKeys.front());
            auto const& jd = ctx["descriptor"];
            auto const& jmv = jd["model"]["numValues"];

            std::array<value_type, numDimsV> bounds{};
            std::ranges::transform(
                jmv,
                bounds.begin(),
                [](auto const& v) { return static_cast<value_type>(v.template get<value_type>()); });

            for(size_t i = 0; i < numDimsV; ++i)
            {
                std::cout << bounds[i];
                if(i + 1 < numDimsV)
                    std::cout << ", ";
            }

            std::size_t loaded = 0;

            for(auto const& jcfg : ctx["configs"])
            {
                ++state.numberOfCheckedConfigs;
                if(!jcfg.is_object())
                {
                    continue;
                }
                if(!jcfg.contains("indices") || !jcfg["indices"].is_array())
                {
                    continue;
                }
                if(jcfg["indices"].size() != numDimsV)
                {
                    continue;
                }

                std::array<value_type, numDimsV> arr{};
                bool inRange = true;
                for(std::size_t i = 0; i < numDimsV; ++i)
                {
                    auto const v = jcfg["indices"][i].get<long long>();
                    if(static_cast<value_type>(v) >= bounds[i])
                    {
                        inRange = false;
                        break;
                    }
                    arr[i] = static_cast<value_type>(v);
                }

                if(!inRange)
                {
                    continue;
                }

                T_Config cfg(arr);
                auto& entry = history.getOrCreate(cfg);

                uint64_t stampTmp = jcfg["stamp"].get<std::int64_t>();
                if(stampTmp == -1)
                {
                    entry.state = config::ConfigState::Invalid;
                    entry.stamp = -1;
                    continue;
                }

                entry.state = config::ConfigState::Empty;
                entry.stamp = state.numValidConfigs++;
                for(size_t i = 0; i < numDimsV; ++i)
                {
                    std::cout << arr[i];
                    if(i + 1 < numDimsV)
                        std::cout << ",";
                }
                std::cout << "]" << std::endl;

                if(jcfg.contains("measurements") && jcfg["measurements"].is_array())
                {
                    for(auto const& m : jcfg["measurements"])
                    {
                        auto val = m.get<double_t>();
                        core::peripherals::updateMetrics<false, T_MetricInterface>(entry, state, val);
                    }
                }

                ++loaded;
            }
            return loaded;
        }

        // WRITE: returns number of configs written
        template<typename... ModelArgs, typename T_Config, typename T_ConfigDescriptor>
        std::size_t write(
            KernelTuningModel<ModelArgs...> const& model,
            ActiveHistory<T_Config> const& history,
            KernelTuningMetadata<T_Config, T_ConfigDescriptor> const& metadata) noexcept
        {
            using namespace detail;
            std::scoped_lock lk(m_mx);

            auto root = read_json_file(m_filename);
            if(!root.is_object())
            {
                root = nlohmann::json::object();
            }

            // Compute our id from the active runtime (not derived from file)
            auto md = make_metadata_json(metadata);

            auto descFull = make_descriptor_json(metadata, model);

            auto contextId = compute_context_id(md, descFull);


            nlohmann::json ctx;
            ctx["contextId"] = contextId; // store inside for cross-checks
            ctx["metadata"] = md;
            ctx["descriptor"] = descFull;
            ctx["version"] = 1;
            ctx["configs"] = nlohmann::json::array();

            std::cout << "[PersistentHistory] Writing retired configs..." << std::endl;
            std::size_t written = 0;
            for(auto const& ref : history.getOrderedHistory())
            {
                auto const& rec = ref.get();
                using StateT = decltype(rec.state);
                if(rec.state == config::ConfigState::Empty || rec.state == config::ConfigState::Uninitialized
                   || rec.state == config::ConfigState::WarmUp)
                    continue;
                // invalid,ci reached,initialized, retired
                ctx["configs"].push_back(detail::config_to_json(rec));
                ++written;
            }


            // Merge/replace our top-level key with our computed id
            root[contextId] = std::move(ctx);

            bool const ok = atomically_write(m_filename, root);

            if(!ok)
            {
                std::cerr << "[PersistentHistory] ERROR: Failed to atomically write JSON file: " << m_filename
                          << std::endl;
                return 0;
            }

            return written;
        }


    private:
        std::string m_filename;
        mutable std::mutex m_mx;
        std::atomic<uint64_t> nr_StakeHolders{0};
    };
} // namespace alpaka::tune::IO

#endif // TUNINGHISTORY_H
