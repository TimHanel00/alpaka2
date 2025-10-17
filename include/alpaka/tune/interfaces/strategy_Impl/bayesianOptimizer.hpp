//
// Created by tim on 01.08.25.
//
#if defined(strategy_bayesianOptimization)
#    ifndef BAYESIANOPTIMIZER_H
#        define BAYESIANOPTIMIZER_H
// brings in GaussianProcess & MultiAcquisition

#        include <Eigen/Dense>
#        include <cassert>
#        include <limits>
#        include <optional>
#        include <vector>

namespace detail
{
    // Normal CDF and PDF for double precision
    inline double phi(double x)
    {
        static double const inv_sqrt_2pi = 0.3989422804014327; // 1 / sqrt(2π)
        return inv_sqrt_2pi * std::exp(-0.5 * x * x);
    }

    inline double Phi(double x)
    {
        return 0.5 * (1.0 + std::erf(x / std::sqrt(2.0)));
    }

    // Matérn‑3/2 kernel
    inline double matern32(Eigen::VectorXd const& a, Eigen::VectorXd const& b, double length_scale)
    {
        double const r = (a - b).norm();
        double const s = std::sqrt(3.0) * r / length_scale;
        return (1.0 + s) * std::exp(-s);
    }
} // namespace detail

// ------------------------------------------------------ Gaussian‑Process model
class GaussianProcess
{
public:
    explicit GaussianProcess(double length_scale = 2.0, double noise = 1e-6) : ell_(length_scale), noise_(noise)
    {
    }

    void fit(std::vector<Eigen::VectorXd> const& X, std::vector<double> const& y)
    {
        std::size_t const n = X.size();
        if(n == 0)
            return; // nothing to fit yet

        // Build covariance matrix K
        Eigen::MatrixXd K(n, n);
        for(std::size_t i = 0; i < n; ++i)
        {
            K(i, i) = detail::matern32(X[i], X[i], ell_) + noise_;
            for(std::size_t j = i + 1; j < n; ++j)
            {
                double k = detail::matern32(X[i], X[j], ell_);
                K(i, j) = k;
                K(j, i) = k;
            }
        }

        // Cholesky factorisation: K = LLᵀ
        llt_.compute(K);
        alpha_ = llt_.solve(Eigen::Map<Eigen::VectorXd const>(y.data(), y.size()));
        X_ = &X; // keep reference (safe: GP outlives X in optimiser)
    }

    /**
     * Predict mean µ and variance σ² for a single point x
     */
    std::pair<double, double> predict(Eigen::VectorXd const& x) const
    {
        if(!X_ || X_->empty())
        {
            // GP untrained – return broad prior
            return {0.0, 1.0};
        }

        std::size_t const n = X_->size();
        Eigen::VectorXd k(n);
        for(std::size_t i = 0; i < n; ++i)
            k(i) = detail::matern32((*X_)[i], x, ell_);

        double const mu = k.dot(alpha_);
        auto const v = llt_.matrixL().solve(k);
        double const var = std::max(1e-12, detail::matern32(x, x, ell_) - v.squaredNorm());
        return {mu, var};
    }

private:
    double ell_;
    double noise_;
    Eigen::LLT<Eigen::MatrixXd> llt_;
    Eigen::VectorXd alpha_;
    std::vector<Eigen::VectorXd> const* X_ = nullptr; // non‑owning
};

// ----------------------------------------------------- acquisition functions
class AcquisitionBase
{
public:
    virtual ~AcquisitionBase() = default;

    /**
     * Return an *error metric* (lower is better because we *minimise* runtime).
     * The caller passes GP‑predicted mean µ, std‑dev σ and current best f⁺.
     */
    virtual double score(double mu, double sigma, double best) const = 0;

    virtual char const* name() const = 0;
};

class ExpectedImprovement : public AcquisitionBase
{
public:
    double score(double mu, double sigma, double best) const override
    {
        if(sigma < 1e-12)
            return 0.0; // no uncertainty ⇒ no improvement
        double const improvement = best - mu;
        double const z = improvement / sigma;
        double const ei = improvement * detail::Phi(z) + sigma * detail::phi(z);
        return -ei; // convert to minimisation (max EI ⇒ min -EI)
    }

    char const* name() const override
    {
        return "EI";
    }
};

class ProbabilityOfImprovement : public AcquisitionBase
{
public:
    double score(double mu, double sigma, double best) const override
    {
        if(sigma < 1e-12)
            return 0.0;
        double const z = (best - mu) / sigma;
        return -detail::Phi(z);
    }

    char const* name() const override
    {
        return "POI";
    }
};

class LowerConfidenceBound : public AcquisitionBase
{
public:
    explicit LowerConfidenceBound(double kappa = 2.0) : kappa_(kappa)
    {
    }

    double score(double mu, double sigma, double /*best*/) const override
    {
        return mu - kappa_ * sigma;
    }

    char const* name() const override
    {
        return "LCB";
    }

private:
    double kappa_;
};

// ------------------------------------------------------- multi‑acquisition set
class MultiAcquisition
{
public:
    struct Params
    {
        int skip_threshold{5};
        double discount_factor{0.9};
        Params() {};
    };

    MultiAcquisition(MultiAcquisition&&) noexcept = default;
    MultiAcquisition& operator=(MultiAcquisition&&) noexcept = default;

    explicit MultiAcquisition(Params p = Params{}) : params_(p)
    {
        // The order matters – see Table I
        acq_.emplace_back(std::make_unique<ExpectedImprovement>());
        acq_.emplace_back(std::make_unique<ProbabilityOfImprovement>());
        acq_.emplace_back(std::make_unique<LowerConfidenceBound>());

        active_.assign(acq_.size(), true);
        duplicate_counter_.assign(acq_.size(), 0);
        // reserve per‑acquisition observations history (discounted score)
        history_.resize(acq_.size());
    }

    /**
     * Pick an index from *candidates* that has not yet been evaluated.
     *
     * ‑ gp          : current surrogate model (trained on valid points)
     * ‑ evaluated   : bitmap indicating which candidate indices have already
     *                  been evaluated ⇒ acquisition only considers false bits.
     * ‑ observations: vector of observed costs matching evaluated indices
     *                  (ignored when evaluated[i] == false).
     * ⟶ Returns chosen index (or ‑1 if search exhausted).
     */
    int suggest(
        GaussianProcess const& gp,
        std::vector<Eigen::VectorXd> const& candidates,
        std::vector<bool> const& evaluated,
        std::vector<std::optional<double>> const& observations) // same size as evaluated
    {
        auto const remaining = std::count(evaluated.begin(), evaluated.end(), false);
        if(remaining == 0)
            return -1; // all done

        // Determine current f⁺ (best *valid* observation so far)
        double f_best = std::numeric_limits<double>::infinity();
        for(auto const& o : observations)
            if(o && *o < f_best)
                f_best = *o;
        if(!std::isfinite(f_best))
            f_best = 0.0; // no data yet

        // Round‑robin over active acquisition functions
        for(std::size_t attempt = 0; attempt < acq_.size(); ++attempt)
        {
            size_t idx = (cursor_ + attempt) % acq_.size();
            if(!active_[idx])
                continue; // currently skipped

            // Scan candidates to obtain best according to this acquisition
            int best_id = -1;
            double best_score = std::numeric_limits<double>::infinity();

            for(std::size_t c = 0; c < candidates.size(); ++c)
            {
                if(evaluated[c])
                    continue;
                auto const [mu, var] = gp.predict(candidates[c]);
                double const sigma = std::sqrt(var);
                double const s = acq_[idx]->score(mu, sigma, f_best);
                if(s < best_score)
                {
                    best_score = s;
                    best_id = static_cast<int>(c);
                }
            }

            if(best_id == -1)
                break; // should not happen

            // Duplicate detection: did *another* acquisition already propose
            // the same index in a previous iteration?
            if(!last_suggestions_.empty()
               && std::find(last_suggestions_.begin(), last_suggestions_.end(), best_id) != last_suggestions_.end())
            {
                duplicate_counter_[idx]++;
            }
            else
            {
                duplicate_counter_[idx] = 0; // reset streak
            }

            // Record observation (even duplicates count into history)
            history_[idx].push_back(best_score);

            // React to prolonged duplicates
            if(duplicate_counter_[idx] >= params_.skip_threshold)
            {
                handle_duplicates();
            }

            cursor_ = (idx + 1) % acq_.size(); // next call continues round‑robin
            last_suggestions_.push_back(best_id);
            if(last_suggestions_.size() > acq_.size())
                last_suggestions_.erase(last_suggestions_.begin()); // keep bounded

            return best_id;
        }
        return -1; // no active acquisition could suggest a point
    }

private:
    void handle_duplicates()
    {
        // Calculate discounted observation score d_o for each active acq func
        std::size_t const n_acq = acq_.size();
        std::vector<double> scores(n_acq, std::numeric_limits<double>::infinity());

        for(std::size_t i = 0; i < n_acq; ++i)
        {
            if(!active_[i])
                continue;
            auto const& obs = history_[i];
            double d = 0.0;
            for(std::size_t t = 0; t < obs.size(); ++t)
            {
                double const weight = std::pow(params_.discount_factor, static_cast<int>(obs.size() - 1 - t));
                d += obs[t] * weight;
            }
            scores[i] = d;
        }

        // Retain acquisition func with *lowest* discounted observation score
        std::size_t keep = std::distance(scores.begin(), std::min_element(scores.begin(), scores.end()));
        for(std::size_t i = 0; i < n_acq; ++i)
            active_[i] = (i == keep);
    }

    Params params_;
    std::vector<std::unique_ptr<AcquisitionBase>> acq_;
    std::vector<bool> active_;
    std::vector<int> duplicate_counter_;
    std::vector<std::vector<double>> history_;
    std::vector<int> last_suggestions_;
    std::size_t cursor_ = 0; // round‑robin cursor
};

///***
///
///
///
///
/// Bayesion optimizer strategy IMplementation:
///
///
///
///
///****/
namespace detail
{
    template<class UI, class Fn>
    constexpr void for_each_param(UI&& ui, Fn&& fn)
    {
        std::apply([&](auto&... p) { (fn(p), ...); }, ui);
    }

    template<class UI>
    [[nodiscard]] constexpr std::size_t param_count(UI&& ui)
    {
        std::size_t c = 0;
        for_each_param(ui, [&](auto&&) { ++c; });
        return c;
    }

    template<class UI>
    [[nodiscard]] Eigen::VectorXd encode(UI&& ui, std::vector<std::size_t> const& card)
    {
        Eigen::VectorXd v(card.size());
        std::size_t i = 0;
        for_each_param(
            ui,
            [&](auto& p)
            {
                v(i) = static_cast<double>(p.index) / std::max<std::size_t>(1, card[i] - 1);
                ++i;
            });
        return v;
    }

    template<class UI>
    inline void restore(UI&& ui, std::vector<std::size_t> const& snap)
    {
        std::size_t i = 0;
        for_each_param(
            ui,
            [&](auto& p)
            {
                p.index = snap[i];
                p.value = p.getValues()[p.index];
                ++i;
            });
    }

    // RNG helpers ------------------------------------------------------------------------------
} // namespace detail

namespace alpaka::tune::strategy
{
    template<concepts::MetricInterface T_metricInterface, typename T_TuningModel, typename T_Config>
    void callRandomSearch(
        T_metricInterface& metricInterface, // the user specified metricInterface
        KernelTuningModelView<T_TuningModel>& model,
        ConfigStorage<T_Config>& config_storage, // this already returns the Config for a specific kernel backend Config
        EnvironmentState<T_Config>& environmentState);

    // -----------------------------------------------------------------------------
    // -----------------------------------------------------------------------------
    // bayesianOptimization – refactored
    // -----------------------------------------------------------------------------
    struct bayesianOptimization
    {
        /* ---------- tunables (unchanged) -------------------------------------- */
        std::size_t candidate_pool_size = 1024;
        std::size_t refresh_per_call = 50;

        /* ---------- state (unchanged) ----------------------------------------- */
        bool initialised_ = false;
        std::vector<std::size_t> cardinality_;
        using Vec = Eigen::VectorXd;
        using RowId = std::size_t;

        std::vector<Vec> X_;
        std::vector<double> y_;
        std::unordered_map<std::size_t, RowId> idx_map_;

        GaussianProcess gp_{2.0};
        MultiAcquisition multi_;

        struct Cand
        {
            std::size_t hash;
            Vec vec;
        };

        std::deque<Cand> pool_; // now a deque for cheap pop-front

        /* ---------- configuration -------------------------------------------- */
        static constexpr std::size_t cleanup_interval = 50;
        std::size_t call_counter_ = 0;

        template<typename UI>
        static auto snapshot_indices(UI&& ui) -> std::vector<std::size_t>;
        template<typename UI>
        static Vec encode(UI&& ui, std::vector<std::size_t> const& card);
        template<typename UI>
        static void restore(UI&& ui, std::vector<std::size_t> const& snap);

        template<typename T_Config, typename UI>
        void ingest_or_update(T_Config const& cfg, double median, UI& ui, bool& gp_dirty)
        {
            std::size_t const h = cfg.toHash();
            auto it = idx_map_.find(h);

            if(it == idx_map_.end()) // new Config
            {
                Vec v = encode(ui, cardinality_);
                RowId row = X_.size();
                X_.push_back(std::move(v));
                y_.push_back(median);
                idx_map_[h] = row;
                gp_dirty = true;
            }
            else if(y_[it->second] != median) // updated metric
            {
                y_[it->second] = median;
                gp_dirty = true;
            }
        }

        void remove_from_gp(std::size_t h)
        {
            if(auto it = idx_map_.find(h); it != idx_map_.end())
            {
                RowId row = it->second, last = X_.size() - 1;

                if(row != last) // move last row over the gap
                {
                    X_[row] = std::move(X_[last]);
                    y_[row] = y_[last];
                    for(auto& kv : idx_map_) // repair idx that pointed to 'last'
                        if(kv.second == last)
                        {
                            kv.second = row;
                            break;
                        }
                }
                X_.pop_back();
                y_.pop_back();
                idx_map_.erase(it);
            }
        }

        template<typename Metric, typename T_Model, typename T_Config, typename UI_type>
        bool enqueue_random(
            Metric& metricInterface,
            KernelTuningModelView<T_Model>& model,
            ConfigStorage<T_Config>& history,
            EnvironmentState<T_Config>& env,
            std::vector<std::size_t> const& snap,
            UI_type& ui_ref) // UI_type alias omitted for brevity
        {
            callRandomSearch(metricInterface, model, history, env);
            T_Config cfg = model.toConfig();
            std::size_t h = cfg.toHash();
            if(idx_map_.count(h))
                return false; // already training data

            Vec v = detail::encode(ui_ref, cardinality_);
            pool_.push_back({h, std::move(v)});
            model.fromConfig(cfg); // restore model state
            detail::restore(ui_ref, snap);
            return true;
        }

        template<typename Metric, typename T_Model, typename T_Config, typename UI_type>
        void maintain_pool(
            Metric& metricInterface,
            KernelTuningModelView<T_Model>& model,
            ConfigStorage<T_Config>& history,
            EnvironmentState<T_Config>& env,
            std::vector<std::size_t> const& snap,
            UI_type& ui_ref,
            std::size_t current_h)
        {
            /* cleanup (every cleanup_interval calls) */
            if(++call_counter_ % cleanup_interval == 0)
            {
                pool_.erase(
                    std::remove_if(
                        pool_.begin(),
                        pool_.end(),
                        [&](Cand const& c) { return history.contains(c) || idx_map_.count(c.hash); }),
                    pool_.end());
            }

            /* initial fill / periodic refresh */
            while(pool_.size() < candidate_pool_size)
            {
                std::size_t added = 0;
                while(added < refresh_per_call && pool_.size() < candidate_pool_size)
                    added += enqueue_random(metricInterface, model, history, env, snap, ui_ref);
                if(added == 0)
                    break; // nothing new could be generated
            }
        }

    public:
        template<concepts::MetricInterface T_metricInterface, typename T_Model, typename T_Config>
        void operator()(
            T_metricInterface& metricInterface,
            KernelTuningModelView<T_Model>& model,
            ConfigStorage<T_Config>& history,
            EnvironmentState<T_Config>& env)
        {
            init_once(model, history); // discover cardinalities

            auto& ui_ref = model.getUniformInterface();
            auto snapshot = snapshot_indices(ui_ref); // save param indices

            /* ingest the freshly-measured Config & metric ------------------ */
            T_Config const cfg = model.toConfig();
            std::size_t const h = cfg.toHash();
            bool gp_dirty = false;

            if(history.getOrCreate(cfg).state == ConfigState::Dummy)
            {
                remove_from_gp(h);
                gp_dirty = true;
                randomSample{}(metricInterface, model, history, env); // fallback
                return;
            }

            double median = history.get(cfg).getMetrics().get(median_t{}).template as<t_ns>();
            ingest_or_update(cfg, median, ui_ref, gp_dirty);

            if(gp_dirty && !X_.empty())
                gp_.fit(X_, y_);

            /* maintain candidate pool - the pool of configs from which the suggested Config will be selected */
            maintain_pool(metricInterface, model, history, env, snapshot, ui_ref, h);

            if(pool_.empty())
            {
                randomSample{}(metricInterface, model, history, env);
                return;
            }

            /* run acquisition & choose candidate*/
            std::vector<Vec> search;
            search.reserve(pool_.size() + X_.size());
            std::vector<bool> mask;
            mask.reserve(search.capacity());
            std::vector<std::optional<double>> obs;
            obs.reserve(search.capacity());

            for(auto const& c : pool_)
            {
                search.push_back(c.vec);
                mask.push_back(false);
                obs.emplace_back();
            }
            for(RowId i = 0; i < X_.size(); ++i)
            {
                search.push_back(X_[i]);
                mask.push_back(true);
                obs.emplace_back(y_[i]);
            }

            int sel = multi_.suggest(gp_, search, mask, obs);
            if(sel < 0 || static_cast<std::size_t>(sel) >= pool_.size())
                sel = 0;

            /* decode vec back onto UI & schedule evaluation ---------------- */
            detail::restore(ui_ref, snapshot); // reset indices first
            {
                std::size_t i = 0;
                detail::for_each_param(
                    ui_ref,
                    [&](auto& p)
                    {
                        auto const& vals = p.getValues();
                        std::size_t n = vals.size();
                        std::size_t idx
                            = static_cast<std::size_t>(std::round(pool_[sel].vec(i) * std::max<std::size_t>(1, n - 1)));
                        p.index = std::min(idx, n - 1);
                        p.value = vals[p.index];
                        ++i;
                    });
            }

            model.fromConfig(model.toConfig()); // trigger kernel re-instantiation
            history.getOrCreate(model.toConfig()); // ensure entry exists
        }
    };
} //namespace alpaka::tune::strategy
#endif //BAYESIANOPTIMIZER_H
#endif
