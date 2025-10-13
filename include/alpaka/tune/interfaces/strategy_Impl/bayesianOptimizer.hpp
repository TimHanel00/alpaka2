//
// Created by tim on 01.08.25.
//
#if defined(strategy_bayesianOptimization)
#ifndef BAYESIANOPTIMIZER_H
#define BAYESIANOPTIMIZER_H
// brings in GaussianProcess & MultiAcquisition

#include <Eigen/Dense>
#include <vector>
#include <optional>
#include <limits>
#include <cassert>
namespace detail {
    // Normal CDF and PDF for double precision
    inline double phi(double x)
    {
        static const double inv_sqrt_2pi = 0.3989422804014327; // 1 / sqrt(2π)
        return inv_sqrt_2pi * std::exp(-0.5 * x * x);
    }

    inline double Phi(double x)
    {
        return 0.5 * (1.0 + std::erf(x / std::sqrt(2.0)));
    }

    // Matérn‑3/2 kernel
    inline double matern32(const Eigen::VectorXd &a,
                           const Eigen::VectorXd &b,
                           double length_scale)
    {
        const double r = (a - b).norm();
        const double s = std::sqrt(3.0) * r / length_scale;
        return (1.0 + s) * std::exp(-s);
    }
}

// ------------------------------------------------------ Gaussian‑Process model
class GaussianProcess
{
public:
    explicit GaussianProcess(double length_scale = 2.0,
                             double noise         = 1e-6)
        : ell_(length_scale), noise_(noise) {}

    void fit(const std::vector<Eigen::VectorXd> &X,
             const std::vector<double>          &y)
    {
        const std::size_t n = X.size();
        if (n == 0) return; // nothing to fit yet

        // Build covariance matrix K
        Eigen::MatrixXd K(n, n);
        for (std::size_t i = 0; i < n; ++i) {
            K(i, i) = detail::matern32(X[i], X[i], ell_) + noise_;
            for (std::size_t j = i + 1; j < n; ++j) {
                double k = detail::matern32(X[i], X[j], ell_);
                K(i, j) = k;
                K(j, i) = k;
            }
        }

        // Cholesky factorisation: K = LLᵀ
        llt_.compute(K);
        alpha_ = llt_.solve(Eigen::Map<const Eigen::VectorXd>(y.data(), y.size()));
        X_ = &X; // keep reference (safe: GP outlives X in optimiser)
    }

    /**
     * Predict mean µ and variance σ² for a single point x
     */
    std::pair<double, double> predict(const Eigen::VectorXd &x) const
    {
        if (!X_ || X_->empty()) {
            // GP untrained – return broad prior
            return {0.0, 1.0};
        }

        const std::size_t n = X_->size();
        Eigen::VectorXd k(n);
        for (std::size_t i = 0; i < n; ++i)
            k(i) = detail::matern32((*X_)[i], x, ell_);

        const double mu = k.dot(alpha_);
        const auto v    = llt_.matrixL().solve(k);
        const double var = std::max(1e-12, detail::matern32(x, x, ell_) - v.squaredNorm());
        return {mu, var};
    }

private:
    double ell_;
    double noise_;
    Eigen::LLT<Eigen::MatrixXd> llt_;
    Eigen::VectorXd alpha_;
    const std::vector<Eigen::VectorXd> *X_ = nullptr; // non‑owning
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

    virtual const char *name() const = 0;
};

class ExpectedImprovement : public AcquisitionBase
{
public:
    double score(double mu, double sigma, double best) const override
    {
        if (sigma < 1e-12) return 0.0; // no uncertainty ⇒ no improvement
        const double improvement = best - mu;
        const double z = improvement / sigma;
        const double ei = improvement * detail::Phi(z) + sigma * detail::phi(z);
        return -ei; // convert to minimisation (max EI ⇒ min -EI)
    }
    const char *name() const override { return "EI"; }
};

class ProbabilityOfImprovement : public AcquisitionBase
{
public:
    double score(double mu, double sigma, double best) const override
    {
        if (sigma < 1e-12) return 0.0;
        const double z = (best - mu) / sigma;
        return -detail::Phi(z);
    }
    const char *name() const override { return "POI"; }
};

class LowerConfidenceBound : public AcquisitionBase
{
public:
    explicit LowerConfidenceBound(double kappa = 2.0) : kappa_(kappa) {}

    double score(double mu, double sigma, double /*best*/) const override
    {
        return mu - kappa_ * sigma;
    }
    const char *name() const override { return "LCB"; }

private:
    double kappa_;
};

// ------------------------------------------------------- multi‑acquisition set
class MultiAcquisition
{
public:
    struct Params {
        int    skip_threshold{5};
        double discount_factor{0.9};
        Params(){};
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
    int suggest(const GaussianProcess               &gp,
                const std::vector<Eigen::VectorXd>  &candidates,
                const std::vector<bool>             &evaluated,
                const std::vector<std::optional<double>> &observations) // same size as evaluated
    {
        const auto remaining = std::count(evaluated.begin(), evaluated.end(), false);
        if (remaining == 0) return -1; // all done

        // Determine current f⁺ (best *valid* observation so far)
        double f_best = std::numeric_limits<double>::infinity();
        for (const auto &o : observations)
            if (o && *o < f_best) f_best = *o;
        if (!std::isfinite(f_best)) f_best = 0.0; // no data yet

        // Round‑robin over active acquisition functions
        for (std::size_t attempt = 0; attempt < acq_.size(); ++attempt) {
            size_t idx = (cursor_ + attempt) % acq_.size();
            if (!active_[idx]) continue; // currently skipped

            // Scan candidates to obtain best according to this acquisition
            int best_id = -1;
            double best_score = std::numeric_limits<double>::infinity();

            for (std::size_t c = 0; c < candidates.size(); ++c) {
                if (evaluated[c]) continue;
                const auto [mu, var] = gp.predict(candidates[c]);
                const double sigma  = std::sqrt(var);
                const double s      = acq_[idx]->score(mu, sigma, f_best);
                if (s < best_score) {
                    best_score = s;
                    best_id = static_cast<int>(c);
                }
            }

            if (best_id == -1) break; // should not happen

            // Duplicate detection: did *another* acquisition already propose
            // the same index in a previous iteration?
            if (!last_suggestions_.empty() &&
                std::find(last_suggestions_.begin(), last_suggestions_.end(), best_id) != last_suggestions_.end()) {
                duplicate_counter_[idx]++;
            } else {
                duplicate_counter_[idx] = 0; // reset streak
            }

            // Record observation (even duplicates count into history)
            history_[idx].push_back(best_score);

            // React to prolonged duplicates
            if (duplicate_counter_[idx] >= params_.skip_threshold) {
                handle_duplicates();
            }

            cursor_ = (idx + 1) % acq_.size(); // next call continues round‑robin
            last_suggestions_.push_back(best_id);
            if (last_suggestions_.size() > acq_.size())
                last_suggestions_.erase(last_suggestions_.begin()); // keep bounded

            return best_id;
        }
        return -1; // no active acquisition could suggest a point
    }

private:
    void handle_duplicates()
    {
        // Calculate discounted observation score d_o for each active acq func
        const std::size_t n_acq = acq_.size();
        std::vector<double> scores(n_acq, std::numeric_limits<double>::infinity());

        for (std::size_t i = 0; i < n_acq; ++i) {
            if (!active_[i]) continue;
            const auto &obs = history_[i];
            double d = 0.0;
            for (std::size_t t = 0; t < obs.size(); ++t) {
                const double weight = std::pow(params_.discount_factor, static_cast<int>(obs.size() - 1 - t));
                d += obs[t] * weight;
            }
            scores[i] = d;
        }

        // Retain acquisition func with *lowest* discounted observation score
        std::size_t keep = std::distance(scores.begin(),
                                         std::min_element(scores.begin(), scores.end()));
        for (std::size_t i = 0; i < n_acq; ++i)
            active_[i] = (i == keep);
    }

    Params params_;
    std::vector<std::unique_ptr<AcquisitionBase>>      acq_;
    std::vector<bool>                                  active_;
    std::vector<int>                                   duplicate_counter_;
    std::vector<std::vector<double>>                   history_;
    std::vector<int>                                   last_suggestions_;
    std::size_t                                        cursor_ = 0; // round‑robin cursor
};


#endif //BAYESIANOPTIMIZER_H
#endif
