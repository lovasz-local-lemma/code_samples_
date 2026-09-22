// eigensolve_sparse.cpp
// Shift-invert Lanczos for generalized eigenproblem K*phi = lambda*M*phi.
// Pure C++/Eigen implementation — no ARPACK dependency (Windows-compatible).
#include "solver/eigensolve_sparse.h"
#include <Eigen/SparseCholesky>
#include <Eigen/Dense>
#include <stdexcept>
#include <cmath>
#include <numeric>
#include <random>

namespace modal {

// Simple thick-restart Lanczos with shift-invert for the generalized problem.
// Solves K*phi = lambda*M*phi for the lowest num_modes eigenpairs.
// Uses shift sigma to handle near-singular K (free-free BCs).
EigenResult solve_sparse(const Eigen::SparseMatrix<double>& K,
                          const Eigen::SparseMatrix<double>& M,
                          int num_modes, double rigid_threshold)
{
    int n = (int)K.rows();
    int nev = std::min(num_modes + 5, n - 1);  // extra for rigid-body filtering
    int ncv = std::min(2 * nev + 1, n);         // Krylov subspace size

    // Regularize: add small diagonal to any DOF with zero stiffness AND zero mass
    // (e.g., interior theta DOFs in mixed membrane+beam that are uncoupled).
    Eigen::SparseMatrix<double> Kr = K, Mr = M;
    {
        double eps_k = 0.0, eps_m = 0.0;
        for (int i = 0; i < n; ++i) {
            eps_k = std::max(eps_k, std::abs(K.coeff(i,i)));
            eps_m = std::max(eps_m, std::abs(M.coeff(i,i)));
        }
        double reg_k = std::max(eps_k * 1e-10, 1e-10);
        double reg_m = std::max(eps_m * 1e-10, 1e-10);
        for (int i = 0; i < n; ++i) {
            if (std::abs(Kr.coeff(i,i)) < reg_k && std::abs(Mr.coeff(i,i)) < reg_m) {
                Kr.coeffRef(i,i) += reg_k;
                Mr.coeffRef(i,i) += reg_m;
            }
        }
    }

    // Shift-invert: OP = (K - sigma*M)^{-1} * M
    double sigma = 1e-4;
    Eigen::SparseMatrix<double> Kshift = Kr - sigma * Mr;
    Eigen::SimplicialLDLT<Eigen::SparseMatrix<double>> factored(Kshift);
    if (factored.info() != Eigen::Success)
        throw std::runtime_error("solve_sparse: shift-invert factorization failed");

    // Lanczos iteration
    Eigen::MatrixXd V(n, ncv);   // Krylov basis (M-orthonormal)
    Eigen::VectorXd alpha(ncv), beta(ncv);

    // Random starting vector, M-normalized
    std::mt19937 rng(42);
    std::normal_distribution<double> dist;
    Eigen::VectorXd v(n);
    for (int i = 0; i < n; ++i) v(i) = dist(rng);
    // M-normalize: v = v / sqrt(v^T M v)
    double mv_norm = std::sqrt(v.dot(Mr * v));
    v /= mv_norm;
    V.col(0) = v;

    // Lanczos recurrence: beta[j]*v_{j+1} = OP*v_j - alpha[j]*v_j - beta[j-1]*v_{j-1}
    // OP*v = (K-sigma*M)^{-1} * M * v
    double beta_prev = 0.0;

    for (int j = 0; j < ncv; ++j) {
        // Apply OP: w = factored.solve(M * V.col(j))
        Eigen::VectorXd w = factored.solve(Mr * V.col(j));

        // M-inner product: alpha[j] = V.col(j)^T * M * w
        alpha(j) = V.col(j).dot(Mr * w);

        // Orthogonalize
        w -= alpha(j) * V.col(j);
        if (j > 0) w -= beta_prev * V.col(j - 1);

        // Reorthogonalize (once, for numerical stability)
        for (int k = 0; k <= j; ++k)
            w -= V.col(k).dot(Mr * w) * V.col(k);

        // M-norm for next vector
        double beta_j = std::sqrt(w.dot(Mr * w));
        beta(j) = beta_j;

        if (j + 1 < ncv) {
            if (beta_j < 1e-14) {
                // Lucky breakdown — Krylov space exhausted, fill with random M-orthogonal vector
                Eigen::VectorXd r(n);
                for (int i = 0; i < n; ++i) r(i) = dist(rng);
                for (int k = 0; k <= j; ++k)
                    r -= V.col(k).dot(Mr * r) * V.col(k);
                double rn = std::sqrt(r.dot(Mr * r));
                if (rn < 1e-14) break;
                V.col(j + 1) = r / rn;
                beta_prev = 0.0;
            } else {
                V.col(j + 1) = w / beta_j;
                beta_prev = beta_j;
            }
        }
    }

    // Compute eigenvalues of tridiagonal T (alpha on diagonal, beta on off-diagonal)
    // Use Eigen's SelfAdjointEigenSolver on the small tridiagonal matrix
    Eigen::MatrixXd T = Eigen::MatrixXd::Zero(ncv, ncv);
    for (int j = 0; j < ncv; ++j) {
        T(j, j) = alpha(j);
        if (j + 1 < ncv) {
            T(j, j + 1) = beta(j);
            T(j + 1, j) = beta(j);
        }
    }

    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> eig(T);
    if (eig.info() != Eigen::Success)
        throw std::runtime_error("solve_sparse: tridiagonal eigensolve failed");

    // Ritz values (eigenvalues of OP) and Ritz vectors
    // OP eigenvalue mu = 1/(lambda - sigma), so lambda = 1/mu + sigma
    // We want the LARGEST mu (corresponding to SMALLEST lambda)
    Eigen::VectorXd mu = eig.eigenvalues();   // sorted ascending
    Eigen::MatrixXd S  = eig.eigenvectors();  // columns = Ritz vectors in Krylov basis

    EigenResult res;
    // Process from largest mu (smallest lambda) to smallest
    for (int i = ncv - 1; i >= 0 && (int)res.frequencies.size() < num_modes; --i) {
        if (std::abs(mu(i)) < 1e-14) continue;
        double lambda = 1.0 / mu(i) + sigma;
        if (lambda < 0.0) continue;
        double omega = std::sqrt(lambda);
        if (omega < rigid_threshold) { res.num_rigid++; continue; }

        // Ritz vector: x = V * S.col(i)
        Eigen::VectorXd ritz = V * S.col(i);
        std::vector<double> phi(ritz.data(), ritz.data() + n);

        res.frequencies.push_back(omega);
        res.mode_shapes.push_back(std::move(phi));
    }

    // Sort by frequency (ascending)
    std::vector<size_t> idx(res.frequencies.size());
    std::iota(idx.begin(), idx.end(), 0);
    std::sort(idx.begin(), idx.end(),
              [&](size_t a, size_t b){ return res.frequencies[a] < res.frequencies[b]; });

    EigenResult sorted;
    sorted.num_rigid = res.num_rigid;
    for (size_t i : idx) {
        sorted.frequencies.push_back(res.frequencies[i]);
        sorted.mode_shapes.push_back(std::move(res.mode_shapes[i]));
    }
    return sorted;
}

} // namespace modal
