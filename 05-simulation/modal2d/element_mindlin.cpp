#include "fem/element_mindlin.h"
#include <cmath>
#include <algorithm>

namespace modal {

bool plate_is_thick(const std::vector<std::array<float,2>>& nodes, float thickness)
{
    if (nodes.empty() || thickness <= 0.0f) return false;
    float xmin = 1e30f, xmax = -1e30f, ymin = 1e30f, ymax = -1e30f;
    for (const auto& n : nodes) {
        xmin = std::min(xmin, n[0]); xmax = std::max(xmax, n[0]);
        ymin = std::min(ymin, n[1]); ymax = std::max(ymax, n[1]);
    }
    const float span = std::max(xmax - xmin, ymax - ymin);
    return span > 0.0f && (thickness / span) >= kThickShearRatio;
}

void mindlin_element_matrices(
    const std::array<std::array<float,2>,3>& nodes,
    double E, double nu, double rho, double thickness,
    Eigen::Matrix<double,9,9>& Ke,
    Eigen::Matrix<double,9,9>& Me)
{
    Ke.setZero();
    Me.setZero();

    const double x1 = nodes[0][0], y1 = nodes[0][1];
    const double x2 = nodes[1][0], y2 = nodes[1][1];
    const double x3 = nodes[2][0], y3 = nodes[2][1];

    // Signed 2A (positive CCW); area-coordinate gradients b_i=dL_i/dx, c_i=dL_i/dy.
    const double two_A = (x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1);
    const double A = 0.5 * std::abs(two_A);
    if (A < 1e-20) return;  // degenerate triangle
    const double inv2A = 1.0 / two_A;
    const double b[3] = { (y2 - y3) * inv2A, (y3 - y1) * inv2A, (y1 - y2) * inv2A };
    const double c[3] = { (x3 - x2) * inv2A, (x1 - x3) * inv2A, (x2 - x1) * inv2A };

    const double h = thickness;
    const double D = E * h * h * h / (12.0 * (1.0 - nu * nu));

    // ---- Bending: linear rotations -> constant curvature.
    // DOFs per node i: col 3*i+0 = w_i, 3*i+1 = theta_x_i, 3*i+2 = theta_y_i.
    // beta = (beta_x, beta_y) = (theta_y, -theta_x):
    //   kappa_xx =  sum b_i*theta_y_i
    //   kappa_yy = -sum c_i*theta_x_i
    //   kappa_xy =  sum( c_i*theta_y_i - b_i*theta_x_i )
    Eigen::Matrix<double,3,9> Bb; Bb.setZero();
    for (int i = 0; i < 3; ++i) {
        Bb(0, 3*i + 2) =  b[i];     // kappa_xx <- theta_y
        Bb(1, 3*i + 1) = -c[i];     // kappa_yy <- theta_x
        Bb(2, 3*i + 1) = -b[i];     // kappa_xy <- theta_x
        Bb(2, 3*i + 2) =  c[i];     // kappa_xy <- theta_y
    }
    Eigen::Matrix3d Db;
    Db << D,     D*nu,  0.0,
          D*nu,  D,     0.0,
          0.0,   0.0,   D*(1.0 - nu) * 0.5;
    Ke += A * (Bb.transpose() * Db * Bb);

    // ---- Transverse shear: standard linear shear strain (rank-sufficient),
    // integrated at 3 Gauss points, with Lyly-Stenberg-Vihinen stabilization
    // to remove shear locking. A pure DSG (constant-gamma) shear is only
    // rank 2 which, with the rank-3 bending, leaves the 9-DOF element rank 5
    // (1 spurious zero-energy mode). The linear shear gamma = grad(w) + beta
    // (beta_x=theta_y, beta_y=-theta_x; beta linear via N_i=L_i) is
    // rank-sufficient; the stabilization factor h^2/(h^2 + alpha*hK^2) keeps
    // the shear stiffness from over-constraining (locking) as h -> 0.
    const double G     = E / (2.0 * (1.0 + nu));
    const double kappa = 5.0 / 6.0;
    const double e0 = std::hypot(x2 - x1, y2 - y1);
    const double e1 = std::hypot(x3 - x2, y3 - y2);
    const double e2 = std::hypot(x1 - x3, y1 - y3);
    const double hK = std::max(e0, std::max(e1, e2));   // element size
    const double alpha = 0.1;
    const double Ds = kappa * G * h * (h * h) / (h * h + alpha * hK * hK);
    static const double gp[3][3] = {
        {2.0/3, 1.0/6, 1.0/6}, {1.0/6, 2.0/3, 1.0/6}, {1.0/6, 1.0/6, 2.0/3} };
    for (int q = 0; q < 3; ++q) {
        const double N[3] = { gp[q][0], gp[q][1], gp[q][2] };
        Eigen::Matrix<double,2,9> Bs; Bs.setZero();
        for (int i = 0; i < 3; ++i) {
            Bs(0, 3*i + 0) = b[i];     // gamma_xz <- w  (dw/dx)
            Bs(0, 3*i + 2) = N[i];     // gamma_xz <- theta_y (= beta_x)
            Bs(1, 3*i + 0) = c[i];     // gamma_yz <- w  (dw/dy)
            Bs(1, 3*i + 1) = -N[i];    // gamma_yz <- theta_x (beta_y = -theta_x)
        }
        Ke += (A / 3.0) * Ds * (Bs.transpose() * Bs);
    }

    // ---- Lumped mass (matches element_dkt: rho*h*A/3 on w, rho*h^3/12*A/3 on rotations).
    const double m_w  = rho * h * A / 3.0;
    const double m_th = rho * h * h * h / 12.0 * A / 3.0;
    for (int i = 0; i < 3; ++i) {
        Me(3*i,     3*i)     = m_w;
        Me(3*i + 1, 3*i + 1) = m_th;
        Me(3*i + 2, 3*i + 2) = m_th;
    }
}

} // namespace modal
