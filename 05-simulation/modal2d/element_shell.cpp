#include "fem/element_shell.h"
#include "fem/element_cst.h"
#include "fem/element_dkt.h"
#include "fem/element_mindlin.h"
#include <cmath>

namespace modal {

// Local DOF mapping from CST / DKT element ordering to the 15-DOF
// per-node-interleaved shell ordering (u_x, u_y, w, theta_x, theta_y).
//   cst_to_shell[k] for k in [0..5] (k = 2*node + d, d in {0,1})
//     -> shell local index 5*node + d
//   dkt_to_shell[k] for k in [0..8] (k = 3*node + d, d in {0,1,2})
//     -> shell local index 5*node + 2 + d
static const int kCstToShell[6] = { 0, 1,    5, 6,   10, 11 };
static const int kDktToShell[9] = { 2, 3, 4, 7, 8, 9, 12, 13, 14 };

void shell_element_matrices(
    const std::array<std::array<float,2>,3>& nodes,
    const std::array<float,3>&                z_values,
    float E,
    float nu,
    float rho,
    float thickness,
    Eigen::Matrix<double,15,15>&              Ke,
    Eigen::Matrix<double,15,15>&              Me,
    bool                                      thick_bending)
{
    Eigen::Matrix<double,6,6> K_m, M_m;
    cst_element_matrices(nodes, E, nu, rho, thickness, K_m, M_m);

    // Bending block: thin shells use DKT (unchanged); thick shells use the
    // Mindlin transverse-shear element (v11.1 augment).
    const double t = (double)thickness;
    Eigen::Matrix<double,9,9> K_b, M_b;
    if (thick_bending) {
        mindlin_element_matrices(nodes, (double)E, (double)nu, (double)rho, t, K_b, M_b);
    } else {
        const double D = (double)E * t * t * t / (12.0 * (1.0 - (double)nu * (double)nu));
        dkt_element_matrices(nodes, D, (double)rho, t, K_b, M_b);
    }

    Ke.setZero();
    Me.setZero();

    // Scatter CST membrane block into the shell-local membrane DOFs and
    // DKT bending block into the shell-local bending DOFs.
    for (int i = 0; i < 6; ++i)
        for (int j = 0; j < 6; ++j) {
            Ke(kCstToShell[i], kCstToShell[j]) += K_m(i, j);
            Me(kCstToShell[i], kCstToShell[j]) += M_m(i, j);
        }
    for (int i = 0; i < 9; ++i)
        for (int j = 0; j < 9; ++j) {
            Ke(kDktToShell[i], kDktToShell[j]) += K_b(i, j);
            Me(kDktToShell[i], kDktToShell[j]) += M_b(i, j);
        }

    // v11.0 T2: Marguerre bending-stretching coupling block K_mb.
    //
    // For a linear interpolation of z(x, y) across the triangle, dz/dx
    // and dz/dy are constants over the element. The CST shape-function
    // gradient operators b_i, c_i (the same ones B_m uses for membrane
    // strain) give:
    //   2A = (x1-x0)(y2-y0) - (x2-x0)(y1-y0)   (signed; positive CCW)
    //   b_i = (y_j - y_k) / (2A)
    //   c_i = (x_k - x_j) / (2A)
    //   dz/dx = sum_i b_i * z_i
    //   dz/dy = sum_i c_i * z_i
    //
    // The Marguerre linear-membrane strain gains (small-amplitude limit
    // of the von Karman terms; see Cook-Malkus-Plesha s13):
    //   eps_xx += (dz/dx)(dw/dx)
    //   eps_yy += (dz/dy)(dw/dy)
    //   gam_xy += (dz/dx)(dw/dy) + (dz/dy)(dw/dx)
    //
    // Written compactly with grad_w = (dw/dx, dw/dy)^T:
    //   Marg_strain = Sz * grad_w
    // where:
    //   Sz[3x2] = [ dz/dx   0     ]
    //             [   0   dz/dy   ]
    //             [ dz/dy  dz/dx  ]
    //
    // The element coupling-stiffness contribution is:
    //   K_mb = integral over element of (B_m^T * D_m * Sz * G_w) dV
    //
    // where B_m is the CST membrane strain-displacement operator (3x6),
    // D_m is the plane-stress constitutive (3x3), and G_w is the 2x9
    // out-of-plane gradient operator (acting on the DKT-local DOFs
    // ordered (w_0, theta_x0, theta_y0, w_1, ...)).
    //
    // For the shallow-shell approximation in v11.0 we use CST-linear
    // b_i / c_i for the w-gradient as well (i.e., G_w only carries w
    // DOFs, not theta). This is sufficient for the regimes the spec
    // targets (cymbals, gongs, bowls); the full DKT shape-function
    // derivatives can be wired in later if accuracy demands.
    //
    // Constant integrand over the (linear) element, so the integral
    // collapses to area * integrand.
    const double x0d = (double)nodes[0][0], y0d = (double)nodes[0][1];
    const double x1d = (double)nodes[1][0], y1d = (double)nodes[1][1];
    const double x2d = (double)nodes[2][0], y2d = (double)nodes[2][1];
    const double two_A = (x1d - x0d) * (y2d - y0d) - (x2d - x0d) * (y1d - y0d);
    if (std::abs(two_A) < 1e-20) return;  // degenerate triangle; leave K_mb = 0
    const double inv2A = 1.0 / two_A;
    const double b[3] = {
        (y1d - y2d) * inv2A,
        (y2d - y0d) * inv2A,
        (y0d - y1d) * inv2A
    };
    const double c[3] = {
        (x2d - x1d) * inv2A,
        (x0d - x2d) * inv2A,
        (x1d - x0d) * inv2A
    };
    const double dz_dx = b[0] * (double)z_values[0]
                       + b[1] * (double)z_values[1]
                       + b[2] * (double)z_values[2];
    const double dz_dy = c[0] * (double)z_values[0]
                       + c[1] * (double)z_values[1]
                       + c[2] * (double)z_values[2];
    if (std::abs(dz_dx) < 1e-12 && std::abs(dz_dy) < 1e-12) return;

    // Plane-stress constitutive (isotropic):
    //   D_m = (E*t)/(1-nu^2) * [ 1 nu 0 ; nu 1 0 ; 0 0 (1-nu)/2 ]
    const double E_d  = (double)E;
    const double nu_d = (double)nu;
    const double t_d  = (double)thickness;
    const double k    = E_d * t_d / (1.0 - nu_d * nu_d);
    const double Dm[3][3] = {
        { k,        k * nu_d,                 0.0           },
        { k * nu_d, k,                        0.0           },
        { 0.0,      0.0,        k * (1.0 - nu_d) * 0.5      }
    };

    // CST strain-displacement B_m (3x6) for in-plane membrane DOFs
    // ordered [u0, v0, u1, v1, u2, v2]:
    //   row 0 (eps_xx) = [ b0, 0, b1, 0, b2, 0 ]
    //   row 1 (eps_yy) = [ 0, c0, 0, c1, 0, c2 ]
    //   row 2 (gam_xy) = [ c0, b0, c1, b1, c2, b2 ]
    const double Bm[3][6] = {
        { b[0], 0.0,  b[1], 0.0,  b[2], 0.0  },
        { 0.0,  c[0], 0.0,  c[1], 0.0,  c[2] },
        { c[0], b[0], c[1], b[1], c[2], b[2] }
    };

    // Sz[3x2] -- Marguerre strain-gradient mapping (rows = strain
    // components, cols = grad_w components).
    const double Sz[3][2] = {
        { dz_dx, 0.0   },
        { 0.0,   dz_dy },
        { dz_dy, dz_dx }
    };

    // G_w[2x9] -- CST-linear out-of-plane gradient (acts on DKT-local
    // DOFs (w0, theta_x0, theta_y0, w1, theta_x1, theta_y1, w2, theta_x2, theta_y2)).
    // theta DOFs contribute zero in this shallow-shell approximation.
    const double Gw[2][9] = {
        { b[0], 0.0, 0.0, b[1], 0.0, 0.0, b[2], 0.0, 0.0 },
        { c[0], 0.0, 0.0, c[1], 0.0, 0.0, c[2], 0.0, 0.0 }
    };

    const double area = std::abs(two_A) * 0.5;

    // K_mb[6x9] = (B_m^T D_m Sz G_w) * area  -- the membrane-bending
    // cross-coupling. Built explicitly so the loop nest is unambiguous.
    double Kmb[6][9];
    for (int i = 0; i < 6; ++i) {
        for (int j = 0; j < 9; ++j) {
            double s = 0.0;
            for (int r = 0; r < 3; ++r)
                for (int p = 0; p < 3; ++p)
                    for (int q = 0; q < 2; ++q)
                        s += Bm[r][i] * Dm[r][p] * Sz[p][q] * Gw[q][j];
            Kmb[i][j] = s * area;
        }
    }

    // K_bb_extra[9x9] = (G_w^T (Sz^T D Sz) G_w) * area  -- the extra
    // bending stiffness from Marguerre membrane strain acting on pure
    // transverse motion. WITHOUT this term, K_mb cross-coupling alone
    // can let mixed bending-membrane modes drop BELOW their flat-plate
    // counterparts -- physically incorrect. Adding K_bb_extra restores
    // the expected "curvature stiffens the shell" behaviour.
    //
    // T = Sz^T D Sz  (2x2)
    double T[2][2] = {{0.0, 0.0}, {0.0, 0.0}};
    for (int r = 0; r < 2; ++r)
        for (int s = 0; s < 2; ++s)
            for (int p = 0; p < 3; ++p)
                for (int pp = 0; pp < 3; ++pp)
                    T[r][s] += Sz[p][r] * Dm[p][pp] * Sz[pp][s];

    double Kbb_ext[9][9];
    for (int i = 0; i < 9; ++i) {
        for (int j = 0; j < 9; ++j) {
            double s = 0.0;
            for (int r = 0; r < 2; ++r)
                for (int ss = 0; ss < 2; ++ss)
                    s += Gw[r][i] * T[r][ss] * Gw[ss][j];
            Kbb_ext[i][j] = s * area;
        }
    }

    // Scatter K_mb into Ke at (kCstToShell[i], kDktToShell[j]) and the
    // symmetric transpose, and K_bb_extra into the bending block.
    for (int i = 0; i < 6; ++i)
        for (int j = 0; j < 9; ++j) {
            double v = Kmb[i][j];
            Ke(kCstToShell[i], kDktToShell[j]) += v;
            Ke(kDktToShell[j], kCstToShell[i]) += v;
        }
    for (int i = 0; i < 9; ++i)
        for (int j = 0; j < 9; ++j)
            Ke(kDktToShell[i], kDktToShell[j]) += Kbb_ext[i][j];
}

} // namespace modal
