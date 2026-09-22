#include "spectral/PhaseUnwrap.h"
#include "render/ShaderUtil.h"
#include "core/Log.h"
#include <string>
#include <vector>
#include <cmath>

namespace Prism {

bool PhaseUnwrap::init() {
    if (m_initialised) return true;
    m_maskGenShader.loadCompute("shaders/unwrap/maskGen.comp");
    if (!m_maskGenShader.isValid()) {
        Log::error("PhaseUnwrap: maskGen.comp failed to compile");
        return false;
    }
    m_wlsResidualShader.loadCompute("shaders/unwrap/wlsResidual.comp");
    if (!m_wlsResidualShader.isValid()) {
        Log::error("PhaseUnwrap: wlsResidual.comp failed to compile");
        return false;
    }
    m_poissonFFTShader.loadCompute("shaders/unwrap/poissonFFT.comp");
    if (!m_poissonFFTShader.isValid()) {
        Log::error("PhaseUnwrap: poissonFFT.comp failed to compile");
        return false;
    }
    // fft1d.comp is compiled lazily in ensureResources() once we know N.
    m_packRealShader.loadCompute("shaders/unwrap/packReal.comp");
    if (!m_packRealShader.isValid()) {
        Log::error("PhaseUnwrap: packReal.comp failed to compile");
        return false;
    }
    m_unpackRealShader.loadCompute("shaders/unwrap/unpackReal.comp");
    if (!m_unpackRealShader.isValid()) {
        Log::error("PhaseUnwrap: unpackReal.comp failed to compile");
        return false;
    }
    m_wlsApplyShader.loadCompute("shaders/unwrap/wlsPCG.comp");
    m_axpbyShader.loadCompute("shaders/unwrap/wlsAxpby.comp");
    m_dotShader.loadCompute("shaders/unwrap/wlsDot.comp");
    m_copyToTexShader.loadCompute("shaders/unwrap/wlsCopyToTex.comp");
    m_demodShader.loadCompute("shaders/unwrap/demodulate.comp");
    for (Shader* s : { &m_wlsApplyShader, &m_axpbyShader, &m_dotShader, &m_copyToTexShader, &m_demodShader }) {
        if (!s->isValid()) { Log::error("PhaseUnwrap: CG shader compile failed"); return false; }
    }
    m_initialised = true;
    Log::info("PhaseUnwrap initialised");
    return true;
}

void PhaseUnwrap::shutdown() {
    auto delTex = [](GLuint& t) { if (t) { glDeleteTextures(1, &t); t = 0; } };
    delTex(m_maskTex);
    delTex(m_unwrappedPhaseTex);
    delTex(m_residueTex);
    delTex(m_branchCutTex);
    delTex(m_depthTex);
    auto delBuf = [](GLuint& b) { if (b) { glDeleteBuffers(1, &b); b = 0; } };
    delBuf(m_ssboRHS);
    delBuf(m_ssboX);
    delBuf(m_ssboR);
    delBuf(m_ssboP);
    delBuf(m_ssboAP);
    delBuf(m_ssboZ);
    delBuf(m_ssboFFTBuf);
    delBuf(m_ssboFFTBufB);
    delBuf(m_ssboDotPartials);
    delBuf(m_ssboDemodU);
    m_initialised = false;
}

void PhaseUnwrap::ensureResources(int N) {
    // Re-compile fft1d.comp whenever the requested N changes. The shader
    // bakes N/LOG2N/N_HALF as #define macros so the SSBO size and workgroup
    // layout match the buffers we're about to allocate below.
    if (m_fft1dLoadedN != N) {
        int log2n = 0; for (int t = N; t > 1; t >>= 1) ++log2n;
        m_fft1dShader.loadComputeWithDefines("shaders/fft1d.comp", {
            {"N", std::to_string(N)},
            {"LOG2N", std::to_string(log2n)},
            {"N_HALF", std::to_string(N / 2)}
        });
        if (m_fft1dShader.isValid()) {
            m_fft1dLoadedN = N;
            Log::info("PhaseUnwrap: fft1d.comp compiled for N={}", N);
        } else {
            // Keep m_fft1dLoadedN at the previous valid value so the N-guard
            // in applyPoissonPrecond catches the mismatch instead of silently
            // running a stale shader against freshly-sized buffers.
            Log::error("PhaseUnwrap: fft1d.comp failed to compile for N={}; keeping N={}",
                       N, m_fft1dLoadedN);
        }
    }

    if (m_N == N && m_maskTex != 0) return;

    auto allocTex = [&](GLuint& id) {
        if (id) glDeleteTextures(1, &id);
        glGenTextures(1, &id);
        glBindTexture(GL_TEXTURE_2D, id);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_R32F, N, N, 0, GL_RED, GL_FLOAT, nullptr);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    };
    allocTex(m_maskTex);
    allocTex(m_unwrappedPhaseTex);
    allocTex(m_residueTex);
    allocTex(m_branchCutTex);
    allocTex(m_depthTex);

    GLsizeiptr bytes = (GLsizeiptr)N * N * sizeof(float);
    auto allocBuf = [&](GLuint& id) {
        if (id) glDeleteBuffers(1, &id);
        glGenBuffers(1, &id);
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, id);
        glBufferData(GL_SHADER_STORAGE_BUFFER, bytes, nullptr, GL_DYNAMIC_COPY);
    };
    allocBuf(m_ssboRHS);
    allocBuf(m_ssboX);
    allocBuf(m_ssboR);
    allocBuf(m_ssboP);
    allocBuf(m_ssboAP);
    allocBuf(m_ssboZ);

    // Complex ping-pong buffers for the Poisson preconditioner.
    // vec2 × N×N, so 2 floats per element.
    GLsizeiptr cbytes = (GLsizeiptr)N * N * 2 * sizeof(float);
    auto allocComplex = [&](GLuint& id) {
        if (id) glDeleteBuffers(1, &id);
        glGenBuffers(1, &id);
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, id);
        glBufferData(GL_SHADER_STORAGE_BUFFER, cbytes, nullptr, GL_DYNAMIC_COPY);
    };
    allocComplex(m_ssboFFTBuf);
    allocComplex(m_ssboFFTBufB);
    allocComplex(m_ssboDemodU);

    // Partial-sums SSBO for block-reduction dot products.
    int numBlocks = (N * N + 255) / 256;
    if (m_ssboDotPartials) glDeleteBuffers(1, &m_ssboDotPartials);
    glGenBuffers(1, &m_ssboDotPartials);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, m_ssboDotPartials);
    glBufferData(GL_SHADER_STORAGE_BUFFER, (GLsizeiptr)numBlocks * sizeof(float), nullptr, GL_DYNAMIC_COPY);

    glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);

    m_N = N;
}

void PhaseUnwrap::applyPoissonPrecond(int N, GLuint srcReal, GLuint dstReal) {
    // fft1d.comp is compiled with the N/LOG2N/N_HALF macros that matched the
    // last ensureResources() call. Using it at a different N would silently
    // corrupt the Poisson preconditioner, so guard against mismatch.
    if (N != m_fft1dLoadedN) {
        Log::error("PhaseUnwrap::applyPoissonPrecond: FFT shader compiled for N={} but N={} requested",
                   m_fft1dLoadedN, N);
        return;
    }
    // NOTE on scaling: fft1d.comp applies 1/N on each inverse pass, so two
    // inverse passes deliver 1/N², which exactly cancels the unnormalised
    // DFT-pair factor of N². The output of the chain is the true Poisson
    // solution x (with mean pinned to zero), no extra scaling required.
    // NOTE on bindings: this routine leaves SSBO slots 0 and 1 bound to
    // scratch buffers; callers must not assume those are clean.
    const int wg = (N + 15) / 16;

    // 1. Pack real → complex into m_ssboFFTBuf.
    m_packRealShader.use();
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, srcReal);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, m_ssboFFTBuf);
    m_packRealShader.setInt("uN", N);
    glDispatchCompute(wg, wg, 1);
    glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);

    // 2a. Forward FFT rows: fftBuf → fftBufB.
    m_fft1dShader.use();
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, m_ssboFFTBuf);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, m_ssboFFTBufB);
    m_fft1dShader.setInt("uInverse", 0);
    m_fft1dShader.setInt("uDirection", 0);
    glDispatchCompute(N, 1, 1);
    glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);

    // 2b. Forward FFT columns: fftBufB → fftBuf.
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, m_ssboFFTBufB);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, m_ssboFFTBuf);
    m_fft1dShader.setInt("uDirection", 1);
    glDispatchCompute(N, 1, 1);
    glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);

    // 3. In-place divide by discrete Laplacian eigenvalues; pin (0,0) to 0.
    m_poissonFFTShader.use();
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, m_ssboFFTBuf);
    m_poissonFFTShader.setInt("uN", N);
    glDispatchCompute(wg, wg, 1);
    glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);

    // 4a. Inverse FFT rows: fftBuf → fftBufB.
    m_fft1dShader.use();
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, m_ssboFFTBuf);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, m_ssboFFTBufB);
    m_fft1dShader.setInt("uInverse", 1);
    m_fft1dShader.setInt("uDirection", 0);
    glDispatchCompute(N, 1, 1);
    glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);

    // 4b. Inverse FFT columns: fftBufB → fftBuf.
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, m_ssboFFTBufB);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, m_ssboFFTBuf);
    m_fft1dShader.setInt("uDirection", 1);
    glDispatchCompute(N, 1, 1);
    glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);

    // 5. Unpack complex → real: take .x into dstReal.
    m_unpackRealShader.use();
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, m_ssboFFTBuf);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, dstReal);
    m_unpackRealShader.setInt("uN", N);
    glDispatchCompute(wg, wg, 1);
    glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
}

void PhaseUnwrap::run(GLuint complexFieldBuf,
                      int    N,
                      float  /*uWavelength*/,
                      float  /*dMin*/,
                      float  /*dMax*/,
                      float  uTiltX,
                      float  uTiltY,
                      const PhaseUnwrapConfig& cfg,
                      PhaseUnwrapOutput& out) {
    if (!m_initialised && !init()) return;
    ensureResources(N);

    const int wg = (N + 15) / 16;

    // Mask — |U| is invariant under multiplication by exp(i·carrier) so the
    // raw IFFT buffer is fine here; no need to demodulate first.
    m_maskGenShader.use();
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, complexFieldBuf);
    glBindImageTexture(0, m_maskTex, 0, GL_FALSE, 0, GL_WRITE_ONLY, GL_R32F);
    m_maskGenShader.setInt("uN", N);
    m_maskGenShader.setFloat("uThreshold", cfg.maskThreshold);
    m_maskGenShader.setInt("uErodeRadius", cfg.maskDilate);  // field is named maskDilate but used as erosion radius
    glDispatchCompute(wg, wg, 1);
    glMemoryBarrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT | GL_TEXTURE_FETCH_BARRIER_BIT);

    // Demodulate the carrier — produces m_ssboDemodU = conj(U) · exp(+i·carrier),
    // which is A·exp(+iφ_object) for inputs of the form A·exp(-iφ_O)·exp(+iφ_R)
    // (what the existing +K-sideband pipeline produces). Skipped if the caller
    // already provides A·exp(+iφ_O) directly (e.g. HoloDepthV2).
    GLuint solverInput = complexFieldBuf;
    if (cfg.demodulateCarrier) {
        m_demodShader.use();
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, complexFieldBuf);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, m_ssboDemodU);
        m_demodShader.setInt("uN", N);
        m_demodShader.setFloat("uTiltX", uTiltX);
        m_demodShader.setFloat("uTiltY", uTiltY);
        glDispatchCompute(wg, wg, 1);
        glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
        solverInput = m_ssboDemodU;
    }

    if (cfg.algorithm == PhaseUnwrapConfig::WLS_PCG) {
        runWLS_PCG(solverInput, N, cfg, out);
    } else {
        runGoldstein(solverInput, N, cfg, out);
    }

    out.maskTex           = m_maskTex;
    out.unwrappedPhaseTex = m_unwrappedPhaseTex;
    out.residueTex        = m_residueTex;
    out.branchCutTex      = m_branchCutTex;
    out.depthTex          = m_depthTex;
    out.elapsedMs         = 0.0f;
}

// y = alpha*a + beta*b  (SSBOs, N*N reals)
void PhaseUnwrap::axpby(int N, float alpha, GLuint aBuf, float beta, GLuint bBuf, GLuint dstBuf) {
    m_axpbyShader.use();
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, aBuf);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, bBuf);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, dstBuf);
    m_axpbyShader.setInt("uN", N * N);
    m_axpbyShader.setFloat("uAlpha", alpha);
    m_axpbyShader.setFloat("uBeta", beta);
    glDispatchCompute((N * N + 255) / 256, 1, 1);
    glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
}

// return <a, b>
double PhaseUnwrap::dot(int N, GLuint aBuf, GLuint bBuf) {
    int numBlocks = (N * N + 255) / 256;
    m_dotShader.use();
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, aBuf);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, bBuf);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, m_ssboDotPartials);
    m_dotShader.setInt("uN", N * N);
    glDispatchCompute(numBlocks, 1, 1);
    glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);

    std::vector<float> partials(numBlocks);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, m_ssboDotPartials);
    glGetBufferSubData(GL_SHADER_STORAGE_BUFFER, 0,
                       (GLsizeiptr)numBlocks * sizeof(float), partials.data());
    double s = 0.0;
    for (float v : partials) s += v;
    return s;
}

// y = A·x  (weighted Laplacian; mask-aware; same mask rule as the right-hand side)
void PhaseUnwrap::applyOperator(int N, GLuint complexBuf, GLuint xBuf, GLuint yBuf) {
    m_wlsApplyShader.use();
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, xBuf);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, complexBuf);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, yBuf);
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, m_maskTex);
    m_wlsApplyShader.setInt("uMaskTex", 0);
    m_wlsApplyShader.setInt("uN", N);
    m_wlsApplyShader.setInt("uUnweighted", m_unweightedOperator ? 1 : 0);
    glDispatchCompute((N + 15) / 16, (N + 15) / 16, 1);
    glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
}

void PhaseUnwrap::runWLS_PCG(GLuint complexFieldBuf, int N,
                             const PhaseUnwrapConfig& cfg,
                             PhaseUnwrapOutput& out) {
    // CONTRACT: applyOperator (wlsPCG.comp) uses the SAME mask-skip rule as
    // wlsResidual.comp — row i skips neighbour j iff mj<0.5, row i is the
    // identity row iff mi<0.5. A and b therefore agree.
    const int wg = (N + 15) / 16;

    // 1. Build RHS b = Σ W·wrap(Δφ) into m_ssboRHS.
    m_wlsResidualShader.use();
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, complexFieldBuf);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, m_ssboRHS);
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, m_maskTex);
    m_wlsResidualShader.setInt("uMaskTex", 0);
    m_wlsResidualShader.setInt("uN", N);
    m_wlsResidualShader.setInt("uUnweighted", cfg.useUnweightedLaplacian ? 1 : 0);
    // Stash the flag so applyOperator (called below on each CG iteration)
    // matches the RHS's weighting choice — A and b must use the same W.
    m_unweightedOperator = cfg.useUnweightedLaplacian;
    glDispatchCompute(wg, wg, 1);
    glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);

    // 2. Initialise x = 0, r = b, z = M⁻¹·r, p = z.
    glClearNamedBufferData(m_ssboX, GL_R32F, GL_RED, GL_FLOAT, nullptr);
    glCopyNamedBufferSubData(m_ssboRHS, m_ssboR, 0, 0, (GLsizeiptr)N * N * sizeof(float));
    applyPoissonPrecond(N, m_ssboR, m_ssboZ);
    glCopyNamedBufferSubData(m_ssboZ, m_ssboP, 0, 0, (GLsizeiptr)N * N * sizeof(float));

    double rzOld    = dot(N, m_ssboR, m_ssboZ);
    double r0normSq = dot(N, m_ssboR, m_ssboR);
    double tol2     = (double)cfg.pcgTolerance * (double)cfg.pcgTolerance * r0normSq;

    int    iter    = 0;
    double rnormSq = r0normSq;
    for (; iter < cfg.pcgIterations; ++iter) {
        if (rnormSq < tol2) break;

        applyOperator(N, complexFieldBuf, m_ssboP, m_ssboAP);
        double pAp = dot(N, m_ssboP, m_ssboAP);
        if (pAp <= 1e-30) break;

        float alpha = (float)(rzOld / pAp);
        axpby(N,  1.0f, m_ssboX,  alpha, m_ssboP,  m_ssboX);      // x += α p
        axpby(N,  1.0f, m_ssboR, -alpha, m_ssboAP, m_ssboR);      // r -= α Ap

        rnormSq = dot(N, m_ssboR, m_ssboR);
        if (rnormSq < tol2) { ++iter; break; }

        applyPoissonPrecond(N, m_ssboR, m_ssboZ);                 // z = M⁻¹ r
        double rzNew = dot(N, m_ssboR, m_ssboZ);
        float beta = (float)(rzNew / rzOld);
        axpby(N, 1.0f, m_ssboZ, beta, m_ssboP, m_ssboP);          // p = z + β p
        rzOld = rzNew;
    }

    // 3. Write x into the unwrapped-phase texture.
    m_copyToTexShader.use();
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, m_ssboX);
    glBindImageTexture(0, m_unwrappedPhaseTex, 0, GL_FALSE, 0, GL_WRITE_ONLY, GL_R32F);
    m_copyToTexShader.setInt("uN", N);
    glDispatchCompute(wg, wg, 1);
    glMemoryBarrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT | GL_TEXTURE_FETCH_BARRIER_BIT);

    out.iterations = iter;
    // Guard against r0normSq ≈ 0 (empty mask / black scene) producing a
    // garbage huge ratio; explicit zero is the meaningful sentinel.
    out.residual = (r0normSq > 1e-20)
        ? (float)std::sqrt(rnormSq / r0normSq)
        : 0.0f;
}
void PhaseUnwrap::runGoldstein(GLuint, int, const PhaseUnwrapConfig&, PhaseUnwrapOutput&) {}
void PhaseUnwrap::runPostAndDepth(int, float, float, float,
                                  const PhaseUnwrapConfig&, PhaseUnwrapOutput&) {}

} // namespace Prism
