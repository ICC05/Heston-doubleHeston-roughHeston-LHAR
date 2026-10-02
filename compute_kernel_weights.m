function c = compute_kernel_weights(gammas, alpha, T)
% COMPUTE_KERNEL_WEIGHTS  L^2-optimal weights for the multifactor approximation
% of the rough Heston kernel K(tau) = tau^(alpha-1) / Gamma(alpha).
%
% The approximation K^N(tau) = sum_n c_n * exp(-gamma_n * tau) is fitted by
% minimizing  || K - K^N ||^2_{L^2((0,T))}  in c, subject to c >= 0.
% This is a linear least-squares problem in c with explicit Gram matrix and
% right-hand-side. Non-negativity is enforced through MATLAB's lsqnonneg.
%
% USAGE:
%   c = compute_kernel_weights(gammas, alpha, T)
%
% INPUTS:
%   gammas - (N x 1) vector of positive nodes gamma_n in the geometric grid
%   alpha  - kernel exponent in (1/2, 1); related to the Hurst exponent by
%            H = alpha - 1/2
%   T      - upper limit of the L^2 fit (in days). T should cover the longest
%            kernel-induced memory effect that matters for the auxiliary
%            criterion. T = 2 * monthly horizon (=44 days) is conservative.
%
% OUTPUT:
%   c - (N x 1) non-negative weights c_n
%
% Reference: Abi Jaber and El Euch (2019), Multifactor Approximation of Rough
% Volatility Models, SIAM J. Financial Math.

gammas = gammas(:);
N = length(gammas);

% Gram matrix: A_ij = int_0^T exp(-(gamma_i + gamma_j) tau) dtau
%                  = (1 - exp(-(gamma_i + gamma_j) T)) / (gamma_i + gamma_j)
GamSum = gammas + gammas.';
A = (1 - exp(-GamSum * T)) ./ GamSum;

% Right-hand-side: b_i = int_0^T tau^(alpha-1)/Gamma(alpha) * exp(-gamma_i tau) dtau
% Substitution u = gamma_i tau gives
%   b_i = gamma_i^(-alpha) * gammainc(gamma_i T, alpha)
% where MATLAB's gammainc is the regularized lower incomplete gamma function,
% i.e. gammainc(x, a) = (1/Gamma(a)) * int_0^x u^(a-1) exp(-u) du.
b = gammas.^(-alpha) .* gammainc(gammas * T, alpha);

% Solve with non-negativity (typically all c_n > 0 already, but lsqnonneg
% protects against ill-conditioned high-N geometric grids)
c = lsqnonneg(A, b);

end
