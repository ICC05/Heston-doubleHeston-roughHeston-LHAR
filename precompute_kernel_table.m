function kernel_table = precompute_kernel_table(gammas, H_grid, T_kernel)
% PRECOMPUTE_KERNEL_TABLE  Pre-tabulate the L^2-optimal kernel weights c_n
% over a fine grid of Hurst exponents H, so that, during the indirect
% inference loop, the weights at the candidate H can be recovered through
% cubic interpolation rather than re-solved at every objective evaluation.
%
% This preserves the differentiability of the binding function in H and
% avoids spurious irregularities of the objective surface induced by the
% kernel re-fitting (Section 7.2.3 of the thesis).
%
% USAGE:
%   kernel_table = precompute_kernel_table(gammas, H_grid, T_kernel)
%
% INPUTS:
%   gammas   - (N x 1) geometric grid of positive nodes
%   H_grid   - vector of Hurst values at which to tabulate (e.g. 0.05:0.005:0.49)
%   T_kernel - upper limit of the L^2 fit (in days)
%
% OUTPUT:
%   kernel_table - struct with fields:
%     .gammas    - the geometric grid (passed through)
%     .H_grid    - the H grid (passed through)
%     .T_kernel  - the L^2 horizon (passed through)
%     .C         - (N x length(H_grid)) matrix of c_n weights
%     .interpFun - griddedInterpolant handles, one per node, for cubic interp

gammas = gammas(:);
H_grid = H_grid(:)';
N = length(gammas);
nH = length(H_grid);

C = zeros(N, nH);
for k = 1:nH
    H_k = H_grid(k);
    alpha_k = H_k + 0.5;
    C(:, k) = compute_kernel_weights(gammas, alpha_k, T_kernel);
end

% Build a vector of cubic interpolants, one per node
interpFun = cell(N, 1);
for n = 1:N
    interpFun{n} = griddedInterpolant(H_grid, C(n, :), 'pchip', 'nearest');
end

kernel_table.gammas    = gammas;
kernel_table.H_grid    = H_grid;
kernel_table.T_kernel  = T_kernel;
kernel_table.C         = C;
kernel_table.interpFun = interpFun;

end
