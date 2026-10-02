function c = interp_kernel_weights(kernel_table, H)
% INTERP_KERNEL_WEIGHTS  Recover c_n at the candidate H via the pre-tabulated
% cubic interpolation table built by precompute_kernel_table.

N = length(kernel_table.interpFun);
c = zeros(N, 1);
for n = 1:N
    c(n) = kernel_table.interpFun{n}(H);
end
% Safety: enforce non-negativity (cubic interpolation could in principle
% produce small negative values near the boundary)
c = max(c, 0);

end
