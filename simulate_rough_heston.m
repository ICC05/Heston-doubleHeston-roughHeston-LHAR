function r5min = simulate_rough_heston(ndays_eval, ndays_burn, M_fine, agg_factor, params, kernel_table, w_var, w_perp)
% SIMULATE_ROUGH_HESTON  Forward simulate the rough Heston model under P
% via the multifactor Markovian approximation of Abi Jaber and El Euch (2019),
% with full-truncation positivity correction (Lord, Koekkoek and van Dijk,
% 2010), explicit Euler-Maruyama on the joint (V, U_1, ..., U_N) system,
% exponential price update, and a non-Markovian burn-in window.
%
% The function follows Section 7.2.1-7.2.3 of the thesis:
%   - kernel approximation K^N with N = length(kernel_table.gammas) factors
%   - geometric node grid {gamma_n} held fixed across calls; weights c_n
%     are interpolated from the precomputed kernel_table at the candidate H
%   - simulation grid Delta t = 1 / M_fine days, satisfying the stability
%     constraint gamma_max * Delta t < 1 imposed by the explicit Euler step
%   - 5-minute returns are obtained by aggregating every agg_factor fine
%     intervals, so that 80 = M_fine / agg_factor 5-minute bins per day are
%     produced, matching the empirical RV construction
%   - the first ndays_burn days are discarded after the simulation, so that
%     the initialization U_n,0 = 0 does not contaminate the auxiliary fit.
%
% USAGE:
%   r5min = simulate_rough_heston(ndays_eval, ndays_burn, M_fine, agg_factor, ...
%                                  params, kernel_table, w_var, w_perp)
%
% INPUTS:
%   ndays_eval    - number of evaluation trading days (e.g. 6000)
%   ndays_burn    - number of burn-in days (e.g. 1000)
%   M_fine        - fine-grid intra-day intervals (e.g. 400)
%   agg_factor    - aggregation factor to obtain the 5-minute grid (e.g. 5)
%   params        - struct with fields:
%                     .lambda - mean-reversion speed (1/day)
%                     .theta  - long-run variance level (daily, dimensionless)
%                     .nu     - volatility-of-variance
%                     .rho    - leverage correlation in [-1, 1]
%                     .H      - Hurst exponent in (0, 0.5)
%                     .mu     - drift (typically 0)
%   kernel_table  - struct from precompute_kernel_table; supplies
%                     .gammas    - (N x 1) geometric node grid
%                     .interpFun - cubic interpolants for c_n vs H
%   w_var         - (M_fine x ndays_total) standard normals; variance shock
%                   shared across the N OU factors at each fine step
%   w_perp        - (M_fine x ndays_total) standard normals; orthogonal
%                   component used to build the price-driving shock
%                   Z = rho*w_var + sqrt(1-rho^2)*w_perp
%
% OUTPUT:
%   r5min - (M5 x ndays_eval) matrix of 5-minute log returns, where
%           M5 = M_fine / agg_factor, ready for the same RV / HAR-RV / LHAR
%           construction used on the empirical sample.

% --------------------- unpack ---------------------
lambda = params.lambda;
theta  = params.theta;
nu     = params.nu;
rho    = params.rho;
H      = params.H;
mu     = params.mu;

gammas = kernel_table.gammas(:);
N      = length(gammas);

% Recover the c_n weights at the candidate H via the precomputed table
c = interp_kernel_weights(kernel_table, H);

% --------------------- geometry ---------------------
ndays_total = ndays_eval + ndays_burn;
dt   = 1 / M_fine;            % fine timestep in days
nstep = M_fine * ndays_total; % total number of fine timesteps
M5   = M_fine / agg_factor;   % number of 5-minute bins per day

if M5 ~= round(M5)
    error('simulate_rough_heston:bad_aggregation', ...
        'M_fine (%d) must be divisible by agg_factor (%d).', M_fine, agg_factor);
end

% --------------------- preallocate ---------------------
% U: (N x 1) auxiliary OU factors, started at zero
U  = zeros(N, 1);
% V_t = V0 + sum_n c_n U_n   (with V0 = theta)
V0 = theta;
V  = V0;
% Vectorize the variance shock and orthogonal shock for sequential access
w_var_vec  = double(reshape(w_var,  1, nstep));
w_perp_vec = double(reshape(w_perp, 1, nstep));

% Buffers for fine-grid log-returns; only the evaluation portion is stored
% as 5-minute aggregates to save memory.
r5min = zeros(M5, ndays_eval);

% Helpers for stage assignment
nstep_burn = M_fine * ndays_burn;

% Per-step Euler coefficients that are constant over the path
one_minus_gdt = 1 - gammas * dt;     % (1 - gamma_n * dt) for each n
% --------------------- main recursion ---------------------
% Within evaluation, accumulate fine returns into the current 5-minute bin.
% After each agg_factor fine steps, the 5-min return is finalized.
% After M_fine fine steps, we have a complete day; reset bin index.

bin_idx     = 1;     % which 5-minute bin within the day (1..M5)
day_eval    = 0;     % how many evaluation days completed (0..ndays_eval)
fine_in_bin = 0;     % how many fine steps accumulated in the current bin
sum_in_bin  = 0;     % running sum of fine log-returns in the current bin
in_eval     = false;

for k = 1:nstep
    % Has the burn-in ended at this step?
    if ~in_eval && k > nstep_burn
        in_eval = true;
        bin_idx     = 1;
        fine_in_bin = 0;
        sum_in_bin  = 0;
        day_eval    = 0;
    end

    Vplus = max(V, 0);
    sqVdt = sqrt(Vplus * dt);

    % Variance shock (shared across factors) and orthogonal shock
    n_var  = w_var_vec(k);
    n_perp = w_perp_vec(k);

    % Update each auxiliary factor U_n
    drift_common = lambda * (theta - V) * dt;
    diff_common  = lambda * nu * sqVdt * n_var;
    U = one_minus_gdt .* U + drift_common + diff_common;

    % Reconstruct total variance
    V = V0 + c' * U;

    % Build the price-driving Brownian shock and propagate the log-price
    Z = rho * n_var + sqrt(1 - rho^2) * n_perp;
    dx = (mu - 0.5 * Vplus) * dt + sqVdt * Z;

    % If we are in the evaluation regime, accumulate into the current 5-min bin
    if in_eval
        sum_in_bin  = sum_in_bin + dx;
        fine_in_bin = fine_in_bin + 1;

        if fine_in_bin == agg_factor
            % Finalize this 5-minute bin
            r5min(bin_idx, day_eval + 1) = sum_in_bin;
            bin_idx     = bin_idx + 1;
            fine_in_bin = 0;
            sum_in_bin  = 0;

            if bin_idx > M5
                % End of day
                bin_idx  = 1;
                day_eval = day_eval + 1;
                if day_eval >= ndays_eval
                    return
                end
            end
        end
    end
end

end
