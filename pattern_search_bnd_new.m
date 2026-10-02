function [xbest, fbest, info] = pattern_search_bnd_new(objfun, x0, LB, UB, opts)
% PATTERN_SEARCH_BND  Bound-constrained Generalized Pattern Search
% (Audet and Dennis, 2002) with the standard 2d positive basis,
% Latin-Hypercube polls in the search step, relative-improvement
% early stopping, and dual logging. Used in Stage 2 of the improved
% Heston / Double Heston pipeline.
%
% Key features:
%   - 2d positive basis (+e_i and -e_i) for guaranteed convergence on
%     bound-constrained problems.
%   - LHS poll step after every unsuccessful pattern step, controlled by
%     lhs_polls (default 4): probes random directions inside the current
%     mesh to escape narrow ridges.
%   - Early stopping if best-so-far has not improved by more than
%     rel_stall_tol (default 0.5%) over stall_iters (default 20)
%     consecutive iterations. This avoids burning evaluations once chi^2
%     has plateaued, which is the regime that consumed every Stage-2
%     start to its 150-iter cap in the rough Heston legacy run.
%   - All progress printouts go through dual_log_new.
%
% USAGE:
%   [xbest, fbest, info] = pattern_search_bnd_new(objfun, x0, LB, UB, opts)
%
% INPUTS:
%   objfun        - function handle: f = objfun(x)
%   x0            - initial point (1 x d)
%   LB, UB        - (1 x d) bounds
%   opts          - struct with optional fields:
%                     .init_mesh      initial mesh (default 0.05*(UB-LB))
%                     .min_mesh       termination mesh (default 1e-5*(UB-LB))
%                     .expand         expansion factor (default 2.0)
%                     .contract       contraction factor (default 0.5)
%                     .max_iter       iteration cap (default 300)
%                     .max_feval      function-eval cap (default 1500)
%                     .lhs_polls      LHS poll count (default 4)
%                     .rel_stall_tol  relative improvement threshold
%                                     (default 0.005 = 0.5%)
%                     .stall_iters    stall window (default 20)
%                     .verbose        print progress (default true)
%                     .log_fid        file handle for dual_log_new (default -1)

if nargin < 5, opts = struct(); end

x0 = x0(:)';
LB = LB(:)'; UB = UB(:)';
d  = length(x0);
range = UB - LB;

init_mesh    = getfield_default(opts, 'init_mesh', 0.05 * range);
min_mesh     = getfield_default(opts, 'min_mesh',  1e-5 * range);
expand       = getfield_default(opts, 'expand', 2.0);
contract     = getfield_default(opts, 'contract', 0.5);
max_iter     = getfield_default(opts, 'max_iter', 300);
max_feval    = getfield_default(opts, 'max_feval', 1500);
lhs_polls    = getfield_default(opts, 'lhs_polls', 4);
verbose      = getfield_default(opts, 'verbose', true);
rel_stall_tol = getfield_default(opts, 'rel_stall_tol', 0.005);
stall_iters  = getfield_default(opts, 'stall_iters', 20);
log_fid      = getfield_default(opts, 'log_fid', -1);

x = min(max(x0, LB), UB);
mesh = init_mesh;

try
    f = objfun(x);
catch
    f = 1e20;
end
nFEval = 1;

xbest = x;
fbest = f;

last_sig_fbest = fbest;
last_sig_iter  = 0;

basis = [eye(d); -eye(d)];

if verbose
    dual_log_new(log_fid, 'Pattern search: d=%d, init mesh = ', d);
    dual_log_new(log_fid, '%9.4e ', mesh);
    dual_log_new(log_fid, '\n');
end

iter = 0;
while iter < max_iter && nFEval < max_feval
    iter = iter + 1;

    success = false;
    for k = 1:size(basis, 1)
        if nFEval >= max_feval, break; end
        x_trial = x + basis(k, :) .* mesh;
        x_trial = min(max(x_trial, LB), UB);
        try
            f_trial = objfun(x_trial);
        catch
            f_trial = 1e20;
        end
        nFEval = nFEval + 1;
        if f_trial < fbest
            fbest = f_trial;
            xbest = x_trial;
            success = true;
            break
        end
    end

    if success
        x = xbest;
        mesh = expand * mesh;
        mesh = min(mesh, init_mesh);
    else
        for kk = 1:lhs_polls
            if nFEval >= max_feval, break; end
            r = (rand(1, d) - 0.5) * 2;
            x_trial = x + r .* mesh;
            x_trial = min(max(x_trial, LB), UB);
            try
                f_trial = objfun(x_trial);
            catch
                f_trial = 1e20;
            end
            nFEval = nFEval + 1;
            if f_trial < fbest
                fbest = f_trial;
                xbest = x_trial;
                success = true;
            end
        end

        if success
            x = xbest;
            mesh = expand * mesh;
            mesh = min(mesh, init_mesh);
        else
            mesh = contract * mesh;
        end
    end

    if fbest < last_sig_fbest * (1 - rel_stall_tol)
        last_sig_fbest = fbest;
        last_sig_iter  = iter;
    end

    if verbose && (mod(iter, 10) == 0 || iter == 1)
        dual_log_new(log_fid, 'PS iter %3d: fbest = %12.6e, mesh_max = %10.4e, nFEval = %d\n', ...
            iter, fbest, max(mesh), nFEval);
    end

    if all(mesh < min_mesh)
        if verbose
            dual_log_new(log_fid, 'PS stop: mesh below min_mesh at iter %d (nFEval=%d)\n', iter, nFEval);
        end
        break
    end
    if (iter - last_sig_iter) >= stall_iters
        if verbose
            dual_log_new(log_fid, ['PS stop: no relative improvement of %.2f%% in %d iterations ', ...
                'at iter %d (nFEval=%d, fbest=%.6e)\n'], ...
                rel_stall_tol*100, stall_iters, iter, nFEval, fbest);
        end
        break
    end
end

info.nFEval        = nFEval;
info.iter          = iter;
info.final_mesh    = mesh;
info.last_sig_iter = last_sig_iter;
info.last_sig_fbest = last_sig_fbest;

end


function val = getfield_default(s, field, default)
    if isfield(s, field)
        val = s.(field);
    else
        val = default;
    end
end
