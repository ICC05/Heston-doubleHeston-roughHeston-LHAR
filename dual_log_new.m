function dual_log_new(fid, varargin)
% DUAL_LOG  Mirror a printf-style message to stdout and (if fid > 0) to a
% logfile. Used by the improved estimation pipeline to keep a persistent
% log of every key event without sprinkling fopen/fclose calls inside
% the optimizers (which in the legacy MSc 2026 code resulted in empty
% log files).
%
% USAGE:
%   dual_log_new(fid, 'Generation %d  fbest = %.4e\n', gen, fbest)
%
% A fid <= 0 disables file output silently.

fprintf(varargin{:});
if fid > 0
    fprintf(fid, varargin{:});
end
end
