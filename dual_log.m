function dual_log(fid, varargin)
% DUAL_LOG  Mirror a printf-style message to stdout and (if fid > 0) to a
% logfile. Used by the *_new estimation pipeline to keep a persistent log
% of every key event without sprinkling fopen/fclose calls inside the
% optimizers.
%
% USAGE:
%   dual_log(fid, 'Generation %d  fbest = %.4e\n', gen, fbest)
%
% A fid <= 0 disables file output silently.

fprintf(varargin{:});
if fid > 0
    fprintf(fid, varargin{:});
end
end
