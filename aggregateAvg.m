function [avg_trimmed, avg_full] = aggregateAvg(x, freq)
% AGGREGATEAVG  Compute backward-looking rolling average over freq periods
%
% USAGE: [avg_trimmed, avg_full] = aggregateAvg(x, freq)
%
% INPUTS:
%   x    - (T x 1) column vector
%   freq - averaging window length (1, 5, or 22 for HAR)
%
% OUTPUTS:
%   avg_trimmed - ((T-freq+1) x 1) trimmed to valid entries
%   avg_full    - (T x 1) full-length with NaN for initial entries

T = length(x);
avg_full = NaN(T, 1);

if freq == 1
    avg_full = x;
else
    cs = cumsum(x);
    avg_full(freq:end) = (cs(freq:end) - [0; cs(1:end-freq)]) / freq;
end

avg_trimmed = avg_full(freq:end);

end
