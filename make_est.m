function [est_b,est_covar,est_latent,conf_b,conf_covar,conf_latent] = make_est(all_b,all_covar,all_latent,conf)
%% MAKE_EST extracts point estimates and conf. bounds from a Gibbs Sampler
% _________________________________________________________________________
% 
% DESCRIPTION:
%   MAKE_EST computes the point estimates and confidence bounds from a
%   Dueker(2003) Gibbs-Sampler. The point estimate is derived roughly
%   following Fry and Pagan's method for impulse response functions.
%
% LAST CHANGES 26/07/2012 mei 
% _________________________________________________________________________
%
% SYNTAX: 
%   [est_b,est_covar_all_latent,conf_b,conf_covar,conf_latent] 
%        = make_est(all_b,all_covar,all_latent,conf)
% _________________________________________________________________________
%
% OUTPUT:
%   est_b:          estimated parameters of the Qual VAR
%   est_covar:      estimated covariance matrix of the Qual VAR
%   est_latent:     the estimated latent variable
%   conf_b:         upper conf. bound, median and 
%                      lower confidence bound of b
%   conf_covar:     upper conf. bound, median and 
%                      lower confidence bound of covariance matrix
%   conf_latent:    upper conf. bound, median and 
%                      lower confidence bound of latent variable
% _________________________________________________________________________
%
% INPUT:
%   all_b:          parameters from all gibbssteps
%   all_covar:      covariance matrices from all gibbssteps
%   all_latent:     latent variable from all gibbssteps
%   conf:           width of the confidence bounds
% _________________________________________________________________________
%
% NOTE:
%   The median reported in the confidence bounds is NOT the point estimate!

%% Fixed programmed parameters

all_latent = squeeze(all_latent);

%% Compute distributions

steps = length(all_b);

bsize = numel(all_b(:,:,1));
% csize = numel(all_covar(:,:,1));
% csize = (size(all_covar(:,:,1),1)-1)^2;

b_mat = nan(bsize,steps);
% c_mat = nan(csize,steps);

p_b = nan(steps,1);
% p_c = nan(steps,1);

log_p_y = nan(steps,1);

for n = 1:steps
    b_mat(:,n) = reshape(all_b(:,:,n),bsize,1);
%     c_mat(:,n) = reshape(all_covar(:,:,n),csize,1);
%     c_mat(:,n) = reshape(all_covar(1:end-1,1:end-1,n),csize,1);
end

b_mean = mean(b_mat,2);
% c_mean = mean(c_mat,2);

b_clean = b_mat - repmat(b_mean,1,steps);
% c_clean = c_mat - repmat(c_mean,1,steps);

b_sigma = b_clean*b_clean'/steps;
% c_sigma = c_clean*c_clean'/steps;

y_mean = mean(all_latent,2);
y_sigma = std(all_latent,[],2);

%% Compute probalilities

for n = 1:steps
    if sum(diag(b_sigma)==0)==0
        p_b(n) = mvnpdf(b_mat(:,n),b_mean,b_sigma);
    else
        p_b(n) = 1;
    end
%     p_c(n) = mvnpdf(c_mat(:,n),c_mean,c_sigma);    
    
    if sum(y_sigma==0)==0
        p_y = normpdf(all_latent(:,n),y_mean,y_sigma);        
    else
        p_y = 1;
    end
    log_p_y(n) = sum(log(p_y));
end

%% Compute joint probability and find maximum;

p = nan(steps,1);
p = log_p_y+log(p_b);
% p = log_p_y+log(p_b)+log(p_c);

i = find(p == max(p));
i = i(1);

est_b = all_b(:,:,i);
est_covar = all_covar(:,:,i);
est_latent = all_latent(:,i);

%% Compute confidence bounds
low = (1-conf)/2;
high = 1 - low;
conf_b = quantile(all_b,[low 0.5 high],3);
conf_covar = quantile(all_covar,[low 0.5 high],3);
conf_latent = quantile(all_latent,[low 0.5 high],2);

end

