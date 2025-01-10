function [all_b,all_covar,all_latent] = est_var_dueker(crisis,observable,lagorder,start_values,true_b,true_covar,true_latent)

%% EST_VAR_DUEKER estimates parameters and latent variables along Dueker.
% _________________________________________________________________________
% 
% DESCRIPTION:
%   EST_VAR_DUEKER uses the method of Dueker (2003) to estimate parameters,
%   standard deviations and latent unobservable of a Qual VAR with variable 
%   inputs of true parameters (for testing if the method works).
%   Estimation of the latent variable NEEDS a forecast of lagorder periods
%   after the original sample period.
%
% LAST CHANGES 25/07/2012 mei gsz
% _________________________________________________________________________
%
% SYNTAX: 
%   [all_b,all_covar,all_latent] = est_var_dueker(crisis,observable,
%           lagorder,start_values,true_b,true_sigma,true_latent)
% _________________________________________________________________________
%
% OUTPUT:
%   all_b:          parameters from all gibbssteps
%   all_covar:      covariance matrices from all gibbssteps
%   all_latent:     latent variable from all gibbssteps
% _________________________________________________________________________
%
% INPUT:
%   crisis:         T vector of observable crisis periods
%   observable:     Txk matrix of observable variables
%   lagorder:       lag order of VAR (default = 1)
%   start_values:   starting values of the latent variable (facultative).
%   true_b:         true parameters (facultative).
%   true_covar:     true covariance matrix (facultative).
%   true_latent:    values of the latent variable to be used either for all
%                   first or all periods in all steps (facultative).

%% Fixed programmed parameters
gibbssteps = 1000;                      % # of draws to be saved
%startup = 2000;
%choice = 2;
 startup = 5000;                        % # of burn-in draws
 choice = 5;                            % stepsize of draws that are kept (drop intermediate draws to reduce autocorrelation of MCMC draws)
tot_steps = startup+choice*gibbssteps;


%% Argument check
check_b = 1;
check_covar = 1;
check_y = 1;
check_latstart = 1;
check_start = 0;

if nargin<4
    start_values = [];
else
    if ~isempty(start_values)
        y_start=start_values;
        check_start = 1;
    end        
end
if nargin<5
    check_b = 0;
elseif isempty(true_b)
    check_b = 0;
end
if nargin<6
    check_covar = 0;        
elseif isempty(true_covar)
    check_covar = 0;
end
if nargin<7
    check_latstart = 0;
    check_y = 0;
    true_latent = [];
else
    if isempty(true_latent)
        check_latstart = 0;
        check_y = 0;        
    elseif length(true_latent) == length(crisis)
        latent_start = true_latent(1:lagorder);
    elseif length(true_latent) == lagorder
        latent_start = true_latent;
        check_y = 0;
    else
        error('true_latent does not have one of two possible dimensions');
    end
end
    

%% Preparation of estimation
[T,k] = size(observable);
varcnt = k+1;

all_b = zeros(varcnt*lagorder+1,varcnt,gibbssteps);
all_covar = zeros(varcnt,varcnt,gibbssteps);
all_latent = zeros(T,1,gibbssteps);

%transposed for consistency with Dueker
X = observable';

% First parameters have to be defined here
%transposed for consistency with Dueker
b = nan(lagorder*varcnt+1,varcnt);
covar = nan(varcnt,varcnt);
if check_start
    if size(y_start,1)>size(y_start,2)
        y_start = y_start';
    end
    y = y_start;
else
    if length(true_latent) == length(crisis)
        y = true_latent';
    else
        y = rand(1,T);
        y(~crisis) = -y(~crisis);
    end
end

%lagorder additional values (nan) are added for forecasts
y = [y nan(1,lagorder)];
X = [X nan(k,lagorder)];
    
%% GIBBS sampler

step = 1;
tic;
for counter = 1:tot_steps;    
     tElaps = toc;
    x = dbstack;
    if tElaps > 5 
        fprintf('\t Gibbs step %4.0f (%4.2f%%)\n',counter,100*counter/tot_steps)
        tic;
    end        
    
    %tic;
    %Preparation of exog for sampling of b and covar
    if ~check_b || ~check_covar
        Y = [X;y]';
        Y = Y(1:end-lagorder,:);      %no forecasting for covariance
        exog = ones(T-lagorder,(lagorder+1)*varcnt+1);
        for t = 0:lagorder
            exog(:,t*varcnt+1:(t+1)*varcnt) = Y(lagorder+1-t:end-t,:);
        end
        lhs = exog(:,1:varcnt);
        rhs = exog(:,varcnt+1:end);
    end
    %fprintf('\t Preparation in %4.0f sec\n',toc)
    
    %RESAMPLING OF VAR COEFFICIENTS
    %tic;
    if check_b
        b = true_b;       
    else
        
        X_sq_inv = (rhs'*rhs)^(-1);      
        mu_b =  X_sq_inv * (rhs'*lhs);
        
        if counter == 1
            epsmat = lhs-rhs*mu_b;
            covar = (epsmat'*epsmat) ./ (T-lagorder);
        end
        for var = 1:varcnt
            var_b = covar(var,var) * X_sq_inv;
            b(:,var) = mvnrnd(mu_b(:,var),var_b);          
        end
    end
    %fprintf('\t Resampling b in %4.0f sec\n',toc)
    
    %tic;
    %RESAMPLING COVAR MATRIX
    if check_covar
        covar = true_covar;
    else        
        epsmat = lhs - rhs*b;
%         epsmat = exog*b_exp;        
        covar = iwishrnd(epsmat'*epsmat,T);     %NOT TOO SURE ABOUT df
        std_ee = sqrt(covar(end,end));
        covar(:,end) = covar(:,end)/std_ee;
        covar(end,:) = covar(end,:)/std_ee;
    end
    %fprintf('\t Resampling covar in %4.0f sec\n',toc)
    
    %tic;
    %Variables needed for sampling of latent    
    b_exp = [eye(varcnt);-b];   %NOT transposed - not consistent with Dueker
    covar_inv = covar^(-1);
    
    if check_latstart
        y(1:lagorder) = latent_start;
    end 
    %fprintf('\t Preparation y in %4.0f sec\n',toc)
    
    %RESAMPLING LATENT VARIABLE    
    if check_y
        y = true_latent';
        y = [y nan(1,lagorder)];
    else    
        %tic;
        %(1) LAGORDER FORECASTS for the last values of ystar
        Y = [X;y];
        for t = T+1:T+lagorder
            Y_block = Y(:,t-lagorder:t-1);
            Y_block = Y_block(:,(end:-1:1)); %has to be reordered to be backward looking (matching order of b)
            exog = [reshape(Y_block,1,lagorder*varcnt) 1];
            epsilon = mvnrnd(zeros(varcnt,1),covar);

            Y(:,t) = exog * b + epsilon;
        end
        X = Y(1:end-1,:);
        y = Y(end,:);
        %fprintf('\t Last values y in %4.0f sec\n',toc)
        
       % tic;
        %(2) calculation of time-invariant C with all necessary components
        %as well as the time-invariant parts of D        
        Dpart = cell(lagorder+1,1);
        Dpart{1} = -covar_inv;
        C = covar_inv;
        for l = 1:lagorder
            %transposed for consistency with Dueker
            b_lag = b(varcnt*(l-1)+1:varcnt*l,:)';
            
            Dpart{l+1} = b_lag' * covar_inv;
            C = C + Dpart{l+1} * b_lag;            
        end
        Cinv = C^(-1);
        C11_inv = C(end,end)^(-1);
        C01 = C(end,1:end-1);    
        Cfactor = C11_inv*C01;
        sigma_y = sqrt(C11_inv);
        %fprintf('\t Preparation C in %4.0f sec\n',toc)
        
        %(3) Resampling of periods after the first lags.
        for t = lagorder+1:T
            Y = [X;y];
            
            %(3.1) Calculation of D
            %tic 
            D = 0;
            for l = 0:lagorder
                Y_block = Y(:,t-lagorder+l:t+l);
                Y_block(:,end-l) = 0;
                Y_block = Y_block(:,(end:-1:1)); %has to be reordered to be backward looking (matching order of b)

                %NOT transposed - not consistent with Dueker
                exog = [reshape(Y_block,1,(lagorder+1)*varcnt) 1];

                %transposed for consistency with Dueker
                kappa = (exog*b_exp)';

                %NOT transposed - not consistent with Dueker
%                 D = -b_exp(varcnt*l+1:varcnt*(l+1),:) * covar_inv * kappa;    
                D = D + Dpart{l+1}*kappa;
            end
            %fprintf('\t Preparation D in %4.8f sec\n',toc)
            %tic
            %(3.2) Calculation of variable parameters
            shift_mu = Cinv * D;
            Yhat_t = Y(:,t) - shift_mu;
            Xhat_t = Yhat_t(1:end-1);

            mu_y = Cfactor*Xhat_t;
            mu = -mu_y+shift_mu(end);

            prob_zero = normcdf(0,mu,sigma_y);

            if crisis(t)
                p = rand()*(1-prob_zero) + prob_zero;
            else
                p = rand()*prob_zero;
            end
            %fprintf('\t Variable Parameters D in %4.8f sec\n',toc)
            %tic
            %(3.3) Draw values for y
            yhat_t = norminv(p,mu,sigma_y);             
            yhat_t = min(4*sigma_y,yhat_t);     %avoid inf for y
            y(t) = yhat_t;
            %fprintf('\t Draw y in %4.8f sec\n',toc)
            %save parameters of the distribution of ystar for first periods.
            if t == lagorder+1
                musave = mu;
                sigmasave = sigma_y;
                probsave = prob_zero;
            end
        end    
       % fprintf('\t y in the middle in %4.0f sec\n',toc)
        
        %tic;
        %(3.4) Resampling of first periods.
        for t = 1:lagorder
    %         check = 0;        
            if crisis(t)
                p = rand()*(1-probsave) + probsave;
            else
                p = rand()*probsave;
            end
            ynew = norminv(p,musave,sigmasave);
            yold = y(t);

            %Initialization of numerator and denominator. Scaling can be
            %ignored.        
            denominator = normpdf(yold,musave,sigmasave);
            numerator = normpdf(ynew,musave,sigmasave);

            Yold = [X;y];        
            Ynew = Yold;
            Ynew(end,t) = ynew;
            for l = 1:t
                Y_blockold = Yold(:,l:l+lagorder);            
                Y_blocknew = Ynew(:,l:l+lagorder);
                Y_blockold = Y_blockold(:,end:-1:1);
                Y_blocknew = Y_blocknew(:,end:-1:1);

                exogold = [reshape(Y_blockold,1,(lagorder+1)*varcnt) 1];
                exognew = [reshape(Y_blocknew,1,(lagorder+1)*varcnt) 1];

                epsold = (exogold * b_exp)';
                epsnew = (exognew * b_exp)';

                pold = mvnpdf(epsold,zeros(varcnt,1),covar);
                pnew = mvnpdf(epsnew,zeros(varcnt,1),covar);

                denominator = denominator*pold;
                numerator = numerator*pnew;
            end

            Rd = numerator/denominator;
            if (Rd>1) || (rand()<Rd)
                ynew = min(4*sigma_y,ynew);     %avoid inf for y
                y(t) = ynew;
            end
        end
        %fprintf('\t y start in %4.0f sec\n',toc)
    end
    
    %SAVE THE SAMPLED VALUES
    if (counter>startup) 
        if (choice == 1) || (mod(counter-startup,choice)==1)
            all_covar(:,:,step) = covar;
            all_b(:,:,step) = b;
            all_latent(:,:,step) = y(1:T);
            step = step+1;
        end
    end
end

