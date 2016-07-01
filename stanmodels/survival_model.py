
import patsy
import stanity
import pandas as pd
import numpy as np


def fit_stan_survival_model(df, formula, event_col, model_code,
                                           model_cohort = 'survival model', 
                             time_col = None,
                             sample_id_col = None, sample_col = None,
                             group_id_col = None, group_col = None,
                             timepoint_id_col = None, timepoint_end_col = None,
                             make_inits = None,
                             *args, **kwargs):
    """This function prepares inputs appropriate for stan model model code, and fits that model using Stan.

    Args:
       df (pandas DataFrame):  The data frame containing input data to Survival model.
       formula (chr): Patsy formula to use for covariates. E.g 'met_status + pd_l1'
       event_col (chr): name of column containing event status. Will be coerced to int
       model_code (chr): stan model code to use.

    Kwargs:
       model_cohort (chr): description of this model fit, to be used when plotting or summarizing output
       time_col (chr): name of column containing event time -- used for parameteric (Weibull) model
       sample_id_col (chr): name of column containing numeric sample ids (1-indexed & sequential)
       sample_col (chr): name of column containing sample descriptions - will be converted to an ID
       group_id_col (chr): name of column containing numeric group ids (1-indexed & sequential)
       group_col (chr): name of column containing group descriptions - will be converted to an ID 
       timepoint_id_col (chr): name of column containing timepoint ids (1-indexed & sequential)
       timepoint_end_col (chr): name of column containing end times for each timepoint (will be converted to an ID)

    Returns:
       dictionary of results objects.  Contents::

          df: Pandas data frame containing input data, filtered to non-missing obs & with ID variables created
          x_df: Covariate matrix passed to Stan
          x_names: Column names for the covariate matrix passed to Stan
          data: List passed to Stan - contains dimensions, etc.
          fit: pystan fit object returned from Stan call
          coefs: posterior draws for coefficient values
          loo: psis-loo object returned for fit model. Used for model comparison & summary
          model_cohort: description of this model and/or cohort on which the model was fit

    Raises:
       AttributeError, KeyError

    Generic helper function for fitting variety of survival models using Stan.

    Example:

    >>> testfit = fit_stan_survival_model(
                model_file = stanmodels.stan.pem_survival_model,
                formula = '~ met_status + pd_l1',
                df = dflong,
                sample_col = 'patient_id',
                timepoint_end_col = 'end_time',
                event_col = 'end_failure',
                model_cohort = 'PEM survival model',
                iter = 30000,
                chains = 4,
            )
    >>> print(testfit['fit'])
    >>> seaborn.boxplot(x = 'value', y = 'variable', data = testfit['coefs'])

    """

    ## input covariates given formula
    x_df = patsy.dmatrix(formula,
                          df,
                          return_type='dataframe'
                          )
    x_df = x_df.ix[:, x_df.columns != 'Intercept']
    
    ## construct data frame with all necessary columns
    ## limit to non-missing data 
    ## (if necessary) transform columns to ids
    other_cols = [event_col, time_col,
                  group_id_col, group_col,
                  timepoint_id_col, timepoint_end_col,
                  sample_id_col, sample_col] ## list of possible columns to keep
    
    other_cols = list(set(other_cols))
    other_cols.remove(None)
    
    if other_cols and len(other_cols)>0:
        ## filter other inputs to non-missing observations on input covariates
        df_nonmiss = x_df.join(df[other_cols]).dropna()
    else:
        df_nonmiss = x_df

    ## construct ID vars if necessary
    if timepoint_end_col and not(timepoint_id_col):
        timepoint_id_col = 'timepoint_id'
        df_nonmiss[timepoint_id_col] = df_nonmiss[timepoint_end_col].astype('category').cat.codes + 1

    if sample_col and not(sample_id_col):
        sample_id_col = 'sample_id'
        df_nonmiss[sample_id_col] = df_nonmiss[sample_col].astype('category').cat.codes + 1
        
    if group_col and not(group_id_col):
        group_id_col = 'group_id'
        df_nonmiss[group_id_col] = df_nonmiss[group_col].astype('category').cat.codes + 1

    survival_model_input_data = {
        'N': len(df_nonmiss.index),
        'x': x_df.as_matrix(),
        'event': df_nonmiss[event_col].values.astype(int),
        'M': len(x_df.columns),
    }
    
    if time_col:
        survival_model_input_data['y'] = df_nonmiss[time_col].values

    if group_id_col:
        survival_model_input_data['g'] = df_nonmiss[group_id_col].values.astype(int)
        survival_model_input_data['G'] = len(df_nonmiss[group_id_col].unique())
    
    if sample_id_col:
        survival_model_input_data['s'] = df_nonmiss[sample_id_col].values.astype(int)
        survival_model_input_data['S'] = len(df_nonmiss[sample_id_col].unique())
    
    if timepoint_id_col:
        survival_model_input_data['t'] = df_nonmiss[timepoint_id_col].values.astype(int)
        survival_model_input_data['T'] = len(df_nonmiss[timepoint_id_col].unique())
        
    if timepoint_end_col:
        survival_model_input_data['obs_t'] = df_nonmiss[timepoint_end_col].values.astype(int)
    
    if make_inits:
        kwargs = dict(kwargs, init = make_inits(survival_model_input_data))
    
    survival_fit = stanity.fit(
        model_code = model_code,
        data = survival_model_input_data,
        *args,
        **kwargs
    )
    
    beta_coefs = pd.DataFrame(
        survival_fit.extract()['beta'],
        columns = x_df.columns
    )
    beta_coefs = pd.melt(beta_coefs)
    beta_coefs['model_cohort'] = model_cohort
    
    ## prep by-group coefs if group specified
    if group_id_col:
        if group_col:
            grp_names = df_nonmiss.loc[~df_nonmiss[[group_id_col]].duplicated()].sort_values(group_id_col)[group_col].values
        else:
            grp_names = df_nonmiss.loc[~df_nonmiss[[group_id_col]].duplicated()].sort_values(group_id_col)[group_id_col].values
        grp_coefs_extract = survival_fit.extract()['grp_beta']
        grp_coefs_data = list()
        i = 0
        for grp in grp_names:
            grp_data = pd.DataFrame(grp_coefs_extract[:,:,i], columns = x_df.columns)
            grp_data['group'] = grp 
            grp_coefs_data.append(grp_data)
            i = i+1
        grp_coefs_data = pd.concat(grp_coefs_data)
        grp_coefs = pd.melt(grp_coefs_data, id_vars = 'group')
        grp_coefs['model_cohort'] = model_cohort
    else:
        grp_coefs = beta_coefs
        grp_coefs['group'] = 'Overall'

    loo = stanity.psisloo(survival_fit.extract()['log_lik'])
    
    return {
        'df': df_nonmiss,
        'x_df': x_df,
        'x_names': x_df.columns,
        'data': survival_model_input_data,
        'fit': survival_fit,
        'coefs': beta_coefs,
        'grp_coefs': grp_coefs,
        'loo': loo,
        'model_cohort': model_cohort,
    }

def extract_grp_baseline_hazard(results, timepoint_id_col = 'timepoint_id', timepoint_end_col = 'end_time'):
    """ If model results contain a grp_baseline object, extract & summarize it
    """

    ## TODO check if results are by-group
    ## TODO check if baseline hazard is computable
    grp_baseline_extract = results['fit'].extract()['grp_baseline']
    coef_group_names = results['grp_coefs']['group'].unique()
    i = 0
    grp_baseline_data = list()
    for grp in coef_group_names:
        grp_base = pd.DataFrame(grp_baseline_extract[:,:,i])
        grp_base_coefs = pd.melt(grp_base, var_name=timepoint_id_col, value_name='baseline_hazard')
        grp_base_coefs['group'] = grp
        grp_baseline_data.append(grp_base_coefs)
        i = i+1
    grp_baseline_coefs = pd.concat(grp_baseline_data)
    end_times = _extract_timepoint_end_times(results, timepoint_id_col = timepoint_id_col, timepoint_end_col = timepoint_end_col)
    bs_data = grp_baseline_coefs.join(end_times, by = timepoint_id_col) 
    return(bs_data)

def _extract_timepoint_end_times(results, timepoint_end_col = 'end_time', timepoint_id_col = 'timepoint_id'):
    df_nonmiss = results['df']
    end_times = df_nonmiss.loc[~df_nonmiss[[timepoint_id_col]].duplicated()].sort_values(timepoint_id_col)[[timepoint_end_col, timepoint_id_col]]
    return(end_times)

def extract_baseline_hazard(results, timepoint_id_col = 'timepoint_id', timepoint_end_col = 'end_time'):
    """ If model results contain a baseline object, extract & summarize it
    """
    ## TODO check if baseline hazard is computable
    baseline_extract = results['fit'].extract()['baseline']
    baseline_coefs = pd.DataFrame(baseline_extract)
    bs_coefs = pd.melt(baseline_coefs, var_name = timepoint_id_col, value_name = 'baseline_hazard')
    end_times = _extract_timepoint_end_times(results, timepoint_id_col = timepoint_id_col, timepoint_end_col = timepoint_end_col)
    bs_data = bs_coefs.join(end_times, by = timepoint_id_col) 
    return(bs_data)

## convert wide survival data to long format
def prep_data_long_surv(df, time_col, event_col):
    ''' convert wide survival data to long format
    '''
    ## identify distinct failure/censor times
    failure_times = df[time_col].unique()
    ftimes = pd.DataFrame({'end_time': failure_times, 'key':1})
    
    ## cross join failure times with each observation
    df['key'] = 1
    dflong = pd.merge(df, ftimes, on = 'key')
    
    ## identify end-time & end-status for each sample*failure time
    def gen_end_failure(row):
        if row[time_col] > row['end_time']:
            ## event not yet occurred (time_col is after this timepoint)
            return False
        if row[time_col] == row['end_time']:
            ## event during (==) this timepoint
            return row[event_col]
        if row[time_col] < row['end_time']:
            ## event already occurred (time_col is before this timepoint)
            return np.nan

    dflong['end_failure'] = dflong.apply(lambda row: gen_end_failure(row), axis = 1)
    
    ## confirm total number of non-censor events hasn't changed
    if not(sum(dflong.end_failure.dropna()) == sum(df[event_col].dropna())):
        print('Warning: total number of events has changed from {0} to {1}'.format(sum(df[event_col]), sum(dflong.end_failure)))

    
    ## remove timepoints after failure/censor event
    dflong = dflong.query('end_time <= {0}'.format(time_col))
    return(dflong)

def make_weibull_survival_model_inits(stan_input_dict):
    def f():
        m = {
            'tau_s_raw': abs(np.random.normal(0, 1)),
            'tau_raw': abs(np.random.normal(0, 1, stan_input_dict['M'])),
            'alpha_raw': np.random.normal(0, 0.1),
            'beta_raw': np.random.normal(0, 1, stan_input_dict['M']),
            'mu': np.random.normal(0, 1),
        }
        return m
    return f

