#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Régis Gbenou
@email: regis.gbenou@outlook.fr
"""


###############################################################################
#                                PACKAGES
###############################################################################

import itertools                                     # Library providing iterative functions.
import matplotlib.pyplot as plt                      # Library to plot graphics.
import numpy as np                                   # Library for matrix computation.
import pandas as pd                                  # Library for data manipulation.
import seaborn as sns                                # Library to display graphics and make some basic statistics.
sns.set(style="ticks", color_codes=True)
import statsmodels.api as sm                         # Library for statistical models.
import statsmodels.stats.api as sms                  # Library for statistical models.
from statsmodels.stats.outliers_influence import(
    variance_inflation_factor)                       # Function to evaluate multicolinearity of columns in a matrix.
from statsmodels.stats.stattools import(
    medcouple,
    durbin_watson)                                   # Funcitons that evalue respectively the autocorrelation (AR(1)) in
# a time serie and the skewness (asymmetry) of a distribution.
from statsmodels.graphics.tsaplots import plot_acf   # To plot AutoCorrelation Function (ACF)
import scipy.stats as sci                            # Library of computation interest.
import sklearn                                       # Library of machine Learning Model
from sklearn.linear_model import LinearRegression    # Module to access to Linear Regression, and other regularized functions.

from sklearn.model_selection import(
    KFold, cross_validate, GridSearchCV)             # Functions helping in the selection of features or hyperparameters
from sklearn.pipeline import Pipeline                # Module to produce pipeline estimators.
from sklearn.preprocessing import PowerTransformer   # Module to make Power Transformation.
import time                                          # Library to access to the computer clock.



###############################################################################
#                                FUNCTIONS
###############################################################################


# DESCRIPTIVE PART


def plot_fct(df, cols_to_plot=[], ncols=2, size=(7, 3)):
    '''
    <plot_fct> plots the values, sorted by ascending order of each column having its hearder in <cols_to_plot>,
    if <cols_to_plot> is left empty this fucntion will take in account by default all the
    numercial columns of <df>.
    
    Parameters
    ----------
    df : DataFrame
        Any data frame containing at least 3 numerical columns.
    cols_to_plot : list, optional
        cols_to_plotist of a subset of <df> columns. The default is []
    ncols : int, optional
        Number of subplot by row. The default is 2.
    size : tuple, optional
        (width, height) of all graphs. The default is (10, 7).

    Returns
    -------
    None.
    '''
    # Displaying all numerical column if cols_to_plot is empty.
    if not cols_to_plot:
        cols_to_plot = list(df.select_dtypes('number'))
        
    nrows = int(round(len(cols_to_plot)/ncols))
    fig, axs = plt.subplots(nrows, ncols, figsize=size)
    iterative_var = 0
    
    # i,j belongs to the cartesian product: [1,nrows]X[1,ncols]
    for i,j in itertools.product(np.arange(nrows),np.arange(ncols)): 
        try:                                                  
            # x is a vector going from 0 to the row number of df.
            x =  np.arange(df.shape[0])
            # y is an array containing the sorted values of the columns cols_to_plot[iterative_var].
            y =  df[cols_to_plot[iterative_var]].sort_values().values                
        except IndexError:       
            # Displaying blank plot.
            pass                                              
        else:
            # Displaying y(x).
            axs[i,j].scatter(x, y)
            axs[i,j].set_ylabel(cols_to_plot[iterative_var])
        finally:
            # Always incrementing iterative_var.
            iterative_var += 1 
    plt.tight_layout()
    plt.show()
    return None


def boxPlot_fct(df, L=[], ncols=2, size=(7, 3)):
    '''
    <plot_fct> plots the box plots of the columns having their hearder in <L>,
    if <L> is left empty this fucntion will take in account by default all the
    numercial columns of <df>.
    The box plots are modified using the thumb rule relying on medcouple:
        https://en.wikipedia.org/wiki/Box_plot#Variations
    The outliers according to the previous thumb rule will be colored in red.
        
    Parameters
    ----------
    df : DataFrame
        Any data frame containing at least 3 numerical columns.
    L : list, optional
        Name of the numerical features present in <df> that we want the box plot.The
        default is [].
    ncols : int, optional
        Number of columns. The default is 2.
    size : tuple, optional
        (width, height) of all subgraphs. The default is (10, 7).

    Returns
    -------
    None.
    '''
    if L:
        pass
    else:
        L = list(df.select_dtypes('number'))
    nrows = int(round(len(L)/ncols))
    red_squares = dict(markerfacecolor='r', marker='s')
    fig, ax = plt.subplots(nrows, ncols, figsize=size)
    for i,k in zip(itertools.product(np.arange(nrows), np.arange(ncols)), L):
        mc = medcouple(df[k])
        if mc > 0:
            iq_adj = [1.5*np.exp(-4*mc), 1.5*np.exp(3*mc)] # computation of the adjusted interquartile for a medcouple value greater than 0.
        else:
            iq_adj = [1.5*np.exp(-3*mc), 1.5*np.exp(4*mc)]
        low = np.percentile(df[k], 25) - iq_adj[0]*(np.percentile(df[k], 75) - np.percentile(df[k], 25))
        up = np.percentile(df[k], 75) + iq_adj[1]*(np.percentile(df[k], 75) - np.percentile(df[k], 25))
        a, b = round(sci.percentileofscore(df[k], low), 2), round(sci.percentileofscore(df[k], up), 2)
        ax[i[0], i[1]].boxplot(df[k], flierprops=red_squares, whis=(a, b), vert=False)
        ax[i[0], i[1]].set_title(k)
    plt.tight_layout()
    plt.show()
    return None


def ols_summary_fct(dfo, name_y, intercept=True, cov_fit='', summary=True, graph=True, vif=True, size=(10, 7), method='box-cox',cols_boxCox=[]):
    '''
    <ols_summary_fct> checks the following assumptions:
        - Homoscedasticity (equal variance accross the observations),
        - Linear relationshp,
        - Independency (a weaker criterion that is almost equivalent is autocorrelation),
        - Non-colinearity,
        - Normality

    Parameters
    ----------
    df : DataFrame
        Data frame gathering the response and feature observations.
    name_y : str
        Name of the response in <df>.

    Returns
    -------
    None.
    '''
    # a. OLS Regression summary -----------------------------------------------
    df = dfo.copy()
    df = df.reset_index(drop=True)
    col_to_transf = list(df.reindex(columns=cols_boxCox).dropna(axis=1))
    if col_to_transf:            
        power_t = PowerTransformer(method=method, standardize=False)
        power_t.fit(df.loc[:, col_to_transf])
        df.loc[:, col_to_transf] = power_t.transform(df.loc[:, col_to_transf])
        # df_lambdas = pd.DataFrame({'lambda':power_t.lambdas_},
          #  index=col_to_transf)
    if cols_boxCox:
        print(f'Applying Box-Cox transformation on the following columns: {col_to_transf}')
       
    predictor_names = list(df)
    predictor_names.remove(name_y)
    if intercept:
        X = sm.add_constant(df.loc[:, predictor_names])
    else:
        X = df.loc[:, predictor_names]
    y = df.loc[:, name_y]
    model = sm.OLS(y, X)
    if len(cov_fit):
        results = model.fit(cov_type=cov_fit)
    else:
        results = model.fit()
    if summary:
        print(results.summary())
    else:
        print(f"\nR2_adj: {round(100*results.rsquared_adj, 1)}%")
        print("\nLOG-LIKELIHOOD: ", "{:.2e}".format(round(results.llf, 1)))
        print("\nFscore: ", round(results.fvalue, 1))
        print("\nFpvalue: ", round(results.f_pvalue, 1))
        print("\nAIC: ", "{:.2e}".format(round(results.aic, 1)))
        print("\nBIC: ", "{:.2e}".format(round(results.bic, 1)))
        print("\nCONDITION NUMBER: ", "{:.2e}".format(round(results.condition_number, 1)))
    print("\nRESIDUAL MSE: ", round(results.mse_resid, 1))

    # b. Homoscedasticity: detection of heteroscedasticity --------------------
    print('\nHOMOSCEDASTICITY ASSUMPTION')
    test = sms.het_breuschpagan(results.resid, results.model.exog) # tests heteroscedasticity with
    fscore, fpvalue = round(test[2], 1), round(test[3], 2)         # homoscedasticity as null hypothesis.
    x_het = results.predict()
    y_het = results.resid
    poly_het = np.poly1d(                                          # transforms the array of polynomial coefficients 
        np.polyfit(x_het, y_het, deg=3))                           # in a function that can take a vector as argument.
    xp_het = np.linspace(x_het.min(), x_het.max(), int(y_het.shape[0]/4))
    print(f"Breush-Paga test:\tFscore:{round(fscore,1)}, Pval={round(fpvalue,1)}")
    
    # c. Independency: detection of autocorrelation in residuals --------------
    print('\nINDEPENDENCY ASSUMPTION')
    print('Autocorrelation of the first order:')                                               
    y = results.resid
    test_d = round(durbin_watson(y), 1)                            # tests the autocorrelation of first order, a rule of 
    print(f'The Durbin-Watson test result: {test_d}')              # thumb is that 1.5< test < 2.5 then values are considered as normal. 

    # d. Linearity: detection of non-linearity in residuals -------------------
    x_lin = df.index.values
    np.random.shuffle(x_lin)
    y_lin = results.resid    
    poly_lin = np.poly1d(
        np.polyfit(x_lin, y_lin, deg=3))
    xp_lin = np.linspace(x_lin.min(), x_lin.max(), int(y_lin.shape[0]/4))
    
    # e. Multi plots ---------------------------------------------------
    if graph:
        fig, axs = plt.subplots(nrows=2, ncols=2, figsize=size)
        dct_arg = {'x':[x_het, x_lin], 'y':[y_het, y_lin], 'xp':[xp_het, xp_lin],
            'yp':[poly_het(xp_het), poly_lin(xp_lin)],
            'xlabel':[r'Fitted values $\hat{y_{i}}$', 'observation number'],
            'title':[r"$residuals(\hat{y})$", "Residuals vs observation number"]}
        for i in range(2):
            axs[i, 0].scatter(x=dct_arg['x'][i], y=dct_arg['y'][i], label="residuals")
            axs[i, 0].set_xlabel(dct_arg['xlabel'][i])
            axs[i, 0].set_ylabel('residuals')
            axs[i, 0].plot(dct_arg['xp'][i],dct_arg['yp'][i], color="DarkRed", label="Polynomial estimation")
            axs[i, 0].set_title(dct_arg['title'][i])
            axs[i, 0].legend()
        plot_acf(results.resid, adjusted=True, zero=False,
                 title="Residuals autocorrelation", ax=axs[0,1]) 
        sci.probplot(results.predict(), dist="norm", plot=axs[1,1])
        axs[1, 1].set_title('Q-Q plot')
        plt.tight_layout()
    plt.show()
    
    # f. Multicolinearity: use of VIF criterion -------------------------------
    if vif:
        print('\nNON-COLINEARITY ASSUMPTION')
        X_multi = X.copy()
        if intercept:
            X_multi.drop(columns='const', inplace=True)
        print(pd.DataFrame({
            "VIF": [variance_inflation_factor(X_multi.values, i) 
                    for i in range(X_multi.shape[1])]},
            index = list(X_multi)).sort_values(by='VIF', ascending=False)
            )   
    return None




# INFERENCE AND PREDICTIVE PART


def find_bestHyperparameters(df, name_y, estimator, parameters=dict(), k_cv=5, rs=0, n_jobs=1, graph=True,
                            size=(7, 3)):
    '''
    Return best hyperparameter estimator.  

    Parameters
    ----------
    estimator : sklearn estimator
        Scikit-Learn method.
    df : Data Frame
        Frame containin the response and the predictor columns data.
    name_y : str
        Header of the response columns.
    parameters : dict
        Dictionary of parameters.
        The default is dict().
    k_cv : int, optional
        Cross validation value. The default is 5.
    rs: int, optional
        Random state.
        The default is 0.
    n_jobs : int
        The number of workers used to run the model.
        The default is 1.

    Returns
    -------
    Dcictionary composed of the more suitable hyperparameters for the method.
    And the grid search Cv results.

    '''
    
    duration = np.array(2 * [-time.time()])
    columns_list = list(df)    
    columns_list.remove(name_y)
    X = df.loc[:, columns_list]
    y = df.loc[:, name_y]
    # use of GridSearchCV to find the hyperparameters that reduce the most the mse.
    grid = GridSearchCV(
        estimator=estimator,
        param_grid=parameters,
        n_jobs=n_jobs,
        scoring='neg_mean_squared_error')
    grid.fit(X, y)
    index_bestRanked_param = np.argwhere(grid.cv_results_['rank_test_score']==1)[0][0]
    
    # handle pipe estimator.
    if type(estimator) == sklearn.pipeline.Pipeline: 
        initial_param_dict = estimator.named_steps.copy()
        final_param_dict = initial_param_dict.copy()
        for k in parameters:
            print(k.split('__')[0])
            final_param_dict[k.split('__')[0]] =\
                initial_param_dict[k.split('__')[0]].set_params(
                    **{k.split('__')[1] : grid.cv_results_['params'][index_bestRanked_param][k]})
    # build new estimator handling parameters given in argument.
        estimator_optimized = Pipeline(list(final_param_dict.items()))
    else:
        estimator_optimized = estimator.set_params(**grid.cv_results_['params'][index_bestRanked_param])

    # evaluate the model with the found hyperparameters.
    scores = cross_validate(
        estimator= estimator_optimized,
        X=X, y=y, n_jobs=n_jobs,
        scoring=('r2', 'neg_mean_squared_error'),
        cv=KFold(n_splits=k_cv, shuffle=True, random_state=rs))

    # information to display.
    text = f'For {estimator_optimized} we have MSE:'+\
          f"{int(scores['test_neg_mean_squared_error'].mean())} +-"+\
          f"{int(scores['test_neg_mean_squared_error'].std())}"+\
          f"\nFor {estimator_optimized} we have R^2:"+\
          f"{round(100*scores['test_r2'].mean(), 1)} +-"+\
          f"{round(100*scores['test_r2'].std(), 1)}"+\
          "\nThe above results are estimated by cross_validate() and may slightly differ from those estimated by GridSearchCV()"  
    print(text)
    
    # graphic to identify the improvement of the found hyperparameters in comparison with other ones.
    if graph:
        fig, ax = plt.subplots(figsize=size)
        y = np.abs(grid.cv_results_['mean_test_score'])
        y_min = np.nanmin(y)
        x = np.arange(len(y))
        ax.scatter(x, y)
        ax.plot(x, np.repeat(y_min, repeats=len(x)), linestyle='--', color='r', label=f'min_mse={"{:.2e}".format(round(y_min,1))} (estimated by GrideSearchCV())')
        ax.axvline(x=index_bestRanked_param, linestyle='-.', color='r',
                   label=f"{grid.cv_results_['params'][index_bestRanked_param]}")
        ax.set_xticks(x)
        ax.set_xticklabels(grid.cv_results_['params'], rotation=90)
        ax.legend(loc='best')
        ax.set_title('Mean Squared Error vs Hyperparameters')
        plt.show()
    duration[1] = time.time()
    quotient_hour, remainder_hour = duration.sum()// 3600, duration.sum() % 3600
    quotient_minute, remainder_minute= remainder_hour // 60, remainder_hour % 60
    print(f"Elapsed time: {int(quotient_hour)}h {int(quotient_minute)}min {int(remainder_minute)}s")
    
    return {'bestHyperParams':grid.cv_results_['params'][index_bestRanked_param],
            'gridSearch': grid.cv_results_, 'recap':text}


def mse_r2_score_fct(df, name_y, id_columns, estimator, L, cols_categ=[], method='box-cox',
    pt_standardize=True, cols_boxCox=[], drp_first_level=True, parameters={}, fct=lambda x: x, cv=5, n_jobs=1):
    '''

    Return best hyperparameter estimator for a pipeline estimator. 

    Parameters
    ----------
    df : DataFrame 
        Data frame containing the response and the features.
    estimator : sklearn
        Scikit-Learn method
    parameters : dict
        Dictionary of parameters.
    name_y : str
        The response column name.
    L : list
        Name of all possible columns of df. Note that depending on the value of cols_categ L can change. 
    cols_categ : list
        List of the orignal categorical columns which their levels have to be converted in dummy variables.
        The default is [].
    id_columns : list/array
        Column indexes to select in X.
    parameters : set, optional
        Set having as keys estimator argument and values the values of these arguements. The default is {}.
    fct : function
        Any function to apply to the response to compute a transformed rmse after the fitting.
        The default is lambda x: x (the identity function).
    cv : int, optional
        Cross validation value. The default is 5.
    n_jobs : int, optional
        Core number to use. The default is 1.

    Returns
    -------
    1D array
        [mse_mean, mse_std, r2_mean, r2.std]. Respectively the mean and the standard deviation of both means squared error and explained variance rate.

    '''
    
    # levels by category.
    lvls_by_categ = {k:[] for k in cols_categ}
    col_name = [L[i].split('_') for i in id_columns]
    for k in col_name:
        if k[0] in lvls_by_categ:
            lvls_by_categ[k[0]].append('_'.join(k[1:]))
    # gather column name that are not levels of categorical columns.
    col_name_2 = ['_'.join(k) for k in col_name if not k[0] in lvls_by_categ]
    col_tot = col_name_2 + list(lvls_by_categ)
    X = df.loc[:, col_tot]    
    # names of the catogorical columns.
    keys = list(lvls_by_categ)
    for key in lvls_by_categ:
        if len(lvls_by_categ[key]):
            X = X[X[key].isin(lvls_by_categ[key])].copy()
        else:
            keys.remove(key)
    # boolean will allow to drop the first level of each cateogorical variable if it is possible.
    if lvls_by_categ:
        boolean = min([len(lvls_by_categ[k])!=1 for k in lvls_by_categ.keys()])
    else:
        boolean=False
    if not drp_first_level:
        boolean = False
    X = pd.get_dummies(X, columns=keys, drop_first=boolean).select_dtypes('number')
    y = df.loc[X.index, name_y]
 
    # handle pipe estimator.
    if type(estimator) == sklearn.pipeline.Pipeline: 
        steps_pipe = estimator.named_steps.copy()
        steps_with_params = steps_pipe.copy()
        for k in parameters:
            name_estimator = k.split('__')[0]
            name_parameter = k.split('__')[1]
            print(name_estimator)
            steps_with_params[name_estimator] =\
                steps_pipe[name_estimator].set_params(
                    **{name_parameter: parameters[k]})
    # build new estimator handling parameters given in argument.
        estimator_2 = Pipeline(list(steps_with_params.items()))
    else:
        estimator_2 = estimator.set_params(**parameters)    
    
    
    frame = y.to_frame(name=y.name).join(X)

    score=('neg_means_squared_error', 'r2')   
    res = {k:[0, 0] for k in ['test_'+k for k in score]}
    mse_list, r2_list = [], []
    
    # apply one of the two power transformations provided by Scikit-Learn.
    frame_transf = frame.copy()
    col_to_transf = list(frame.reindex(columns=cols_boxCox).dropna(axis=1))
    if col_to_transf:            
        power_t = PowerTransformer(method=method, standardize=pt_standardize)
        power_t.fit(frame_transf.loc[:, col_to_transf])
        frame_transf.loc[:, col_to_transf] = power_t.transform(frame_transf.loc[:, col_to_transf])
    
    # cross-validation process, it shuffle the row data and make a K-fold. 
    kf = KFold(n_splits=cv, shuffle=True, random_state=0)
    for train, test in kf.split(frame_transf.iloc[:, 1:]):
        X_train, X_test = frame_transf.iloc[train, 1:], frame_transf.iloc[test, 1:]
        # y_test is left unchanged, then it is extracted from df not df_transf.
        y_train, y_test = frame_transf.iloc[train, 0], frame.iloc[test, 0]
        # to "reinitialize" estimator at each loop to avoid that the estimator be influenced by the
        # previous fit.
        estimator_to_use = estimator_2
        estimator_to_use.fit(X_train, y_train)
        y_pred = estimator_to_use.predict(X_test)
        if name_y in col_to_transf:
            y_pred_transf = fct(power_t.inverse_transform(
                np.concatenate(len(col_to_transf)*[y_pred.reshape(-1, 1)], axis=1)))[:,0]
        else:
            y_pred_transf = fct(y_pred)
        y_test_transf = fct(y_test)

        mse = ((y_test_transf - y_pred_transf)**2).mean()
        mse_list.append(mse)
        
        denominator = ((y_test_transf - y_test_transf.mean())**2).sum()
        # avoid case where y_test_transf contains only one value.
        if denominator:
            r2 = 100*(1 - ((y_test_transf - y_pred_transf)**2).sum()/denominator)
        else:
            r2 = np.nan
        r2_list.append(r2)

    mse_mean, mse_std = round(np.array(mse_list).mean(), 1), round(np.array(mse_list).std(), 1)   
    r2_mean, r2_std = round(np.array(r2_list).mean(), 1), round(np.array(r2_list).std(), 1)
    res['test_neg_means_squared_error'][0], res['test_neg_means_squared_error'][1] = mse_mean, mse_std
    res['test_r2'][0], res['test_r2'][1] = r2_mean, r2_std   
    return np.array([res[k] for k in res]).reshape(1,-1)[0]


def find_bestModel_fwd_bwd(df, name_y, estimator=LinearRegression(), mode='forward',
    pt_standardize=True, cols_categ=[], nrows_thresh=None, drp_first=True, fct=lambda x:x,  method='box-cox',
    cols_boxCox=[], parameters=dict(), cv=5, n_jobs=1, min_score = True, graph=True, size=(7, 3),
    thresh=10):
    
    '''
    <cv_forwardStepwiseSelection> makes forward stepwise selection using OLS
    regression as model and relying on estimated MSE, provided by k-fold 
    cross-validation method.

    Parameters
    ----------
    df : DataFrame
        Data frame gathering the response and feature observations, the response must be the
        first column.
    estimator: str, optional
        The scikit-learn estimator in string. The default is 'linear_model.LinearRegression'
    cols_categ: list, optional
        The name_y of columns which we will use as dummy variables. The default is [].
    score: tuple, optional
        The tuple contains the name_y of the metrics that we want get about the estimator. 
        The default value is ('neg_mean_squared_error', 'r2').
    parameters: dict, optional
        The set of hyperparameters that we will use in the estimator. The default is dict().
    cv : int, optional
        The number of splits in a KFold() cross validation function. The default is 5.
    n_jobs : int, optional
        The number of workers to use.

    Returns
    -------
    set
        Assume that the the data frame contains p features, the first item
        will be a data frame containing p + (p-1) + (p-2) + ... + 1 = p(p+1)/2
        rows. The first p rows correspond to the models containing only one 
        feature, next among these first p model we choose this one having the
        lower estimated mse. Then we add to the later model an other feature 
        among the remaining p-1 features and we keep the model having the lower
        estimated mse and so on.
        
        The second item is a data frame is summary of the best models of the 
        first item. It contains p rows, that each one correspond to the best 
        model, that have the lower estimated MSE for a fixed feature number.
        
        For the 2 items the data are organized in 3 columns like below:
        
        +--------------------+---------------------+------------------------+
        | number of features | indexes of features | estimated mse (k-fold) |
        +--------------------+---------------------+------------------------+
        |         3          |     (x8, x3, x6)    |      1587              |
        +--------------------+---------------------+------------------------+
        
    '''
    
    if not mode in {'forward', 'backward'}:
        print("We have to chose between mode takes either 'forward' or 'backward' value.")
        return {}

    duration = np.array(2 * [-time.time()])

    # a. Initialization
    score=('neg_means_squared_error', 'r2')
    score_params = np.array([[k+'_mean', k+'_std'] for k in score]).reshape(1,-1)[0]
    df_tot = pd.get_dummies(df.iloc[:, 1:], drop_first=False, columns=cols_categ)
    column_list = list(df_tot)
    column_df_list = list(df)
    column_df_list.insert(0, column_df_list.pop(column_df_list.index(name_y)))
    df = df.loc[:, column_df_list].copy()
    y = df.iloc[:, 0]
    p = len(column_list)


    list_df_fwd = []

    list_df_fwd.append(pd.DataFrame({"predictorNumb":[0 if mode=='forward' else p][0],
         "combinations":[tuple([] if mode=='forward' else tuple(range(p)))],
             "model": estimator})) 

    def featureless(mode,step='begining'): 

        if mode == 'forward':
            id_columns = tuple(range(1) if step=='begining' else range(p))
        elif mode == 'backward':
            id_columns = tuple(range(p) if step=='begining' else range(1))

        if id_columns == (0,):
            y_m = y.mean()
            y_mean = pd.DataFrame({'mean': [y_m for k in y]}, index=y.index)                    
            yy = y.to_frame(name=y.name).join(y_mean)
            col = list(yy.iloc[:, 1:])
            cols_c = []
        else: 
            yy = df.copy()
            col = column_list
            cols_c = cols_categ

        s = list_df_fwd[-1].loc[:, ['combinations', 'model']].apply(
                lambda x: mse_r2_score_fct(df=yy, name_y=name_y, id_columns=list(id_columns), method=method,
                         pt_standardize=pt_standardize, cols_boxCox=cols_boxCox,estimator=x[1], L=col,
                        cols_categ=cols_c, fct=fct,parameters=parameters, cv=cv, n_jobs=n_jobs), axis=1) 
        columns_to_select = [k.split('_')[-1] for k in col if k.split('_')[0]=='genre']
        if columns_to_select:
            nrows = pd.get_dummies(df[df.genre.isin(columns_to_select)]).shape[0]
        else:
            nrows = 0
        return {'score_serie':s, 'nrows':nrows}

    s = featureless(mode=mode)['score_serie'] 
    for k in score_params:
        list_df_fwd[-1].loc[:, k] = 0
    list_df_fwd[-1].loc[:, score_params] = pd.DataFrame(s.tolist(), columns=score_params, index=s.index)
    score_up = score[0]+'_up'   
    list_df_fwd[-1].loc[:, score_up] = list_df_fwd[-1].loc[:, score_params[0]]  +\
        list_df_fwd[-1].loc[:, score_params[1]]    
    list_model_fwd = list_df_fwd.copy()
    list_model_fwd[-1].loc[:, 'nrows'] = featureless(mode=mode)['nrows']
    print('The model without any predictor (forward) or the one with all predictors (backward) has been treated.')

    id_considered = tuple([] if mode=='forward' else range(p)) # (0, 1, 2, ..., p-1)
    id_not_considered = tuple(k for k in range(p) if not k in id_considered)  

    def rem(x, a):
        l = x.copy()
        l.remove(a)
        return l
    # b. Looping                            
    list_loop = list(range(p) if mode=='forward' else range(p-1, 0, -1))
    for j in list_loop:
        duration_i =  np.array(2 * [-time.time()])                                   # array to compute durationation.
        list_id_considered = list(id_considered)
        if mode=='forward':
            id_to_consider = np.array([list_id_considered+[k] for k in id_not_considered])
        else:
            id_to_consider = np.array([rem(list_id_considered, k) for k in id_considered])

        df_fwd_j = pd.DataFrame({
            "predictorNumb": np.array([j+1 if mode=='forward' else j for iter_var in range(len(id_to_consider))]),        
            "combinations": [tuple(x) for x in id_to_consider],
            "model": [estimator for k in id_to_consider]
            })
        
        s = df_fwd_j.loc[:, ['combinations', 'model']].apply(
            lambda x: mse_r2_score_fct(df=df, name_y=name_y, id_columns=list(x[0]), estimator=x[1], fct=fct,
                L=column_list, cols_categ=cols_categ, parameters=parameters, cv=cv, method=method,
                     pt_standardize=pt_standardize,cols_boxCox=cols_boxCox, n_jobs=n_jobs), axis=1)  
        if not nrows_thresh:
            nrows_thresh = df.shape[0]+1 if mode == 'forward' else -1
        fct_nrows = (lambda x : 0 if x>=nrows_thresh else 1) if mode=='forward'\
            else (lambda x : 0 if x<=nrows_thresh else 1)
        for k in score_params:
            df_fwd_j.loc[:, k] = 0
        df_fwd_j.loc[:, score_params] = pd.DataFrame(s.tolist(), columns=score_params, index=s.index)
        score_up = score[0]+'_up'
        df_fwd_j[score_up] = df_fwd_j[score_params[0]] + df_fwd_j[score_params[1]]
        print(f"Iteration number: {j}.")       
        list_df_fwd.append(df_fwd_j)
        list_model_fwd.append(
            df_fwd_j.sort_values(by=score[0]+'_mean', ascending=min_score).iloc[[0], :])
        tuple_from_best = list_model_fwd[-1].combinations.values[0]
        columns_selected = [column_list[i].split('_')[-1] for i in tuple_from_best if column_list[i].split('_')[0]=='genre']
        list_model_fwd[-1].loc[:, 'nrows'] = pd.get_dummies(df[df.genre.isin(columns_selected)]).shape[0]
        if mode == 'forward':
            set_diff = set(tuple_from_best) - set(id_considered)
            assert len(set_diff) == 1
            id_considered += tuple(set_diff)
        else:
            set_diff = set(id_considered) - set(tuple_from_best)
            assert len(set_diff) == 1
            value_to_remove = list(set_diff)[0]
            id_considered = tuple(rem(list_id_considered, value_to_remove))

        id_not_considered = tuple([k for k in np.arange(p) if not k in id_considered])

        duration_i[1] = time.time()
        quotient_hour, remainder_hour = duration_i.sum()// 3600, duration_i.sum() % 3600
        quotient_minute, remainder_minute= remainder_hour // 60, remainder_hour % 60
        print(f"Model with :{j}/{p-1} predictors.",\
        f"\nElapsed time: {int(quotient_hour)}h {int(quotient_minute)}min {int(remainder_minute)}s.")
        nrows = list_model_fwd[-1].loc[:, 'nrows'].values[0]
        if fct_nrows(nrows):
            continue
        else:
            break

    list_df_fwd.append(pd.DataFrame({"predictorNumb":[p if mode=='forward' else 0][0],
         "combinations":[tuple(range(p) if mode=='forward' else [])],
             "model": estimator})) 
    s = featureless(mode=mode, step='end')['score_serie']
    for k in score_params:
        list_df_fwd[-1].loc[:, k] = 0
    list_df_fwd[-1].loc[:, score_params] = pd.DataFrame(s.tolist(), columns=score_params, index=s.index)
    score_up = score[0]+'_up'   
    list_df_fwd[-1].loc[:, score_up] = list_df_fwd[-1].loc[:, score_params[0]]  +\
        list_df_fwd[-1].loc[:, score_params[1]]  
    list_model_fwd.append(
        list_df_fwd[-1].sort_values(by=score[0]+'_mean', ascending=min_score).iloc[[0], :])  
    list_model_fwd[-1].loc[:, 'nrows'] = featureless(mode=mode)['nrows']

    # c. Adding
    df_fwd_total = pd.concat(list_df_fwd).reset_index(drop=True)
    model_fwd_total = pd.concat(list_model_fwd).reset_index(drop=True)
    duration[1] = time.time()
    quotient_hour, remainder_hour = duration.sum()// 3600, duration.sum() % 3600
    quotient_minute, remainder_minute= remainder_hour // 60, remainder_hour % 60
    print(f"Elapsed time: {int(quotient_hour)}h {int(quotient_minute)}min {int(remainder_minute)}s")
    if graph:
        graph_dict = {model_fwd_total.predictorNumb.values[i]:
          model_fwd_total.neg_means_squared_error_mean.values[i] for i in range(model_fwd_total.shape[0])}
        graph_dict = dict(sorted(graph_dict.items(), key=lambda x: x[0])) if mode=='forward'\
            else dict(sorted(graph_dict.items(), key=lambda x: x[0], reverse=True))
        df_graph = pd.DataFrame({
            'pred_numb': list(graph_dict.keys()),
            'mse': list(graph_dict.values())
        })
        df_graph['mse_shift'] = df_graph.mse.shift(periods=-1)
        df_graph['proportion'] = round(100*(df_graph['mse']-df_graph['mse_shift'])/df_graph['mse'])
        print(df_graph)
        mse_min = df_graph[df_graph['proportion']>=thresh].iloc[-1, 1]
        pred_min = df_graph[df_graph['proportion']>=thresh].iloc[-1, 0]
        print(f'mse_min:{mse_min},     pred_min:{pred_min}')
        fig, ax = plt.subplots(figsize=size)
        ax.scatter(x=list(graph_dict.keys()), y=list(graph_dict.values()))
        ax.plot(list(graph_dict.keys()), np.repeat(mse_min, repeats=len(graph_dict)), color='red', linestyle='-.', alpha=.5,label=f'mse:{mse_min}')
        ax.axvline(x=pred_min, color='red', linestyle='--', label=f'predictor_numb:{pred_min}')
        ax.legend(loc='best')
        ax.set_xlabel('Predictor number')
        ax.set_ylabel('MSE')
        ax.set_xlim(xmin=.5, xmax=max(graph_dict.keys())+1)
        ax.set_title('MSE(predictors)')
        # sort by predictor number then select the one for which we have the last larger decreasing in the MSE
        plt.show()
    return {'best_models': model_fwd_total, 'all_models':df_fwd_total}


def compute_rmse_r2(dfo, name_y, estimator=LinearRegression(),
                      powerTransf = False, cols_boxCox=[], method='box-cox', pt_standardize=True,
                    fct=lambda x: x, parameters=dict(), cv=5, rd=0, ncols=2, size=(10, 7)):
    '''
    <compute_rmse_r2> computes the rmse (root mean squared error) and $R^{2}$ (the explained variance rate)
    for a regression estimator, that could been piped through the Pipeline() function from sklearn.

    Parameters
    ----------
    df : DataFrame
        Data frame gathering the response and feature observations.
    name_y : str
        Name of the response in <df>.
    estimator: sklearn estimator
        Either regression estimator. The default is LinearRegression().
    fct : function
        Any function to apply to the response to compute a transformed rmse after the fitting.
        The default is lambda x: x (the identity function).
    parameters : dictionary
        Dictionary that takes parameters of the estimator as keys and their value as values. Note that we have to respect the syntax of piped estimators.
        The default is dict().
    cv : int
        Split number for the estimation of rmse and r2 with cross-validation procedure.
        The default is 5.
    rd: int
        Random state. The default is 0.
    ncols: int
        Number of columns to display subplots.
    size: (int, int)
        Tuple respectively representing width and height.
        The default is (15, 10).

    Returns
    -------
    Dictionary.
    '''
    # array of two time values to evaluate the take time to run the script.
    duration = np.array(2*[-time.time()])
    df = dfo.copy()
    # array of 4 values which will be the mean and the standard deviation of both rmse and r2.
    res = {k:{'mean':np.nan, 'std':np.nan} for k in ['rmse', 'r2']}
    if not name_y in list(df):
        print(f'{name_y} is not in the column list')
    else:
        col_list = list(df)
        # inserts the response column to the very first place.
        col_list.insert(0, col_list.pop(col_list.index(name_y)))
        
        # handle pipe estimator.        
        if type(estimator) == sklearn.pipeline.Pipeline: 
            steps_pipe = estimator.named_steps.copy()
            steps_with_params = steps_pipe.copy()
            for k in parameters:
                name_estimator = k.split('__')[0]
                name_parameter = k.split('__')[1]
                print(name_estimator)
                steps_with_params[name_estimator] =\
                    steps_pipe[name_estimator].set_params(
                        **{name_parameter: parameters[k]})
        # build new estimator handling parameters given in argument.
            estimator_2 = Pipeline(list(steps_with_params.items()))
        else:
            estimator_2 = estimator.set_params(**parameters) 
                
        # apply one of the two power transformations provided by Scikit-Learn.
        col_to_transf = list(df.reindex(columns=cols_boxCox).dropna(axis=1))
        df_transf = df.copy()
        if col_to_transf:            
            power_t = PowerTransformer(method=method, standardize=pt_standardize)
            power_t.fit(df.loc[:, col_to_transf])
            df_transf.loc[:, col_to_transf] = power_t.transform(df_transf.loc[:, col_to_transf])
            print(f'Applying {method} transformation on the following columns: {col_to_transf}')
            
        rmse_list, r2_list = [], []
        
        # build plot frame.
        f = lambda x: int(x)+1 if (x - int(x))>=.5 else int(x) # an alternative of the round function
        nrows = f(cv/ncols)
        ind = [[i, j] for i,j in itertools.product(np.arange(nrows),np.arange(ncols))]
        fig, axs = plt.subplots(nrows, ncols, figsize=size)
        iter_var = 0
        
        # cross-validation process, it shuffle the row data and make a K-fold. 
        kf = KFold(n_splits=cv, shuffle=True, random_state=0)
        for train, test in kf.split(df.iloc[:, 1:]):
            X_train, X_test = df_transf.iloc[train, 1:], df_transf.iloc[test, 1:]
            # y_test is left unchanged, then it is extracted from df not df_transf.
            y_train, y_test = df_transf.iloc[train, 0], df.iloc[test, 0] 
            # to "reinitialize" estimator at each loop to avoid that the estimator be influenced by the
            # previous fit.
            estimator_to_use = estimator_2
            estimator_to_use.fit(X_train, y_train)
            y_pred = estimator_to_use.predict(X_test)
            # remove transformation of the response to compute the mse.
            if name_y in col_to_transf:
                y_pred_transf = fct(power_t.inverse_transform(
                    np.concatenate(len(col_to_transf)*[y_pred.reshape(-1, 1)], axis=1)))[:,0]
            else:
                y_pred_transf = fct(y_pred)
            # recall that fct is a function given in argument, by default is the identity.
            y_test_transf = fct(y_test)

            # compute rmse and r2.
            rmse = np.sqrt(((y_test_transf - y_pred_transf)**2).mean())
            rmse_list.append(rmse)
            r2 = 100*(1 - ((y_test_transf - y_pred_transf)**2).sum()/\
                ((y_test_transf - y_test_transf.mean())**2).sum())
            r2_list.append(r2)
            
            # plot graphics to visualize the prediction power of the model.
            x = np.arange(len(test))
            axs[ind[iter_var][0], ind[iter_var][1]].scatter(x, y_test_transf, label='Test')
            axs[ind[iter_var][0], ind[iter_var][1]].scatter(x, y_pred_transf, label='Prediction')
            axs[ind[iter_var][0], ind[iter_var][1]].set_title(
               fr'{cv}-fold number {iter_var+1},    $rmse$ = {round(rmse, 1)},    $R^{2}={round(r2, 1)}\%$')
            axs[ind[iter_var][0], ind[iter_var][1]].legend(loc='best')
            iter_var += 1
        # information about the rmse and the r2, they are put here to dipslay them above the graphic.
        rmse_mean, rmse_std = round(np.array(rmse_list).mean(), 1), round(np.array(rmse_list).std(), 1)   
        r2_mean, r2_std = round(np.array(r2_list).mean(), 1), round(np.array(r2_list).std(), 1)
        res['rmse']['mean'], res['rmse']['std'] = rmse_mean, rmse_std
        res['r2']['mean'], res['r2']['std'] = r2_mean, r2_std    
        print(res)
       
        # plot the graph.
        plt.suptitle(f'Estimator: {estimator_2}.')
        plt.tight_layout()
        plt.show()
    
    duration[1] = time.time()
    quotient_hour, remainder_hour = duration.sum()// 3600, duration.sum() % 3600
    quotient_minute, remainder_minute= remainder_hour // 60, remainder_hour % 60
    print(f"Elapsed time: {int(quotient_hour)}h {int(quotient_minute)}min {int(remainder_minute)}s")
    return res