import time

import numpy as np

import multiprocessing
from itertools import product

from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_predict, StratifiedKFold
from sklearn.metrics import f1_score, precision_score, recall_score, roc_curve, auc, roc_auc_score

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import AdaBoostClassifier

import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt
import numpy as np

import matplotlib as mpl
import importlib
import itertools

from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler
from sklearn.utils import resample

from collections import Counter

from pathlib import Path

import IPython
#IPython.embed()

from sklearn.feature_selection import RFECV, RFE

from collections import namedtuple, Counter

from sklearn.model_selection import learning_curve


# from https://scikit-learn.org/stable/auto_examples/model_selection/plot_learning_curve.html
''' Example of use
scoring = 'f1_weighted'
modelname = 'Logit'
plot_learning_curve(clf, modelname+', scoring='+scoring, X, Y, scoring=scoring, axes=None, ylim=None, cv=None, n_jobs=None, train_sizes=np.linspace(.1, 1.0, 5), filename=fig_folder+'/learning_curve_'+modelname+'_scoring_'+scoring+'.png')
'''
def plot_learning_curve(estimator, title, X, y, scoring='roc_auc', axes=None, ylim=None, cv=3, n_jobs=None, train_sizes=np.linspace(.4, 1.0, 5), filename=None):

    if axes is None:
        plt.figure(figsize=(10, 5))

    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")

    train_sizes, train_scores, test_scores = learning_curve(estimator, X, y, cv=cv, n_jobs=n_jobs,
                       train_sizes=train_sizes,
                       scoring=scoring,
                       shuffle=True,
                       )
                       
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)

    # Plot learning curve
    plt.grid()
    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                         train_scores_mean + train_scores_std, alpha=0.1,
                         color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                         test_scores_mean + test_scores_std, alpha=0.1,
                         color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
                 label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
                 label="Cross-validation score")
    plt.legend(loc="best")

    if filename:
      plt.savefig(filename)
    else:
      plt.show()


###
def eliminate_cols_nan(df, thr):
  # thr entre 0 e 1
  n_records = df.shape[0]
  
  if (type(thr) is float) and (thr <= 1 and thr >= 0):
    n_nans_max = int(n_records * thr)
  elif type(thr) is int:
    n_nans_max = int(thr)
  else:
    print ('Variavel threshold nao reconhecida.')
    return None
  
  feats_to_delete = list()

  for col in df.columns:
    try:
      if np.isnan(df[col]).sum() > n_nans_max:
        feats_to_delete.append(col)
    except:
      print ('Erro na col '+str(col)+'. Coluna ignorada.')

  df = df.drop(columns=feats_to_delete)

  return df, feats_to_delete

###  
def eliminate_records_nan(df, thr, reset_index=False):
  # thr entre 0 e 1
  n_feat = df.shape[1] # as feats eh df.shape[1]
  
  df = df.reset_index(drop=True)

  if (type(thr) is float) and (thr <= 1 and thr >= 0):
    n_feat_nan_max = int(n_feat * thr)
  elif type(thr) is int:
    n_feat_nan_max = int(thr)
  else:
    print ('Variavel threshold nao reconhecida.')
    return None


  records_to_delete = (np.isnan(df).sum(axis=1)>n_feat_nan_max).to_numpy().nonzero()[0]

  #IPython.embed()

  df = df.drop(records_to_delete, axis=0)

  return df, records_to_delete



def cros_val(clf, X, Y, metrics=['accuracy', 'recall'], resampling='undersampling', cv=3, multiclass=False, seed=None, standardize=False, bootstrap=False):

  assert isinstance(seed, int)

  n_classes = len(set(Y))

  # falta instanciar
  return_named_tuple = namedtuple('return_named_tuple', ('clf', 'smote', 'cv', 'accuracy', 'recall', 'auc', 'f1_score'))

  # laco dos folds
  cv_folds = StratifiedKFold(n_splits=cv, shuffle=True, random_state=seed)

  scores_f1 = list()
  scores_precision = list()
  scores_recall = list()
  scores_auc = list()

  Y_pred_proba_geral = np.zeros(shape=Y.shape)
  Y_pred_geral = np.zeros(shape=Y.shape)

  for train, test in cv_folds.split(X, Y):

    X_train = X[train].copy()

    # essa linha h soh pra setar caso o augmented seja None
    if resampling == None:
      X_train_aug, Y_train_aug = X_train, Y[train]

    # agora eh necessario checar o aumento de dados
    if resampling == 'smote':
      X_train_aug, Y_train_aug = SMOTE().fit_resample(X_train, Y[train])

    if resampling == 'undersampling':
      s = s + 'aug check: undersampling\n'
      rus = RandomUnderSampler()
      X_train_aug, Y_train_aug = rus.fit_resample(X_train, Y[train])

    if resampling == 'oversampling':
      s = s + 'aug check: oversampling\n'
      ros = RandomOverSampler()
      X_train_aug, Y_train_aug = ros.fit_resample(X_train, Y[train])

    if bootstrap:
      #X_train_aug, Y_train_aug
      # sorteia nrows indices com replacement
      np.random.seed(seed)
      idxs_rows_bootstrap = np.random.choice(np.arange(X_train_aug.shape[0]), size=X_train_aug.shape[0])
      X_train_aug = X_train_aug[idxs_rows_bootstrap, :]
      Y_train_aug = Y_train_aug[idxs_rows_bootstrap]

    X_test = X[test].copy() # soh pra definir X_test quando std=False   
    if standardize:
      mean_ = X_train_aug.mean(axis=0)
      std_ = X_train_aug.std(axis=0)
      X_train_aug = ((X_train_aug-mean_)/std_).copy()
      X_test = ((X[test]-mean_)/std_).copy()

      # sometimes we have NaN when standardizing
      idxs_rows_nans_X_train_aug = np.isnan(X_train_aug).sum(axis=1).astype('bool')
      X_train_aug = X_train_aug[~idxs_rows_nans_X_train_aug, :]

      #
      idxs_rows_nans_X_test = np.isnan(X_test).sum(axis=1).astype('bool')
      X_test = X_test[~idxs_rows_nans_X_test, :]


    # treino e predicoes
    clf.fit(X_train_aug, Y_train_aug)
    Y_pred = clf.predict(X_test)
    Y_true = Y[test]

    # guarda na matrizona geral
    if multiclass:
      roc_temporario_ = 0
      for i in range(1, n_classes+1):
        Y_pred_proba_geral[test] = clf.predict_proba(X_test)[:, i-1] # pegar o proba 1 aqui 
        roc_temporario_ = roc_temporario_ + roc_auc_score((Y_true==i).astype('int'), Y_pred_proba_geral[test])
      roc_temporario_ = roc_temporario_ / n_classes
      scores_auc.append(roc_temporario_)
    else:
      Y_pred_proba_geral[test] = clf.predict_proba(X_test)[:, 1] # pegar o proba 1 aqui 
      scores_auc.append(roc_auc_score(Y_true, Y_pred_proba_geral[test]))


    #Y_pred_proba_geral[test] = clf.predict_proba(X[test])[:, 1].copy() # pegar o proba 1 aqui 
    Y_pred_geral[test] = clf.predict(X_test).copy()


    # guardando os scores
    #scores_f1.append(f1_score(Y_true, Y_pred, average=average))
    #scores_precision.append(precision_score(Y_true, Y_pred, average=average))
    #scores_recall.append(recall_score(Y_true, Y_pred, average=average))
    #scores_auc.append(roc_auc_score(Y_true, Y_pred_proba_geral[test]))

    # guardando as confmatrix de cada fold
    confm = confusion_matrix(Y_true, Y_pred) 
    #s = s + str(confm) + '\n'

  scores_f1 = np.array(scores_f1)
  scores_precision = np.array(scores_precision)
  scores_recall = np.array(scores_recall)
  scores_auc = np.array(scores_auc)

  # conf matrix
  #Y_pred = cross_val_predict(clf, X, Y, cv=cv)
  #Y_true = Y.copy()
  #confm = confusion_matrix(Y_true, Y_pred_geral)

  r = return_named_tuple (clf, resampling, cv, scores_precision, scores_recall, scores_auc, scores_f1)
  
  return r



################# PARALELO

def get_model_ml_(params):

  #print (params)

  if params.split()[0] == 'svc':
    if params.split()[1] == 'none':
      C_parameter=1.0
    else:
      C_parameter=float(params.split()[1])

    if params.split()[2] == 'none':
      gamma_parameter='scale'
    else:
      gamma_parameter=float(params.split()[2])
      
    clf = svm.SVC(C=C_parameter, gamma=gamma_parameter, kernel='rbf', probability=True)

  elif params.split()[0] == 'rfc':
    if len(params.split()) != 4:
      if params.split()[1].lower()  == 'none':
        clf = RandomForestClassifier(n_estimators=100, max_depth=None, min_samples_split=int(params.split()[2]), n_jobs=1)
      else:        
        clf = RandomForestClassifier(n_estimators=100, max_depth=int(params.split()[1]), min_samples_split=int(params.split()[2]), n_jobs=1)
    else: # aqui eh pra entrar o n_est
      try:
        max_depth = int(params.split()[1])
      except:
        max_depth = None
      try:
        min_samples_split = int(params.split()[2])
      except:
        print ('Erro '+params)
        sys.exit(0)
      try:
        n_estimators = int(params.split()[3])
      except:
        print ('Erro '+params)
        sys.exit(0)
      clf = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, min_samples_split=min_samples_split, n_jobs=1)

  elif params.split()[0] == 'logit':
    if params.split()[1].lower()  == 'none':
      clf = LogisticRegression(penalty='l2', solver='lbfgs', multi_class='multinomial', C=1.0)
    else:
      clf = LogisticRegression(penalty='l2', solver='lbfgs', multi_class='multinomial', C=float(params.split()[1]))

  elif params.split()[0] == 'ada':
      clf = AdaBoostClassifier(n_estimators=int(params.split()[1]), learning_rate=float(params.split()[2]))

  else:
    print ('Nao foi identificado o classificador. {}'.format(params))
    
  return clf

### function to assist parallel implementation
def f(params, X, Y, cv, resampling, seed, standardize, bootstrap):

  print ('rodando params: {}'.format(params))

  clf = get_model_ml_(params)
  
  if len(set(Y)) > 2:
    multiclass = True
  elif len(set(Y)) == 2:
    multiclass = False
  else:
    print ('Erro! flag multiclass.')

  return_ = cros_val(clf, X, Y, metrics=['accuracy', 'recall'], resampling=resampling, cv=cv, multiclass=multiclass, seed=seed, standardize=standardize, bootstrap=bootstrap)

  return return_.auc.mean()

  print ('params {} finalizado.'.format(params))

  #params_scores[k] = return_.auc.mean()
  #params_std_scores[k] = return_.auc.std()



def grid_search_nested_parallel(X, Y, cv=3, writefolder=None, n_jobs=30, resampling='undersample', roc_curve_output=False, standardize=False, bootstrap=False):

  n_classes = len(set(Y))

  seed = int(time.time())
  

  if roc_curve_output:
    mean_fpr = np.linspace(0, 1, 100)
    tprs = list()


  if len(set(Y)) > 2:
    multiclass = True
  elif len(set(Y)) == 2:
    multiclass = False
  else:
    print ('Erro! flag multiclass.')

  # laco dos folds
  cv_folds = StratifiedKFold(n_splits=cv, shuffle=True, random_state=int(time.time()))

  Y_pred_proba_geral = np.zeros(shape=Y.shape)
  Y_pred_geral = np.zeros(shape=Y.shape)


  '''
  # SVC
  C_list = np.logspace(np.log10(0.1), np.log10(4000), num=10)
  C_list = [str(x) for x in C_list]
  gamma_list = np.logspace(np.log10(0.0001), np.log10(4.2), num=10)
  gamma_list = [str(x) for x in gamma_list]
  svc_kernel = 'rbf'
  svc_params_list = list(itertools.product(C_list, gamma_list))
  svc_params_list = ['svc '+' '.join(x) for x in svc_params_list]
  # Adaboost
  ada_n_estimators = ['3', '5', '20', '60']
  ada_learning_rate = ['1.0', '0.1', '4.0', '5.0']
  ada_params_list = list(itertools.product(ada_n_estimators, ada_learning_rate))
  ada_params_list = ['ada '+' '.join(x) for x in ada_params_list]
  
  # RF
  max_depth_list = ['2', '4', '8', '16', 'None']
  rfc_params_list = ['rfc '+' '+x for x in max_depth_list]
  # Logit
  C_list = np.logspace(np.log10(0.001), np.log10(200), num=80)
  C_list = [str(x) for x in C_list]
  logit_params_list = ['logit '+' '+x for x in C_list]
  '''
  
  # SVC
  C_list = np.logspace(np.log10(0.1), np.log10(1000), num=20)
  C_list = [str(x) for x in C_list]
  gamma_list = np.logspace(np.log10(0.0001), np.log10(1), num=20)
  gamma_list = [str(x) for x in gamma_list]

  svc_kernel = 'rbf'
  svc_params_list = list(itertools.product(C_list, gamma_list))
  svc_params_list = ['svc '+' '.join(x) for x in svc_params_list]
  
  # RF
  #max_depth_list = ['1', '2', '3', '4', '5', '8', '10', '12', '16', '20', '26', '34', '40', '60', 'None']
  max_depth_list = ['1', '2', '4', '8', '20','60', 'None']
  min_samples_split_list = ['2', '4', '8', '20']

  rfc_params_list = list(itertools.product(max_depth_list, min_samples_split_list))
  rfc_params_list = ['rfc '+' '.join(x) for x in rfc_params_list]
  #rfc_params_list = ['rfc '+' '+x for x in max_depth_list]

  # Logit
  C_list = np.logspace(np.log10(0.0001), np.log10(10), num=50)
  C_list = [str(x) for x in C_list]
  logit_params_list = ['logit '+' '+x for x in C_list]

  params_list = svc_params_list + rfc_params_list + logit_params_list
   
  params_scores = np.zeros((len(params_list),))
  params_std_scores = np.zeros((len(params_list),))

  s = ''

  if (writefolder != None):
    plt.figure(figsize=(18,10))
    plt.ylabel('AUC score')
    plt.xlabel('Parameter set number')
    plt.title('')


  best_params_all = list()
  best_auc_scores_holdout = list()
  
  #plt.figure(figsize=(12,8))

  for i, (train, test) in enumerate(cv_folds.split(X, Y)):


    parameters_vector_total = [(x, X[train].copy(), Y[train].copy(), cv, resampling, seed, standardize, bootstrap) for x in params_list]

    params_scores_partial = list()
    for parameters_vector in [parameters_vector_total[j:j+n_jobs] for j in range(0, len(parameters_vector_total), n_jobs)]:
      with multiprocessing.Pool(processes=n_jobs) as pool:
        params_scores_partial = params_scores_partial + pool.starmap(f, parameters_vector)

    params_scores = np.array(params_scores_partial)
  
    best_params = params_list[params_scores.argmax()]
    best_params_all.append(best_params)
    best_params_idx = params_scores.argmax()

    clf = get_model_ml_(best_params)
    
    ##
    if resampling == None or resampling == 'None':
      X_train_aug, Y_train_aug = X[train], Y[train]

    # agora eh necessario checar o aumento de dados
    if resampling == 'smote':
      X_train_aug, Y_train_aug = SMOTE().fit_resample(X[train], Y[train])

    if resampling == 'undersampling':
      rus = RandomUnderSampler()
      X_train_aug, Y_train_aug = rus.fit_resample(X[train], Y[train])

    if resampling == 'oversampling':
      ros = RandomOverSampler()
      X_train_aug, Y_train_aug = ros.fit_resample(X[train], Y[train])

    if bootstrap:
      # sorteia nrows indices com replacement
      #np.random.seed(seed)
      idxs_rows_bootstrap = np.random.choice(np.arange(X_train_aug.shape[0]), size=X_train_aug.shape[0])
      X_train_aug = X_train_aug[idxs_rows_bootstrap, :]
      Y_train_aug = Y_train_aug[idxs_rows_bootstrap]


    X_test = X[test].copy() # soh pra definir X_test quando std=False   
    if standardize:
      mean_ = X_train_aug.mean(axis=0)
      std_ = X_train_aug.std(axis=0)
      X_train_aug = ((X_train_aug-mean_)/std_).copy()
      X_test = ((X[test]-mean_)/std_).copy()

      # sometimes we have NaN when standardizing
      idxs_rows_nans_X_train_aug = np.isnan(X_train_aug).sum(axis=1).astype('bool')
      X_train_aug = X_train_aug[~idxs_rows_nans_X_train_aug, :]

      #
      idxs_rows_nans_X_test = np.isnan(X_test).sum(axis=1).astype('bool')
      X_test = X_test[~idxs_rows_nans_X_test, :]



    # treino e predicoes
    clf.fit(X_train_aug, Y_train_aug)
    Y_pred = clf.predict(X_test)
    #Y_true = Y[test]
    ##
    
    
    #clf.fit(X[train], Y[train])
    Y_true = Y[test]
    

    if writefolder:
      s = s + '####### FOLD {} of {} #####\n'.format(i+1, cv)
      for param, score, std in zip(params_list, params_scores, params_std_scores):
        s = s + 'param: {}, score: {:.3} ({:.4})\n'.format(param, score, std)
      s = s + '* Best params: {}, idx: {} - score: {:.3}\n'.format(best_params, best_params_idx, params_scores[best_params_idx])
      
      s = s + '*** Evaluation phase ***\n'
      
      if multiclass:
        roc_temporario_ = 0
        for j in range(1, n_classes+1):
          Y_pred_proba_geral[test] = clf.predict_proba(X[test])[:, j-1] # pegar o proba 1 aqui 
          roc_temporario_ = roc_temporario_ + roc_auc_score((Y_true==j).astype('int'), Y_pred_proba_geral[test])
        roc_temporario_ = roc_temporario_ / n_classes
        auc_ = roc_temporario_
      else:
        auc_ = roc_auc_score(Y[test], clf.predict_proba(X[test])[:, 1])

        if roc_curve_output:
          Y_pred_proba = clf.predict_proba(X[test])
          fpr_, tpr_, thresholds_ = roc_curve(Y_true, Y_pred_proba[:, 1])

          interp_tpr = np.interp(mean_fpr, fpr_, tpr_)
          interp_tpr[0] = 0.0
          tprs.append(interp_tpr)


      s = s + 'AUC Ev. score: {:.3}\n'.format(auc_)
      s = s + '###########################\n'

    best_auc_scores_holdout.append(auc_)


    if (writefolder != None):
      plt.plot(params_scores, 's-', label='fold {}'.format(i+1))
      plt.plot([0,len(params_scores)], [auc_, auc_], label='auc fold {}: {:.3}'.format(i+1, auc_))


  if roc_curve_output:
    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0

  
  file_ = open(writefolder+'/'+'report_nested_cross_validation_hyperparameter_tuning.txt', 'w')
  file_.write(s)
  file_.close()

  if (writefolder != None):
    plt.legend(loc="lower right")
    plt.savefig(writefolder+'/'+'nested_cross_validation_scores.png', dpi=100)
  

  if roc_curve_output:
    return best_params_all, best_auc_scores_holdout, (mean_fpr, mean_tpr)
  else:
    return best_params_all, best_auc_scores_holdout



def part_feature_study(feature, df_a, y_name, pathfile=None, resampling='undersampling'):

  type_var = None

  # primeiro decidir se a feature eh categorica ou continua
  feature_subgroup = set(df_a[feature])
  if len(feature_subgroup) < 8: # criterio meio abstrato aqui
    type_var = 'Cat'

    feature_subgroup = set(df_a[feature])

  else:
    type_var = 'Cont'
    nbins = 4
    x = np.linspace(0, 100, nbins+1)
    percentis = np.percentile(df_a[feature], x) # nbins + 1 comeca no zero

    hist, edges = np.histogram(df_a[feature].values, bins=percentis)
    
    feature_subgroup = list()
    
    for i in range(nbins):
      feature_subgroup.append( np.logical_and((df_a[feature] > edges[i]) , (df_a[feature] < edges[i+1])) )

  # criando uma figura para todos plots    
  plt.figure(figsize=(15,8))

  for i, partial_feat in enumerate(feature_subgroup):

    if type_var == 'Cat':
      df_b = df_a[df_a[feature]==partial_feat]

    elif type_var == 'Cont':
      df_b = df_a[partial_feat]


    #
    Y = df_b[y_name].values
    df_x = df_b.drop(columns=y_name)
    X = df_x.values

    df_b = df_b.drop(columns=feature)
    # 
        
    clf = RandomForestClassifier(n_estimators=40)
    if (Y==1).sum()>10:
      auc_ = cros_val(clf, X, Y, metrics=['accuracy', 'recall'], resampling=resampling, cv=3, multiclass=False).auc.mean()
    else:
      auc_ = 'na'
    #

    # correlacao com o y
    plt.clf()
    #plt.figure(figsize=(15,8))
    corr_ = df_b.corr()[y_name].values
    x = range(len(corr_))
    plt.plot(x, corr_, 'o--')
    plt.plot([0, x[-1]], [0, 0], '-')
    plt.xticks(x, list(df_b.corr()[y_name].index), rotation=75)
    #plt.tight_layout()
    plt.title('AUC RF: {:.3} / quant. dados: {} / quant. ocorr {}:'.format(auc_, len(df_b), (Y==1).sum()))
    if pathfile:
      if type_var == 'Cat':
        plt.savefig(pathfile+'/'+feature+'_'+str(partial_feat)+'_'+'corr_with'+y_name+'.png', dpi=120)
      elif type_var == 'Cont':      
        plt.savefig(pathfile+'/'+feature+'_'+str(edges[i])+'_'+str(edges[i+1])+'_'+'corr_with'+y_name+'.png', dpi=120)
    else:
      plt.show()


def rfe(X, Y, pathfile=None, labels=[]):

  linewidth = 4
  figsize = (12, 8)

  plt.clf()
  plt.figure(figsize=figsize)

  plt.ylabel('ranking (menor eh melhor)')
  plt.xlabel('features')
  x_ = range(len(labels))
  plt.xticks(x_, labels, rotation=60)

  # estimador 1
  estimator = svm.SVC(kernel="linear")
  selector = RFE(estimator, 1, step=1)
  selector = selector.fit(X, Y)
  print (selector.ranking_)

  plt.plot (x_, selector.ranking_, linewidth=linewidth, label='SVC linear')

  # est 2
  estimator = RandomForestClassifier(n_estimators=100)
  selector = RFE(estimator, 1, step=1)
  selector = selector.fit(X, Y)
  print (selector.ranking_)

  plt.plot (x_, selector.ranking_, linewidth=linewidth, label='RFC')

  # est 3
  estimator = RandomForestClassifier(n_estimators=1000, max_depth=2, n_jobs=10)
  selector = RFE(estimator, 1, step=1)
  selector = selector.fit(X, Y)
  print (selector.ranking_)

  plt.plot (x_, selector.ranking_, linewidth=linewidth, label='RFC-2')

  # est 4
  estimator = LogisticRegression(solver='lbfgs', multi_class='multinomial')
  selector = RFE(estimator, 1, step=1)
  selector = selector.fit(X, Y)
  print (selector.ranking_)

  plt.plot (x_, selector.ranking_, linewidth=linewidth, label='Logit')

  plt.legend(loc="upper right")

  if pathfile:
    plt.savefig(pathfile+'/rfe_class_weight_none.png')
  else:
    plt.show()

  ## balanced=True

  plt.clf()
  plt.figure(figsize=figsize)

  plt.ylabel('ranking (menor eh melhor)')
  plt.xlabel('features')
  x_ = range(len(labels))
  plt.xticks(x_, labels, rotation=60)

  # estimador 1
  estimator = svm.SVC(kernel="linear", class_weight='balanced')
  selector = RFE(estimator, 1, step=1)
  selector = selector.fit(X, Y)
  print (selector.ranking_)

  plt.plot (x_, selector.ranking_, linewidth=linewidth, label='SVC linear')

  # est 2
  estimator = RandomForestClassifier(n_estimators=100, class_weight='balanced')
  selector = RFE(estimator, 1, step=1)
  selector = selector.fit(X, Y)
  print (selector.ranking_)

  plt.plot (x_, selector.ranking_, linewidth=linewidth, label='RFC')

  # est 3
  estimator = RandomForestClassifier(n_estimators=600, max_depth=2, class_weight='balanced', n_jobs=10)
  selector = RFE(estimator, 1, step=1)
  selector = selector.fit(X, Y)
  print (selector.ranking_)

  plt.plot (x_, selector.ranking_, linewidth=linewidth, label='RFC-2')

  # est 4
  estimator = LogisticRegression(solver='lbfgs', multi_class='multinomial', class_weight='balanced')
  selector = RFE(estimator, 1, step=1)
  selector = selector.fit(X, Y)
  print (selector.ranking_)

  plt.plot (x_, selector.ranking_, linewidth=linewidth, label='Logit')

  plt.legend(loc="upper right")

  if pathfile:
    plt.savefig(pathfile+'/rfe_class_weigths_balanced.png')
  else:
    plt.show()


  ## Undersampling

  rus = RandomUnderSampler()
  Xus, Yus = rus.fit_resample(X, Y)


  plt.clf()
  plt.figure(figsize=figsize)

  plt.ylabel('ranking (menor eh melhor)')
  plt.xlabel('features')
  x_ = range(len(labels))
  plt.xticks(x_, labels, rotation=60)

  # estimador 1
  estimator = svm.SVC(kernel="linear", class_weight='balanced')
  selector = RFE(estimator, 1, step=1)
  selector = selector.fit(Xus, Yus)
  print (selector.ranking_)

  plt.plot (x_, selector.ranking_, linewidth=linewidth, label='SVC linear')

  # est 2
  estimator = RandomForestClassifier(n_estimators=100, class_weight='balanced')
  selector = RFE(estimator, 1, step=1)
  selector = selector.fit(Xus, Yus)
  print (selector.ranking_)

  plt.plot (x_, selector.ranking_, linewidth=linewidth, label='RFC')

  # est 3
  estimator = RandomForestClassifier(n_estimators=600, max_depth=2, class_weight='balanced', n_jobs=10)
  selector = RFE(estimator, 1, step=1)
  selector = selector.fit(Xus, Yus)
  print (selector.ranking_)

  plt.plot (x_, selector.ranking_, linewidth=linewidth, label='RFC-2')

  # est 4
  estimator = LogisticRegression(solver='lbfgs', multi_class='multinomial', class_weight='balanced')
  selector = RFE(estimator, 1, step=1)
  selector = selector.fit(Xus, Yus)
  print (selector.ranking_)

  plt.plot (x_, selector.ranking_, linewidth=linewidth, label='Logit')

  plt.legend(loc="upper right")

  if pathfile:
    plt.savefig(pathfile+'/rfe_undersampling.png')
  else:
    plt.show()


####################




def rfe_cv(list_models_params, X, Y, cv=3, pathfile=None, labels=[]):

  linewidth = 4

  s = '######## Report RFE CV ######\n'

  # Plot number of features VS. cross-validation scores
  plt.figure(figsize=(14,8))
  plt.xlabel("Number of features selected")
  plt.ylabel("Cross validation score")

  # Create the RFE object and compute a cross-validated score.
  '''
  svc = svm.SVC(kernel="linear", class_weight='balanced', probability=False)
  rfecv = RFECV(estimator=svc, step=1, cv=cv, scoring='accuracy')
  rfecv.fit(X, Y)
  plt.plot(range(1, len(rfecv.grid_scores_) + 1), rfecv.grid_scores_, linewidth=linewidth, label='SVC score accuracy')
 
  s = s + '#############################\n'
  s = s + 'SVC score accuracy\n'
  s = s + 'features selecionadas: {}\n'.format(' '.join(labels[rfecv.support_]))
  s = s + 'Ranking:\n'
  for line in  np.array(sorted(list(zip(rfecv.ranking_,labels)))):
    s = s + str(line) + '\n'
  '''

  '''
  svc = svm.SVC(kernel="linear", class_weight='balanced')
  rfecv = RFECV(estimator=svc, step=1, cv=cv, scoring='recall')
  rfecv.fit(X, Y)
  plt.plot(range(1, len(rfecv.grid_scores_) + 1), rfecv.grid_scores_, linewidth=linewidth, label='SVC score recall')
  s = s + '#############################\n'
  s = s + 'SVC score recall\n'
  s = s + 'features selecionadas: {}\n'.format(' '.join(labels[rfecv.support_]))
  s = s + 'Ranking:\n'
  for line in  np.array(sorted(list(zip(rfecv.ranking_,labels)))):
    s = s + str(line) + '\n'
  '''

  if list_models_params:
    for model in list_models_params:
      modelname = model.replace(' ','_')
      clf = get_model_ml_(model)
      clf.fit(np.array([1,0,0,0]).reshape(-1, 1), [1,0,0,0]) # isso eh soh para abrir o atributo feature_importances
      if not (hasattr(clf, 'coef_') or hasattr(clf, 'feature_importances_')):
        print ('modelo '+model+' nao suporta feature importance')
        continue
      
      # Create the RFE object and compute a cross-validated score.
      #svc = svm.SVC(kernel="linear", class_weight='balanced', probability=True)
      rfecv = RFECV(estimator=clf, step=1, cv=cv, scoring='roc_auc')
      rfecv.fit(X, Y)
      plt.plot(range(1, len(rfecv.grid_scores_) + 1), rfecv.grid_scores_, linewidth=linewidth, label='AUC-'+modelname)


      s = s + '#############################\n'
      s = s + ''+modelname+' score auc\n'
      s = s + 'features selecionadas: {}\n'.format(' '.join(labels[rfecv.support_]))
      s = s + 'Ranking:\n'
      for line in  np.array(sorted(list(zip(rfecv.ranking_,labels)))):
        s = s + str(line) + '\n'

  
  # Create the RFE object and compute a cross-validated score.
  svc = svm.SVC(kernel="linear", class_weight='balanced', probability=True)
  rfecv = RFECV(estimator=svc, step=1, cv=cv, scoring='roc_auc')
  rfecv.fit(X, Y)
  plt.plot(range(1, len(rfecv.grid_scores_) + 1), rfecv.grid_scores_, linewidth=linewidth, label='AUC-SVC Linear')


  s = s + '#############################\n'
  s = s + 'SVC score auc\n'
  s = s + 'features selecionadas: {}\n'.format(' '.join(labels[rfecv.support_]))
  s = s + 'Ranking:\n'
  for line in  np.array(sorted(list(zip(rfecv.ranking_,labels)))):
    s = s + str(line) + '\n'


  # Create the RFE object and compute a cross-validated score.
  '''
  estimator = LogisticRegression(solver='lbfgs', multi_class='multinomial', class_weight='balanced')
  rfecv = RFECV(estimator=estimator, step=1, cv=cv, scoring='accuracy')
  rfecv.fit(X, Y)
  plt.plot(range(1, len(rfecv.grid_scores_) + 1), rfecv.grid_scores_, linewidth=linewidth, label='Logit score accuracy')
  s = s + '#############################\n'
  s = s + 'Logit score accuracy\n'
  s = s + 'features selecionadas: {}\n'.format(' '.join(labels[rfecv.support_]))
  s = s + 'Ranking:\n'
  for line in  np.array(sorted(list(zip(rfecv.ranking_,labels)))):
    s = s + str(line) + '\n'
  '''
  
  ####
  estimator = LogisticRegression(solver='lbfgs', multi_class='multinomial', class_weight='balanced')
  rfecv = RFECV(estimator=estimator, step=1, cv=cv, scoring='roc_auc', n_jobs=10)
  rfecv.fit(X, Y)
  plt.plot(range(1, len(rfecv.grid_scores_) + 1), rfecv.grid_scores_, linewidth=linewidth, label='AUC-Logit')
  s = s + '#############################\n'
  s = s + 'Logit score recall\n'
  s = s + 'features selecionadas: {}\n'.format(' '.join(labels[rfecv.support_]))
  s = s + 'Ranking:\n'
  for line in  np.array(sorted(list(zip(rfecv.ranking_,labels)))):
    s = s + str(line) + '\n'

  ###
  '''
  estimator = LogisticRegression(solver='lbfgs', multi_class='multinomial', class_weight='balanced')
  rfecv = RFECV(estimator=estimator, step=1, cv=cv, scoring='recall')
  rfecv.fit(X, Y)
  plt.plot(range(1, len(rfecv.grid_scores_) + 1), rfecv.grid_scores_, linewidth=linewidth, label='Logit score recall')
  s = s + '#############################\n'
  s = s + 'Logit score recall\n'
  s = s + 'features selecionadas: {}\n'.format(' '.join(labels[rfecv.support_]))
  s = s + 'Ranking:\n'
  for line in  np.array(sorted(list(zip(rfecv.ranking_,labels)))):
    s = s + str(line) + '\n'
  '''
  
  estimator = RandomForestClassifier(n_estimators=2000, max_depth=2, class_weight='balanced')
  rfecv = RFECV(estimator=estimator, step=1, cv=cv, scoring='roc_auc', n_jobs=20)
  rfecv.fit(X, Y)
  plt.plot(range(1, len(rfecv.grid_scores_) + 1), rfecv.grid_scores_, linewidth=linewidth, label='AUC-RF-MaxDepth-2')
  s = s + '#############################\n'
  s = s + 'RF maxdepth-2 score recall\n'
  s = s + 'features selecionadas: {}\n'.format(' '.join(labels[rfecv.support_]))
  s = s + 'Ranking:\n'
  for line in  np.array(sorted(list(zip(rfecv.ranking_,labels)))):
    s = s + str(line) + '\n'
 
  plt.legend(loc="lower right")

  file_ = open(pathfile+'/rfe_cv_report.txt', 'w')
  file_.write(s)
  file_.close()

  if pathfile:
    plt.savefig(pathfile+'/rfe_cv_balanced_true.png')
  else:
    plt.show()

  if pathfile:
    file_ = open(pathfile+'/report_RFE_CV.txt', 'w')
    file_.write(s)
    file_.close()
  else:
    print (s)





def tchart(features, values, title=None, pathfile=None):
  num_features = len(features)
  pos = np.arange(num_features) + .5    # bars centered on the y axis
  fig, ax_right = plt.subplots(ncols=1, figsize=(14,12))
  ax_right.barh(pos, values, align='center', facecolor='steelblue')
  ax_right.set_yticks(pos)
  # x moves tick labels relative to left edge of axes in axes units
  ax_right.set_yticklabels(features, ha='right', x=0.00)
  #ax_right.set_xlabel('Proficiency')

  if title:
    plt.suptitle(title)

  plt.tight_layout()

  if pathfile:
    plt.savefig(pathfile)
  else:
    plt.show()






def checknamefolder(folder):
  fileName = Path(folder)

  if fileName.is_file():
      print ("File exist")
      # se termina com _XX , XX numero
      try:
        n = int(folder.split('_')[-1])
        folder = checknamefolder('_'.join(folder.split('_')[:-1]) + str(n+1))
        return folder
      except:
        folder = folder + '_1'
        folder = checknamefolder(folder)
        return folder
  else:
      return folder






def gerar_hists(df, fig_folder):
  fig = plt.figure(figsize=(8,5))
  for variable in df.columns:
    fig.clf()
    print ('Histograma '+variable)
    df[variable].plot.hist(stacked=False, bins=20)
    plt.xlabel(variable)
    plt.ylabel('ocurrences')
    variable = variable.replace(' ', '_')
    variable = variable.replace('/', '_')
    if fig_folder != None:
      plt.savefig('./'+fig_folder+'/histograma_var_'+variable+'.png')
    else:
      plt.show()



def general_model_report(modelstring, X, Y, write_folder=None, cv=3, balanced=True, labels=[], augmented=None):

  modelname = modelstring.split()[0]

  string_output = '=========== ' + modelname + ' Report ===================================\n'
  s = string_output + 'X shape: {}\n'.format(X.shape)
  s = s + 'Y shape: {}\n'.format(Y.shape)
  if write_folder:
    s = s + 'write_folder: ' + write_folder + '\n'
  else:
    s = s + 'No write_folder set. Report output in terminal.\n'
  s = s + 'cv: '+str(cv) + '\n'
  if balanced:
    s = s + 'balanced: ' + 'True\n'
  else:
    s = s + 'balanced: ' + 'False\n'
  if augmented != None:
    s = s + 'augmented: ' + augmented + '\n'

  s = s + '**********************\n'

  if balanced:
    class_weight = 'balanced'
  else:
    class_weight = None

  clf = get_model_ml_(modelstring)
  s = s + repr(clf) + '\n'

  # laco dos folds
  cv_folds = StratifiedKFold(n_splits=cv, random_state=int(time.time()), shuffle=True)
  scores_f1 = list()
  scores_precision = list()
  scores_recall = list()
  Y_pred_proba_geral = np.zeros(shape=Y.shape)
  Y_pred_geral = np.zeros(shape=Y.shape)

  for train, test in cv_folds.split(X, Y):

    # essa linha h soh pra setar caso o augmented seja None
    if augmented == None or augmented == 'custom':
      X_train_aug, Y_train_aug = X[train].copy(), Y[train].copy()

    # agora eh necessario checar o aumento de dados
    if augmented == 'smote':
      s = s + 'aug check: smote\n'
      X_train_aug, Y_train_aug = SMOTE().fit_resample(X[train], Y[train])
      s = s + 'X[train] shape: {}, Y[train] shape: {}\n'.format(X[train].shape, Y[train].shape)
      s = s + 'X_train_aug shape: {}, Y_train_aug shape: {}\n'.format(X_train_aug.shape, Y_train_aug.shape)
      s = s + 'Y_train {}\n'.format(sorted(Counter(Y[train]).items()))
      s = s + 'Y_train_aug {}\n'.format(sorted(Counter(Y_train_aug).items()))

    if augmented == 'undersampling':
      s = s + 'aug check: undersampling\n'
      rus = RandomUnderSampler()
      X_train_aug, Y_train_aug = rus.fit_resample(X[train], Y[train])
      s = s + 'X[train] shape: {}, Y[train] shape: {}\n'.format(X[train].shape, Y[train].shape)
      s = s + 'X_train_aug shape: {}, Y_train_aug shape: {}\n'.format(X_train_aug.shape, Y_train_aug.shape)
      s = s + 'Y_train {}\n'.format(sorted(Counter(Y[train]).items()))
      s = s + 'Y_train_aug {}\n'.format(sorted(Counter(Y_train_aug).items()))

    if augmented == 'oversampling':
      s = s + 'aug check: oversampling\n'
      ros = RandomOverSampler()
      X_train_aug, Y_train_aug = ros.fit_resample(X[train], Y[train])
      s = s + 'X[train] shape: {}, Y[train] shape: {}\n'.format(X[train].shape, Y[train].shape)
      s = s + 'X_train_aug shape: {}, Y_train_aug shape: {}\n'.format(X_train_aug.shape, Y_train_aug.shape)
      s = s + 'Y_train {}\n'.format(sorted(Counter(Y[train]).items()))
      s = s + 'Y_train_aug {}\n'.format(sorted(Counter(Y_train_aug).items()))

    # treino e predicoes
    clf.fit(X_train_aug, Y_train_aug)
    Y_pred = clf.predict(X[test])
    Y_true = Y[test].copy()

    # guarda na matrizona geral
    Y_pred_proba_geral[test] = clf.predict_proba(X[test])[:, 1].copy() # pegar o proba 1 aqui, analise binaria aqui !!! 
    Y_pred_geral[test] = clf.predict(X[test]).copy()

    # guardando os scores
    scores_f1.append(f1_score(Y_true, Y_pred))
    scores_precision.append(precision_score(Y_true, Y_pred))
    scores_recall.append(recall_score(Y_true, Y_pred))

    # guardando as confmatrix de cada fold
    confm = confusion_matrix(Y_true, Y_pred) 
    s = s + str(confm) + '\n'

  scores_f1 = np.array(scores_f1)
  scores_precision = np.array(scores_precision)
  scores_recall = np.array(scores_recall)
  s = s + '***************  Cross-validation scores  ***********\n'
  s = s + 'F1-score: {}, mean: {}\n'.format(scores_f1, scores_f1.mean())
  s = s + 'Precision: {}, mean: {}\n'.format(scores_precision, scores_precision.mean())
  s = s + 'Recall: {}, mean: {}\n'.format(scores_recall, scores_recall.mean())

  # conf matrix
  #Y_pred = cross_val_predict(clf, X, Y, cv=cv)
  Y_true = Y.copy()
  confm = confusion_matrix(Y_true, Y_pred_geral)
  s = s + '***************  Confusion Matrix  *****\n'
  s = s + str(confm) + '\n'

  #y_true = set(Y)
  #y_pred = set(Y)
  plt.clf()
  data = confm
  df_cm = pd.DataFrame(data, columns=np.unique(Y), index = np.unique(Y))
  df_cm.index.name = 'Actual'
  df_cm.columns.name = 'Predicted'
  plt.figure(figsize = (10,7))
  sn.set(font_scale=1.4) #for label size
  sn.heatmap(df_cm, cmap="Blues", fmt='', annot=True, annot_kws={"size": 16,})# font size
  if write_folder:
    plt.savefig(write_folder+'/confusion_matrix_' + modelname + '_.png')
  else:
    plt.show()

  # fit to tell the features importances and roc curve
  #clf.fit(X, Y)
  #proba = cross_val_predict(clf, X, Y, cv=cv, method='predict_proba')

  fpr_all, tpr_all, thresholds = roc_curve(Y_true, Y_pred_proba_geral)
  #auc_ = auc(fpr_all, tpr_all)
  
  auc_ = roc_auc_score(Y_true, Y_pred_proba_geral)

  # AUC
  s = s + 'AUC: {:.3}\n'.format(auc_)

  # plot dos thresholds
  '''
  plt.clf()
  plt.xlabel('Thresholds')
  plt.ylabel('Fpr / Tpr')
  plt.xlim([0,1])
  plt.plot(thresholds[::-1], fpr, 'green', label='False Positive rate')
  plt.plot(thresholds[::-1], tpr, 'red', label='True Positive rate')
  plt.legend(loc="upper right")
  if write_folder:
    plt.savefig(write_folder+'/plot_thresholds_tpr_fpr_RFC_.png')
  else:
    plt.show()
  '''


  # roc curve
  cv_folds = StratifiedKFold(n_splits=cv, random_state=int(time.time()), shuffle=True)

  plt.figure(figsize=(10,7))
  plt.clf()
  plt.xlabel('False positive')
  plt.ylabel('True positive')
  plt.title('ROC curve - AUC:{:.3}'.format(auc_))

  # vou fazer esse laco de novo mas o certo eh joger ele no de cima ainda
  for train, test in cv_folds.split(X, Y):

    # essa linha soh pra setar caso o augmented seja None
    X_train_aug, Y_train_aug = X[train].copy(), Y[train].copy()

    # agora eh necessario checar o aumento de dados
    if augmented == 'smote':
      X_train_aug, Y_train_aug = SMOTE().fit_resample(X[train], Y[train])

    if augmented == 'undersampling':
      rus = RandomUnderSampler()
      X_train_aug, Y_train_aug = rus.fit_resample(X[train], Y[train])

    if augmented == 'oversampling':
      ros = RandomOverSampler()
      X_train_aug, Y_train_aug = ros.fit_resample(X[train], Y[train])

    probas_ = clf.fit(X_train_aug, Y_train_aug).predict_proba(X[test])[:,1]
    
    fpr, tpr, thresholds = roc_curve(Y[test], probas_)
    #auc_fold_ = auc(fpr, tpr)
    
    auc_fold_ = roc_auc_score(Y[test], probas_)

    plt.plot(fpr, tpr, 'gray', label='Auc:{:.3}'.format(auc_fold_))



  # plot do modelo todo
  plt.plot(fpr_all, tpr_all, 'red', label='Auc:{:.3}'.format(auc_), linewidth=4)
  #plt.plot(fpr, thresholds, 'brown', label='Thresholds')
  #plt.plot(fpr, thresholds[::-1], 'yellow', label='Thresholds')




  plt.grid()
  plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='b', label='Chance', alpha=.8)
  plt.xlim([-0.05, 1.05])
  plt.ylim([-0.05, 1.05])
  plt.legend(loc='lower right')
  #plt.tight_layout()
  if write_folder:
    plt.savefig(write_folder+'/roc_curve_'+modelstring.replace(' ', '_')+'.png')
  else:
    plt.show()

  s = s + '***************  Feature importances  *******\n'

  X_aug, Y_aug = X, Y

  # feat importances
  # agora eh necessario checar o aumento de dados
  if augmented == 'smote':
    X_aug, Y_aug = SMOTE().fit_resample(X, Y)

  if augmented == 'undersampling':
    rus = RandomUnderSampler()
    X_aug, Y_aug = rus.fit_resample(X, Y)

  if augmented == 'oversampling':
    ros = RandomOverSampler()
    X_aug, Y_aug = ros.fit_resample(X, Y)


  if modelname == 'logit' or modelname == 'rfc':
    # treinar com tudo
    clf.fit(X_aug, Y_aug)

    if modelname == 'logit':
      feat_magn = clf.coef_[0] # BINARIO AQUI 
    elif modelname == 'rfc':
      feat_magn = clf.feature_importances_

    if len(labels) > 0:
      lst_feat_imp = list()
      for feat, imp in zip(feat_magn, labels):
        print (feat, imp)
        lst_feat_imp.append([feat, imp])
      a = sorted(lst_feat_imp, key=lambda x:x[0])
      s = s + str(a) + '\n'
    else:
      s = s + str(feat_magn) + '\n'

    s = s + '============================================================'

    tchart(np.array(a).T[1], np.array(a).T[0].astype('float'), title='Feature importance', pathfile=write_folder+'/tchart_'+modelstring.replace(' ','_')+'.png')

  if write_folder:
    file_ = open(write_folder+'/report_'+modelstring.replace(' ','_')+'.txt', 'w')
    file_.write(s)
    file_.close()
  else:
    print (s)


