import multiprocessing
from itertools import product

from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_predict, StratifiedKFold
from sklearn.metrics import f1_score, precision_score, recall_score, roc_curve, auc, roc_auc_score

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

from IPython.terminal.embed import InteractiveShellEmbed
from IPython.config.loader import Config

from sklearn.feature_selection import RFECV, RFE

from collections import namedtuple

###
def eliminate_cols_nan(df, thr):
  # thr entre 0 e 1
  n_records = df.shape[0]
  
  if thr <= 1 and thr >= 0:
    n_nans_max = int(n_records * thr)
  else:
    n_nans_max = int(thr)
  
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
def eliminate_records_nan(df, thr):
  # thr entre 0 e 1
  n_feat = df.shape[1]

  if thr <= 1 and thr >= 0:
    n_feat_max = int(n_feat * thr)
  else:
    n_feat_max = int(thr)

  records_to_delete = (np.isnan(df).sum(axis=1)>n_feat_max).to_numpy().nonzero()[0]

  df = df.drop(records_to_delete)

  return df, records_to_delete



def cros_val(clf, X, Y, metrics=['accuracy', 'recall'], smote=True, cv=3, multiclass=False):

  n_classes = len(set(Y))


  # falta instanciar
  return_named_tuple = namedtuple('return_named_tuple', ('clf', 'smote', 'cv', 'accuracy', 'recall', 'auc', 'f1_score'))

  # laco dos folds
  cv_folds = StratifiedKFold(n_splits=cv)

  scores_f1 = list()
  scores_precision = list()
  scores_recall = list()
  scores_auc = list()

  Y_pred_proba_geral = np.zeros(shape=Y.shape)
  Y_pred_geral = np.zeros(shape=Y.shape)

  for train, test in cv_folds.split(X, Y):

    # essa linha h soh pra setar caso o augmented seja None
    if smote == False:
      X_train_aug, Y_train_aug = X[train], Y[train]

    # agora eh necessario checar o aumento de dados
    if smote:
      X_train_aug, Y_train_aug = SMOTE().fit_resample(X[train], Y[train])

    '''
    if augmented == 'undersampling':
      s = s + 'aug check: undersampling\n'
      rus = RandomUnderSampler()
      X_train_aug, Y_train_aug = rus.fit_resample(X[train], Y[train])

    if augmented == 'oversampling':
      s = s + 'aug check: oversampling\n'
      ros = RandomOverSampler()
      X_train_aug, Y_train_aug = ros.fit_resample(X[train], Y[train])
    '''
    
    # treino e predicoes
    clf.fit(X_train_aug, Y_train_aug)
    Y_pred = clf.predict(X[test])
    Y_true = Y[test]

    '''
    ipshell = InteractiveShellEmbed(config=Config(),
                         banner1 = 'Dropping into IPython',
                         exit_msg = 'Leaving Interpreter, back to program.')

    ipshell('***Called from top level. '
          'Hit Ctrl-D to exit interpreter and continue program.\n'
          'Note that if you use %kill_embedded, you can fully deactivate\n'
          'This embedded instance so it will never turn on again')
    '''

    # guarda na matrizona geral
    if multiclass:
      roc_temporario_ = 0
      for i in range(1, n_classes+1):
        Y_pred_proba_geral[test] = clf.predict_proba(X[test])[:, i-1] # pegar o proba 1 aqui 
        roc_temporario_ = roc_temporario_ + roc_auc_score((Y_true==i).astype('int'), Y_pred_proba_geral[test])
      roc_temporario_ = roc_temporario_ / n_classes
      scores_auc.append(roc_temporario_)
    else:
      Y_pred_proba_geral[test] = clf.predict_proba(X[test])[:, 1] # pegar o proba 1 aqui 
      scores_auc.append(roc_auc_score(Y_true, Y_pred_proba_geral[test]))


    #Y_pred_proba_geral[test] = clf.predict_proba(X[test])[:, 1].copy() # pegar o proba 1 aqui 
    Y_pred_geral[test] = clf.predict(X[test]).copy()


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

  r = return_named_tuple (clf, smote, cv, scores_precision, scores_recall, scores_auc, scores_f1)
  
  return r

def grid_search_nested(X, Y, cv=3, writefolder=None):

  if len(set(Y)) > 2:
    multiclass = True
  elif len(set(Y)) == 2:
    multiclass = False
  else:
    print ('Erro! flag multiclass.')

  # laco dos folds
  cv_folds = StratifiedKFold(n_splits=cv)

  Y_pred_proba_geral = np.zeros(shape=Y.shape)
  Y_pred_geral = np.zeros(shape=Y.shape)



  # SVC
  C_list = np.logspace(np.log10(1), np.log10(1000), num=6)
  C_list = [str(x) for x in C_list]
  gamma_list = np.logspace(np.log10(0.0001), np.log10(1), num=6)
  gamma_list = [str(x) for x in gamma_list]

  svc_kernel = 'rbf'
  svc_params_list = list(itertools.product(C_list, gamma_list))
  svc_params_list = ['svc '+' '.join(x) for x in svc_params_list]
  
  # RF
  max_depth_list = ['2', '4', '8', 'None']
  rfc_params_list = ['rfc '+' '+x for x in max_depth_list]

  # Logit
  C_list = np.logspace(np.log10(0.001), np.log10(4), num=10)
  C_list = [str(x) for x in C_list]
  #clf = LogisticRegression(penalty='l2', solver='lbfgs', multi_class='multinomial')
  logit_params_list = ['logit '+' '+x for x in C_list]

  params_list = svc_params_list + rfc_params_list + logit_params_list

  
  params_scores = np.zeros((len(params_list),))
  params_std_scores = np.zeros((len(params_list),))

  s = ''

  if (writefolder != None):
    plt.figure(figsize=(14,8))
    plt.ylabel('AUC score')
    plt.xlabel('Parameter set number')
    plt.title('')


  for i, (train, test) in enumerate(cv_folds.split(X, Y)):
  
    for k, params in enumerate(params_list):

      print ('parametro {} de {}'.format(k, len(params_list)))


      if params.split()[0] == 'svc':
        clf = svm.SVC(C=float(params.split()[1]), gamma=float(params.split()[2]), kernel=svc_kernel, probability=True)
      elif params.split()[0] == 'rfc':
        if params.split()[1].lower()  == 'none':
          clf = RandomForestClassifier(n_estimators=100, max_depth=None)
        else:        
          clf = RandomForestClassifier(n_estimators=100, max_depth=int(params.split()[1]))
      elif params.split()[0] == 'logit':
          clf = LogisticRegression(penalty='l2', solver='lbfgs', multi_class='multinomial')
      else:
        print ('Nao foi identificado o classificador. {}'.format(params))
      
      return_ = cros_val(clf, X[train], Y[train], metrics=['accuracy', 'recall'], smote=True, cv=3, multiclass=multiclass)
      params_scores[k] = return_.auc.mean()
      params_std_scores[k] = return_.auc.std()
       
   
    best_params = params_list[params_scores.argmax()]
    best_params_idx = params_scores.argmax()
    

    if writefolder:
      s = s + '####### FOLD {} of {} #####\n'.format(i+1, cv)
      for param, score, std in zip(params_list, params_scores, params_std_scores):
        s = s + 'param: {}, score: {:.2} ({:.2})\n'.format(param, score, std)
      s = s + '* Best params: {}, idx: {} - score: {:.2}\n'.format(best_params, best_params_idx, params_scores[best_params_idx])
      
      s = s + '*** Evaluation phase ***\n'
      
      clf.fit(X[train], Y[train])
      
      auc_ = roc_auc_score(Y[test], clf.predict_proba(X[test])[:, 1])
      s = s + 'AUC Ev. score: {:.2}\n'.format(auc_)
      s = s + '###########################\n'

    else:
      print ('####### FOLD {} of {} #####'.format(i+1, cv))
      for param, score, std in zip(params_list, params_scores, params_std_scores):
        print ('param: {}, score: {:.2} ({:.2})'.format(param, score, std))  
      print ('* Best params: {}, idx: {} - score: {:.2}'.format(best_params, best_params_idx, params_scores[best_params_idx]))
      
      print ('*** Evaluation phase ***')
      clf.fit(X[train], Y[train])
      
      auc_ = roc_auc_score(Y[test], clf.predict_proba(X[test])[:, 1])
      print ('AUC Ev. score: {:.2}'.format(auc_))
      print ('###########################')


    if (writefolder != None):
      plt.plot(params_scores, 's-', label='fold {}'.format(i+1))
      plt.plot([0,len(params_scores)], [auc_, auc_], label='auc fold {}: {:.2}'.format(i+1, auc_))

  
  file_ = open(writefolder+'/'+'report_nested_cross_validation_hyperparameter_tuning.txt', 'w')
  file_.write(s)
  file_.close()

  if (writefolder != None):
    plt.legend(loc="lower right")
    plt.savefig(writefolder+'/'+'nested_cross_validation_scores.png', dpi=120)


################# PARALELO

def get_model_ml_(params):

  if params.split()[0] == 'svc':
    clf = svm.SVC(C=float(params.split()[1]), gamma=float(params.split()[2]), kernel='rbf', probability=True)

  elif params.split()[0] == 'rfc':
    if params.split()[1].lower()  == 'none':
      clf = RandomForestClassifier(n_estimators=100, max_depth=None)
    else:        
      clf = RandomForestClassifier(n_estimators=100, max_depth=int(params.split()[1]))

  elif params.split()[0] == 'logit':
      clf = LogisticRegression(penalty='l2', solver='lbfgs', multi_class='multinomial')

  elif params.split()[0] == 'ada':
      clf = AdaBoostClassifier(n_estimators=int(params.split()[1]), learning_rate=float(params.split()[2]))

  else:
    print ('Nao foi identificado o classificador. {}'.format(params))
    
  return clf

#
def f(params, X, Y):

  print ('rodando params: {}'.format(params))

  clf = get_model_ml_(params)
  
  if len(set(Y)) > 2:
    multiclass = True
  elif len(set(Y)) == 2:
    multiclass = False
  else:
    print ('Erro! flag multiclass.')

  return_ = cros_val(clf, X, Y, metrics=['accuracy', 'recall'], smote=True, cv=3, multiclass=multiclass)

  return return_.auc.mean()

  print ('params {} finalizado.'.format(params))

  #params_scores[k] = return_.auc.mean()
  #params_std_scores[k] = return_.auc.std()



def grid_search_nested_parallel(X, Y, cv=3, writefolder=None, n_jobs=30):

  n_classes = len(set(Y))

  if len(set(Y)) > 2:
    multiclass = True
  elif len(set(Y)) == 2:
    multiclass = False
  else:
    print ('Erro! flag multiclass.')

  # laco dos folds
  cv_folds = StratifiedKFold(n_splits=cv)

  Y_pred_proba_geral = np.zeros(shape=Y.shape)
  Y_pred_geral = np.zeros(shape=Y.shape)


  # SVC
  C_list = np.logspace(np.log10(1), np.log10(2000), num=16)
  C_list = [str(x) for x in C_list]
  gamma_list = np.logspace(np.log10(0.0001), np.log10(2.2), num=12)
  gamma_list = [str(x) for x in gamma_list]

  svc_kernel = 'rbf'
  svc_params_list = list(itertools.product(C_list, gamma_list))
  svc_params_list = ['svc '+' '.join(x) for x in svc_params_list]

  # Adaboost
  ada_n_estimators = ['5', '20', '60']
  ada_learning_rate = ['1.0', '0.1', '2.0']
  ada_params_list = list(itertools.product(ada_n_estimators, ada_learning_rate))
  ada_params_list = ['ada '+' '.join(x) for x in ada_params_list]

  
  # RF
  max_depth_list = ['2', '4', '8', '16', 'None']
  rfc_params_list = ['rfc '+' '+x for x in max_depth_list]

  # Logit
  C_list = np.logspace(np.log10(0.001), np.log10(40), num=12)
  C_list = [str(x) for x in C_list]
  logit_params_list = ['logit '+' '+x for x in C_list]

  params_list = svc_params_list + rfc_params_list + logit_params_list + ada_params_list
  
  params_scores = np.zeros((len(params_list),))
  params_std_scores = np.zeros((len(params_list),))

  s = ''

  if (writefolder != None):
    plt.figure(figsize=(14,8))
    plt.ylabel('AUC score')
    plt.xlabel('Parameter set number')
    plt.title('')


  for i, (train, test) in enumerate(cv_folds.split(X, Y)):


    parameters_vector_total = [(x, X[train], Y[train]) for x in params_list]

    params_scores_partial = list()
    for parameters_vector in [parameters_vector_total[j:j+n_jobs] for j in range(0, len(parameters_vector_total), n_jobs)]:
      with multiprocessing.Pool(processes=n_jobs) as pool:
        params_scores_partial = params_scores_partial + pool.starmap(f, parameters_vector)

    params_scores = np.array(params_scores_partial)

    '''
    with multiprocessing.Pool(processes=len(params_list)) as pool:
      parameters_vector = [(x, X[train], Y[train]) for x in params_list]
      params_scores = np.array(pool.starmap(f, parameters_vector))
    '''
   
    best_params = params_list[params_scores.argmax()]
    best_params_idx = params_scores.argmax()

    clf = get_model_ml_(best_params)
    clf.fit(X[train], Y[train])
    Y_true = Y[test]
    

    if writefolder:
      s = s + '####### FOLD {} of {} #####\n'.format(i+1, cv)
      for param, score, std in zip(params_list, params_scores, params_std_scores):
        s = s + 'param: {}, score: {:.2} ({:.2})\n'.format(param, score, std)
      s = s + '* Best params: {}, idx: {} - score: {:.2}\n'.format(best_params, best_params_idx, params_scores[best_params_idx])
      
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


      s = s + 'AUC Ev. score: {:.2}\n'.format(auc_)
      s = s + '###########################\n'

    else:
      print ('####### FOLD {} of {} #####'.format(i+1, cv))
      for param, score, std in zip(params_list, params_scores, params_std_scores):
        print ('param: {}, score: {:.2} ({:.2})'.format(param, score, std))  
      print ('* Best params: {}, idx: {} - score: {:.2}'.format(best_params, best_params_idx, params_scores[best_params_idx]))
      
      print ('*** Evaluation phase ***')

      if multiclass:
        roc_temporario_ = 0
        for j in range(1, n_classes+1):
          Y_pred_proba_geral[test] = clf.predict_proba(X[test])[:, j-1] # pegar o proba 1 aqui 
          roc_temporario_ = roc_temporario_ + roc_auc_score((Y_true==j).astype('int'), Y_pred_proba_geral[test])
        roc_temporario_ = roc_temporario_ / n_classes
        auc_ = roc_temporario_
      else:
        auc_ = roc_auc_score(Y[test], clf.predict_proba(X[test])[:, 1])


      print ('AUC Ev. score: {:.2}'.format(auc_))
      print ('###########################')





    if (writefolder != None):
      plt.plot(params_scores, 's-', label='fold {}'.format(i+1))
      plt.plot([0,len(params_scores)], [auc_, auc_], label='auc fold {}: {:.2}'.format(i+1, auc_))

  
  file_ = open(writefolder+'/'+'report_nested_cross_validation_hyperparameter_tuning.txt', 'w')
  file_.write(s)
  file_.close()

  if (writefolder != None):
    plt.legend(loc="lower right")
    plt.savefig(writefolder+'/'+'nested_cross_validation_scores.png', dpi=120)
  



def part_feature_study(feature, df_a, y_name, pathfile=None):

  type_var = None

  # primeiro decidir se a feature eh categorica ou continua
  feature_subgroup = set(df_a[feature])
  if len(feature_subgroup) < 8: # criterio meio abstrato aqui
    type_var = 'Cat'

    feature_subgroup = set(df_a[feature])

  else:
    type_var = 'Cont'
    nbins = 4
    x = np.linspace(0,100,nbins+1)
    percentis = np.percentile(df_a[feature], x) # nbins + 1 comeca no zero

    hist, edges = np.histogram(df_a[feature].values, bins=percentis)
    
    feature_subgroup = list()
    
    for i in range(nbins):
      feature_subgroup.append( np.logical_and((df_a[feature] > edges[i]) , (df_a[feature] < edges[i+1])) )
    

  for i, partial_feat in enumerate(feature_subgroup):

    if type_var == 'Cat':
      df_b = df_a[df_a[feature]==partial_feat]

    elif type_var == 'Cont':
      df_b = df_a[partial_feat]

    df_b = df_b.drop(columns=feature)

    # correlacao com o y
    plt.clf()
    plt.figure(figsize=(15,8))
    corr_ = df_b.corr()[y_name].values
    x = range(len(corr_))
    plt.plot(x, corr_, 'o--')
    plt.plot([0, x[-1]], [0, 0], '-')
    plt.xticks(x, list(df_b.corr()[y_name].index), rotation=75)
    plt.tight_layout()
    plt.title('quant. dados: {}'.format(len(df_b)))
    if pathfile:
      if type_var == 'Cat':
        plt.savefig(pathfile+'/'+feature+'_'+str(partial_feat)+'_'+'corr_with'+y_name+'.png', dpi=120)
      elif type_var == 'Cont':      
        plt.savefig(pathfile+'/'+feature+'_'+str(edges[i])+'_'+str(edges[i+1])+'_'+'corr_with'+y_name+'.png', dpi=120)
    else:
      plt.show()

    plt.clf()


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
  estimator = RandomForestClassifier(n_estimators=600, max_depth=2)
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
  estimator = RandomForestClassifier(n_estimators=600, max_depth=2, class_weight='balanced')
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
  estimator = RandomForestClassifier(n_estimators=600, max_depth=2, class_weight='balanced')
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




def rfe_cv(X, Y, cv=3, pathfile=None, labels=[]):

  s = '######## Report RFE CV ######\n'

  # Plot number of features VS. cross-validation scores
  plt.figure(figsize=(14,8))
  plt.xlabel("Number of features selected")
  plt.ylabel("Cross validation score")

  # Create the RFE object and compute a cross-validated score.
  svc = svm.SVC(kernel="linear", class_weight='balanced')
  rfecv = RFECV(estimator=svc, step=1, cv=cv, scoring='accuracy')
  rfecv.fit(X, Y)
  plt.plot(range(1, len(rfecv.grid_scores_) + 1), rfecv.grid_scores_, label='SVC score accuracy')

  '''
  ipshell = InteractiveShellEmbed(config=Config(),
                       banner1 = 'Dropping into IPython',
                       exit_msg = 'Leaving Interpreter, back to program.')

  ipshell('***Called from top level. '
        'Hit Ctrl-D to exit interpreter and continue program.\n'
        'Note that if you use %kill_embedded, you can fully deactivate\n'
        'This embedded instance so it will never turn on again')
  '''
  
  s = s + '#############################\n'
  s = s + 'SVC score accuracy\n'
  s = s + 'features selecionadas: {}\n'.format(' '.join(labels[rfecv.support_]))
  s = s + 'Ranking:\n'
  for line in  np.array(sorted(list(zip(rfecv.ranking_,labels)))):
    s = s + str(line) + '\n'


  svc = svm.SVC(kernel="linear", class_weight='balanced')
  rfecv = RFECV(estimator=svc, step=1, cv=cv, scoring='recall')
  rfecv.fit(X, Y)
  plt.plot(range(1, len(rfecv.grid_scores_) + 1), rfecv.grid_scores_, label='SVC score recall')
  s = s + '#############################\n'
  s = s + 'SVC score recall\n'
  s = s + 'features selecionadas: {}\n'.format(' '.join(labels[rfecv.support_]))
  s = s + 'Ranking:\n'
  for line in  np.array(sorted(list(zip(rfecv.ranking_,labels)))):
    s = s + str(line) + '\n'



  # Create the RFE object and compute a cross-validated score.
  estimator = LogisticRegression(solver='lbfgs', multi_class='multinomial', class_weight='balanced')
  rfecv = RFECV(estimator=estimator, step=1, cv=cv, scoring='accuracy')
  rfecv.fit(X, Y)
  plt.plot(range(1, len(rfecv.grid_scores_) + 1), rfecv.grid_scores_, label='Logit score accuracy')
  s = s + '#############################\n'
  s = s + 'Logit score accuracy\n'
  s = s + 'features selecionadas: {}\n'.format(' '.join(labels[rfecv.support_]))
  s = s + 'Ranking:\n'
  for line in  np.array(sorted(list(zip(rfecv.ranking_,labels)))):
    s = s + str(line) + '\n'


  estimator = LogisticRegression(solver='lbfgs', multi_class='multinomial', class_weight='balanced')
  rfecv = RFECV(estimator=estimator, step=1, cv=cv, scoring='recall')
  rfecv.fit(X, Y)
  plt.plot(range(1, len(rfecv.grid_scores_) + 1), rfecv.grid_scores_, label='Logit score recall')
  s = s + '#############################\n'
  s = s + 'Logit score recall\n'
  s = s + 'features selecionadas: {}\n'.format(' '.join(labels[rfecv.support_]))
  s = s + 'Ranking:\n'
  for line in  np.array(sorted(list(zip(rfecv.ranking_,labels)))):
    s = s + str(line) + '\n'

 
  plt.legend(loc="upper right")

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
  for variable in df.columns:
    print ('Histograma '+variable)
    fig = plt.figure(figsize=(5,4))
    df[variable].plot.hist(stacked=False, bins=20)
    plt.xlabel(variable)
    plt.ylabel('ocurrences')
    if fig_folder != None:
      plt.savefig('./'+fig_folder+'/histograma_var_'+variable+'.png')
    else:
      plt.show()
    plt.cla()
    plt.close(fig)
    
def rfc_model_report(X, Y, write_folder=None, cv=4, balanced=True, labels=[], augmented=None):

  string_output = '=========== RFC Report ===================================\n'
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

  clf = RandomForestClassifier(n_estimators=100, class_weight=class_weight)
  s = s + repr(clf) + '\n'

  # laco dos folds
  cv_folds = StratifiedKFold(n_splits=cv)
  scores_f1 = list()
  scores_precision = list()
  scores_recall = list()
  Y_pred_proba_geral = np.zeros(shape=Y.shape)
  Y_pred_geral = np.zeros(shape=Y.shape)

  for train, test in cv_folds.split(X, Y):

    # essa linha h soh pra setar caso o augmented seja None
    if augmented == None:
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

    if augmented == 'oversampling':
      s = s + 'aug check: oversampling\n'
      ros = RandomOverSampler()
      X_train_aug, Y_train_aug = ros.fit_resample(X[train], Y[train])

    # treino e predicoes
    clf.fit(X_train_aug, Y_train_aug)
    Y_pred = clf.predict(X[test])
    Y_true = Y[test].copy()

    # guarda na matrizona geral
    Y_pred_proba_geral[test] = clf.predict_proba(X[test])[:, 1].copy() # pegar o proba 1 aqui 
    Y_pred_geral[test] = clf.predict(X[test]).copy()

    # guardando os scores
    #scores_f1.append(f1_score(Y_true, Y_pred))
    #scores_precision.append(precision_score(Y_true, Y_pred))
    #scores_recall.append(recall_score(Y_true, Y_pred))

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

  y_true = set(Y)
  y_pred = set(Y)
  data = confm
  df_cm = pd.DataFrame(data, columns=np.unique(Y), index = np.unique(Y))
  df_cm.index.name = 'Actual'
  df_cm.columns.name = 'Predicted'
  plt.figure(figsize = (10,7))
  sn.set(font_scale=1.4) #for label size
  sn.heatmap(df_cm, cmap="Blues", fmt='', annot=True, annot_kws={"size": 16,})# font size
  if write_folder:
    plt.savefig(write_folder+'/confusion_matrix_RFC_.png')
  else:
    plt.show()

  # fit to tell the features importances and roc curve
  #clf.fit(X, Y)
  #proba = cross_val_predict(clf, X, Y, cv=cv, method='predict_proba')

  fpr_all, tpr_all, thresholds = roc_curve(Y_true, Y_pred_proba_geral)
  #auc_ = auc(fpr_all, tpr_all)
  
  auc_ = roc_auc_score(Y_true, Y_pred_proba_geral)

  # AUC
  s = s + 'AUC: {:.2}\n'.format(auc_)

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
  cv_folds = StratifiedKFold(n_splits=cv)

  plt.clf()
  plt.figure(figsize=(10,7))
  plt.xlabel('False positive')
  plt.ylabel('True positive')
  plt.title('ROC CURVE - AUC:{:.3}'.format(auc_))

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
    '''
    ipshell = InteractiveShellEmbed(config=Config(),
                       banner1 = 'Dropping into IPython',
                       exit_msg = 'Leaving Interpreter, back to program.')

    ipshell('***Called from top level. '
        'Hit Ctrl-D to exit interpreter and continue program.\n'
        'Note that if you use %kill_embedded, you can fully deactivate\n'
        'This embedded instance so it will never turn on again')
    '''
    
    auc_fold_ = roc_auc_score(Y[test], probas_)

    plt.plot(fpr, tpr, 'gray', label='Auc:{:.3}'.format(auc_fold_))



  # plot do modelo todo
  plt.plot(fpr_all, tpr_all, 'red', label='Auc:{:.3}'.format(auc_))
  plt.plot(fpr, thresholds, 'brown', label='Thresholds')
  #plt.plot(fpr, thresholds[::-1], 'yellow', label='Thresholds')




  plt.grid()
  plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='b', label='Chance', alpha=.8)
  plt.xlim([-0.05, 1.05])
  plt.ylim([-0.05, 1.05])
  plt.legend(loc="lower right")
  plt.tight_layout()
  if write_folder:
    plt.savefig(write_folder+'/roc_curve_RFC_.png')
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

  # treinar com tudo
  clf.fit(X_aug, Y_aug)

  if len(labels) > 0:
    lst_feat_imp = list()
    for feat, imp in zip(clf.feature_importances_, labels):
      print (feat, imp)
      lst_feat_imp.append([feat, imp])
    a = sorted(lst_feat_imp, key=lambda x:x[0])
    s = s + str(a) + '\n'
  else:
    s = s + str(clf.feature_importances_) + '\n'

  s = s + '============================================================'

  tchart(np.array(a).T[1], np.array(a).T[0].astype('float'), title='Feature importance', pathfile=write_folder+'/tchart_RFC.png')

  if write_folder:
    file_ = open(write_folder+'/report_RFC.txt', 'w')
    file_.write(s)
    file_.close()
  else:
    print (s)


  importlib.reload(plt); importlib.reload(mpl)









## LOGIT

def logit_model_report(X, Y, write_folder=None, cv=4, balanced=True, labels=[], augmented=None):


  clf = LogisticRegression(random_state=0, solver='lbfgs', multi_class='multinomial').fit(X, Y)

  string_output = '=========== Logit Report ===================================\n'
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

  clf = LogisticRegression(random_state=0, solver='lbfgs', multi_class='auto').fit(X, Y)
  s = s + repr(clf) + '\n'

  # laco dos folds
  cv_folds = StratifiedKFold(n_splits=cv)
  scores_f1 = list()
  scores_precision = list()
  scores_recall = list()
  Y_pred_proba_geral = np.zeros(shape=Y.shape)
  Y_pred_geral = np.zeros(shape=Y.shape)

  for train, test in cv_folds.split(X, Y):

    # essa linha h soh pra setar caso o augmented seja None
    if augmented == None:
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

    if augmented == 'oversampling':
      s = s + 'aug check: oversampling\n'
      ros = RandomOverSampler()
      X_train_aug, Y_train_aug = ros.fit_resample(X[train], Y[train])

    # treino e predicoes
    clf.fit(X_train_aug, Y_train_aug)
    Y_pred = clf.predict(X[test])
    Y_true = Y[test].copy()

    # guarda na matrizona geral, isso seria o cross_val_predict
    Y_pred_proba_geral[test] = clf.predict_proba(X[test])[:, 1].copy() # pegar o proba 1 aqui 
    Y_pred_geral[test] = clf.predict(X[test]).copy()

    # guardando os scores
    #scores_f1.append(f1_score(Y_true, Y_pred))
    #scores_precision.append(precision_score(Y_true, Y_pred))
    #scores_recall.append(recall_score(Y_true, Y_pred))

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

  y_true = set(Y)
  y_pred = set(Y)
  data = confm
  df_cm = pd.DataFrame(data, columns=np.unique(Y), index = np.unique(Y))
  df_cm.index.name = 'Actual'
  df_cm.columns.name = 'Predicted'
  plt.figure(figsize = (10,7))
  sn.set(font_scale=1.4) #for label size
  sn.heatmap(df_cm, cmap="Blues", fmt='', annot=True, annot_kws={"size": 16,})# font size
  if write_folder:
    plt.savefig(write_folder+'/confusion_matrix_Logit_.png')
  else:
    plt.show()

  # fit to tell the features importances and roc curve
  #clf.fit(X, Y)
  #proba = cross_val_predict(clf, X, Y, cv=cv, method='predict_proba')

  fpr_all, tpr_all, thresholds = roc_curve(Y_true, Y_pred_proba_geral)
  #auc_ = auc(fpr_all, tpr_all)

  auc_ = roc_auc_score(Y_true, Y_pred_proba_geral)

  # AUC
  s = s + 'AUC: {:.2}\n'.format(auc_)

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
  cv_folds = StratifiedKFold(n_splits=cv)

  plt.clf()
  plt.figure(figsize=(10,7))
  plt.xlabel('False positive')
  plt.ylabel('True positive')
  plt.title('ROC CURVE - AUC:{:.3}'.format(auc_))

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
  plt.plot(fpr_all, tpr_all, 'red', label='Auc:{:.3}'.format(auc_))
  plt.plot(fpr, thresholds, 'brown', label='Thresholds')
  #plt.plot(fpr, thresholds[::-1], 'yellow', label='Thresholds')




  plt.grid()
  plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='b', label='Chance', alpha=.8)
  plt.xlim([-0.05, 1.05])
  plt.ylim([-0.05, 1.05])
  plt.legend(loc="lower right")
  plt.tight_layout()
  if write_folder:
    plt.savefig(write_folder+'/roc_curve_Logit_.png')
  else:
    plt.show()

  s = s + '***************  Feature coefs  *******\n'

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

  # treinar com tudo
  clf.fit(X_aug, Y_aug)

  s = s + '*** Features ****************\n'  
  s = s + '\n'.join(labels) + '\n'

  #if len(labels) > 0:
  print ('labels size in logit report: {}'.format(len(labels)))
  if len(labels) > 0:
    lst_feat_imp = list()
    for feat, imp in zip(clf.coef_[0], labels):
      print ('{} {}\n'.format(feat, imp))
      lst_feat_imp.append([feat, imp])
    a = sorted(lst_feat_imp, key=lambda x:abs(x[0]))
    s = s + str(a) + '\n'
  else:
    s = s + str(clf.coef_) + '\n'
  
  s = s + '============================================================'

  tchart(np.array(a).T[1], np.array(a).T[0].astype('float'), title='Logit coeficients', pathfile=write_folder+'/tchart_Logit.png')

  '''
  ipshell = InteractiveShellEmbed(config=Config(),
                       banner1 = 'Dropping into IPython',
                       exit_msg = 'Leaving Interpreter, back to program.')

  ipshell('***Called from top level. '
        'Hit Ctrl-D to exit interpreter and continue program.\n'
        'Note that if you use %kill_embedded, you can fully deactivate\n'
        'This embedded instance so it will never turn on again')
  '''


  if write_folder:
    file_ = open(write_folder+'/report_Logit.txt', 'w')
    file_.write(s)
    file_.close()
  else:
    print (s)


  importlib.reload(plt); importlib.reload(mpl)



























##################################
def svc_model_report(X, Y, write_folder=None, cv=4, balanced=True, labels=[], probability=True, augmented=None):


  string_output = '=========== SVC Report ===================================\n'
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

  clf = svm.SVC(gamma='auto', probability=probability, class_weight=class_weight)
  s = s + repr(clf) + '\n'

  # laco dos folds
  cv_folds = StratifiedKFold(n_splits=cv)
  scores_f1 = list()
  scores_precision = list()
  scores_recall = list()
  Y_pred_proba_geral = np.zeros(shape=Y.shape)
  Y_pred_geral = np.zeros(shape=Y.shape)

  for train, test in cv_folds.split(X, Y):

    # essa linha h soh pra setar caso o augmented seja None
    if augmented == None:
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

    if augmented == 'oversampling':
      s = s + 'aug check: oversampling\n'
      ros = RandomOverSampler()
      X_train_aug, Y_train_aug = ros.fit_resample(X[train], Y[train])

    # treino e predicoes
    clf.fit(X_train_aug, Y_train_aug)
    Y_pred = clf.predict(X[test])
    Y_true = Y[test].copy()

    # guarda na matrizona geral
    Y_pred_proba_geral[test] = clf.predict_proba(X[test])[:, 1].copy() # pegar o proba 1 aqui 
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
  s = s + '***************  Confusion Matrix Geral *****\n'
  s = s + str(confm) + '\n'

  y_true = set(Y)
  y_pred = set(Y)
  data = confm
  df_cm = pd.DataFrame(data, columns=np.unique(Y), index = np.unique(Y))
  df_cm.index.name = 'Actual'
  df_cm.columns.name = 'Predicted'
  plt.figure(figsize = (10,7))
  sn.set(font_scale=1.4) #for label size
  sn.heatmap(df_cm, cmap="Blues", fmt='', annot=True, annot_kws={"size": 16,})# font size
  if write_folder:
    plt.savefig(write_folder+'/confusion_matrix_SVC_.png')
  else:
    plt.show()

  # fit to tell the features importances and roc curve
  #clf.fit(X, Y)
  #proba = cross_val_predict(clf, X, Y, cv=cv, method='predict_proba')

  fpr_all, tpr_all, thresholds = roc_curve(Y_true, Y_pred_proba_geral)
  #auc_ = auc(fpr_all, tpr_all)

  auc_ = roc_auc_score(Y_true, Y_pred_proba_geral)


  # AUC
  s = s + 'AUC: {:.2}\n'.format(auc_)

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
  cv_folds = StratifiedKFold(n_splits=cv)

  plt.clf()
  plt.figure(figsize=(10,7))
  plt.xlabel('False positive')
  plt.ylabel('True positive')
  plt.title('ROC CURVE - SVC - AUC:{:.3}'.format(auc_))

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

    probas_ = clf.fit(X_train_aug, Y_train_aug).predict_proba(X[test])[:, 1]
    fpr, tpr, thresholds = roc_curve(Y[test], probas_)
    #auc_fold_ = auc(fpr, tpr)

    auc_fold_ = roc_auc_score(Y[test], probas_)


    plt.plot(fpr, tpr, 'gray', label='Auc:{:.3}'.format(auc_fold_))



  # plot do modelo todo
  plt.plot(fpr_all, tpr_all, 'red', label='Auc:{:.3}'.format(auc_))
  plt.plot(fpr, thresholds, 'brown', label='Thresholds')
  #plt.plot(fpr, thresholds[::-1], 'yellow', label='Thresholds')

  plt.grid()
  plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='b', label='Chance', alpha=.8)
  plt.xlim([-0.05, 1.05])
  plt.ylim([-0.05, 1.05])
  plt.legend(loc="lower right")
  plt.tight_layout()
  if write_folder:
    plt.savefig(write_folder+'/roc_curve_SVC_.png')
  else:
    plt.show()

  if write_folder:
    file_ = open(write_folder+'/report_SVC.txt', 'w')
    file_.write(s)
    file_.close()
  else:
    print (s)

  importlib.reload(plt); importlib.reload(mpl)

