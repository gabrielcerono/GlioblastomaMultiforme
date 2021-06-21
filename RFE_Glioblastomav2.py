from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from random import randint
import pandas as pd
from sklearn.metrics import matthews_corrcoef
import xgboost as xgb
from xgboost.sklearn import XGBClassifier
from imblearn.ensemble import BalancedBaggingClassifier
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score
from sklearn import metrics
from sklearn.metrics import precision_recall_curve

class RankingRE():
  def __init__(self, X, y, loops, classifier):
   #X and Y in pandas dataframe.
    self.X = X
    self.y = y
    self.loops = loops
    self.classifier = classifier
    #It always gives the same seeds, now we can loop through these seeds
    
  def ranking_by_matthew_punishment(self):
    std = np.zeros(len(self.X.columns),)
    rankings = np.zeros(len(self.X.columns),)
    mttavg= []
    favg= []
    accuracyavg = []
    tpavg = []
    tnavg = []
    rocavg = []
    praucavg = []
    a= 0
    randseed = np.random.randint(9999, size = 1500)
    
    for x in range(self.loops):
        
    #This little trick right here, will let us have the same splitting here, and in the top feature dataset below. 
      a += 1
      seed = randseed[a]
    #Splits the dataset in a further set of rank feature. 1/3 for training 1/3 for feature ranking / 1/3 for validation
      X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size=0.33, random_state= seed)      
      X_train, X_rank, y_train, y_rank = train_test_split(X_train, y_train, test_size=0.5, random_state= seed)
    #Initializing the classifier
      classif = self.classifier
    #Fits the Random forest and we calculate a matthew score on ranking set. 
      classif.fit(X_train, y_train)
      mttoriginal = matthews_corrcoef(y_rank, classif.predict(X_rank))
    #That's for the ranking, now we gotta do for the validation set and also for all the metrics
      mttaverage = matthews_corrcoef(y_test, classif.predict(X_test))
      mttavg.append(mttaverage)
      #f1 score
      faverage = f1_score(y_test, classif.predict(X_test))
      favg.append(faverage)
      #accuracy score
      accuracyaverage = accuracy_score(y_test, classif.predict(X_test))
      accuracyavg.append(accuracyaverage)
      #TP and TN rate 
      tn, fp, fn, tp = confusion_matrix(y_test, classif.predict(X_test), labels =[0,1]).ravel()
      tprate =  tp / (tp + fn)
      tnrate = tn / (tn + fp)
      tpavg.append(tprate)
      tnavg.append(tnrate)
      # ROC AUC 
      roc = roc_auc_score(y_test, classif.predict_proba(X_test)[:, 1])
      rocavg.append(roc)
      
     #Precision recall area under de curve
      precision, recall, _thresholds = metrics.precision_recall_curve(y_test, classif.predict_proba(X_test)[:, 1])
      prauc = metrics.auc(recall, precision)     
      praucavg.append(prauc)
      
    #We initialize 2 lists to append values from the RFE loop.
      mttrf= []
      columnsrf= []
      
  

      for x in self.X.columns:
      #We use the same Seed that we used in the first splitting, so we don't get data leakeage.
        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size=0.33, random_state= seed)      
        X_train, X_rank, y_train, y_rank = train_test_split(X_train, y_train, test_size=0.5, random_state= seed)
    
        
    #We drop a different column each loop.
        X_train = X_train.drop([x], axis=1)
        X_rank = X_rank.drop([x], axis=1)
    #We fit our random forest again, but this time our training dataset lacks a feature.
        classif.fit(X_train, y_train)
        mtt = matthews_corrcoef(y_rank, classif.predict(X_rank))
    #We append to the list each column that we dropped.
        columnsrf.append(x)
    #And we also append, the drop (or gain), in matthew score that we got when the feature was missing.
        mttrf.append(mttoriginal - mtt)
        #End of loop
        
        
    #Lists need to be passed into Numpy arrays. The drop in MCC score is passsed as the variable outcome.
      outcome = np.array(mttrf)
     #Add the the gain or loss of matthew score, to the outcome list, and then index it to the ranking NP array.
      rankings = np.add(outcome, rankings)
      
     #Index Again the Mtt score difference, as outcome array, to the STD array.
      std = np.vstack((outcome, std))
      
    
    #Portion out of the looping. Dividimos el ranking por loops, since this its been adding up. 
    rankings = np.true_divide(rankings, self.loops)
    #The last portion is all 0s so we delete it
    std = np.delete(std, -1, axis = 0)
    #We take the standart deviation, leaving out us with one column of the STDs
    std = np.std(std, axis = 0)
    std = np.dstack((columnsrf, std))
    std = pd.DataFrame(data = np.squeeze(std, axis = 0), columns =['Categories', 'SD_of_matt_punishment'])
    featuresranks = np.dstack((columnsrf, rankings))
    borda = pd.DataFrame(data = np.squeeze(featuresranks, axis=0), columns=['Categories', 'average-mtt-punishment'])
    borda['ranking'] = borda['average-mtt-punishment'].rank(ascending = False)
    borda = borda.merge(std, on = 'Categories',)
    borda.sort_values(by='average-mtt-punishment', inplace = True, ascending = False)
    
    #Rank feature part is done, this part here is the validation training with all features
    
    mttavgmean = np.mean(np.array(mttavg))
    favgmean = np.mean(np.array(favg))
    accuracyavgmean = np.mean(np.array(accuracyavg))
    tpavgmean = np.mean(np.array(tpavg))
    tnavgmean = np.mean(np.array(tnavg))
    rocavgmean = np.mean(np.array(rocavg))
    praucavgmean = np.mean(np.array(praucavg))
    #Some error going on with the final table here
    metricss = np.array([mttavgmean, favgmean, accuracyavgmean, tpavgmean, tnavgmean, rocavgmean, praucavgmean])
    metricss = metricss.reshape(1,7)
    metric_final = pd.DataFrame(data = metricss, columns = ['matthewscore', 'f1-score', 'Accuracy', 'True Positives', 'True Negatives', 'ROC AUC', 'PR AUC'])
    
    ################################################################################################################
    ################################################################################################################
    
    #Testing on top 2 features
    mttavg= []
    favg= []
    accuracyavg = []
    tpavg = []
    tnavg = []
    rocavg = []
    praucavg = []
    
    #Getting the first 2 features
    b= 0
    top_features = borda.iloc[[0,1],[0]].to_numpy()
    X_top = self.X[top_features.reshape(2)]   
    #Now we have X, with the top features
    for x in range(self.loops):
    #This seeding will let us have the same splitting that in the ranking feature    
      b += 1
      seedtf = randseed[b]    
      X_train, X_test, y_train, y_test = train_test_split(X_top, self.y, test_size=0.33, random_state= seedtf)      
      X_train, X_rank, y_train, y_rank = train_test_split(X_train, y_train, test_size=0.5, random_state= seedtf)
      classif = self.classifier
   #Fits the Random forest and we calculate a matthew score on ranking set. 
      classif.fit(X_train, y_train)
      mttoriginal = matthews_corrcoef(y_rank, classif.predict(X_rank))
    #That's for the ranking, now we gotta do for the validation set and also for all the metrics
      mttaverage = matthews_corrcoef(y_test, classif.predict(X_test))
      mttavg.append(mttaverage)
      #f1 score
      faverage = f1_score(y_test, classif.predict(X_test))
      favg.append(faverage)
      #accuracy score
      accuracyaverage = accuracy_score(y_test, classif.predict(X_test))
      accuracyavg.append(accuracyaverage)
      #TP and TN rate 
      tn, fp, fn, tp = confusion_matrix(y_test, classif.predict(X_test), labels =[0,1]).ravel()
      tprate =  tp / (tp + fn)
      tnrate = tn / (tn + fp)
      tpavg.append(tprate)
      tnavg.append(tnrate)
      # ROC AUC 
      roc = roc_auc_score(y_test, classif.predict_proba(X_test)[:, 1])
      rocavg.append(roc)
      
     #Precision recall area under de curve
      precision, recall, _thresholds = metrics.precision_recall_curve(y_test, classif.predict_proba(X_test)[:, 1])
      prauc = metrics.auc(recall, precision)     
      praucavg.append(prauc)
      
    mttavgmean = np.mean(np.array(mttavg))
    favgmean = np.mean(np.array(favg))
    accuracyavgmean = np.mean(np.array(accuracyavg))
    tpavgmean = np.mean(np.array(tpavg))
    tnavgmean = np.mean(np.array(tnavg))
    rocavgmean = np.mean(np.array(rocavg))
    praucavgmean = np.mean(np.array(praucavg))
    #Some error going on with the final table here
    metricss = np.array([mttavgmean, favgmean, accuracyavgmean, tpavgmean, tnavgmean, rocavgmean, praucavgmean])
    metricss = metricss.reshape(1,7)
    metric_final_tf = pd.DataFrame(data = metricss, columns = ['matthewscore', 'f1-score', 'Accuracy', 'True Positives', 'True Negatives', 'ROC AUC', 'PR AUC'])
    
    return borda, metric_final, metric_final_tf
    

