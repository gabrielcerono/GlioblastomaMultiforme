from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from random import randint
import pandas as pd
from sklearn.metrics import matthews_corrcoef
import xgboost as xgb
from xgboost.sklearn import XGBClassifier
from imblearn.ensemble import BalancedBaggingClassifier

class RankingRE():
  def __init__(self, X, y, loops, classifier):
   #X and Y in pandas dataframe.
    self.X = X
    self.y = y
    self.loops = loops
    self.classifier = classifier
    
  def ranking_by_matthew_punishment(self):

    std = np.zeros(len(self.X.columns),)
    rankings = np.zeros(len(self.X.columns),)

    for x in range(self.loops):
      mttavg= []  
      seed = randint(0, 10000)
    #Splits the dataset in a further set of rank feature. 1/3 for training 1/3 for feature ranking / 1/3 for validation
      X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size=0.33, random_state= seed)      
      X_train, X_rank, y_train, y_rank = train_test_split(X_train, y_train, test_size=0.5, random_state= seed)
    #Initializing the classifier
      classif = self.classifier
  #Fits the Random forest and we calculate a matthew score. 
      classif.fit(X_train, y_train)
      mttoriginal = matthews_corrcoef(y_rank, classif.predict(X_rank))
  #That's for the ranking, now we gotta do for the validation set
      mttaverage = matthews_corrcoef(y_rank, classif.predict(X_rank))
      mttavg.append(mttaverage)
  #We initialize 2 lists to append values from the next loop.
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

      outcome = np.array(mttrf)
      rankings = np.add(outcome, rankings)
      std = np.vstack((outcome, std))
    
    rankings = np.true_divide(rankings, self.loops)
    std = np.delete(std, -1, axis = 0)
    std = np.std(std, axis = 0)
    std = np.dstack((columnsrf, std))
    std = pd.DataFrame(data = np.squeeze(std, axis = 0), columns =['Categories', 'SD_of_matt_punishment'])
    featuresranks = np.dstack((columnsrf, rankings))
    borda = pd.DataFrame(data = np.squeeze(featuresranks, axis=0), columns=['Categories', 'average-mtt-punishment'])
    borda['ranking'] = borda['average-mtt-punishment'].rank(ascending = False)
    borda = borda.merge(std, on = 'Categories',)
    borda.sort_values(by='average-mtt-punishment', inplace = True, ascending = False)
    matthewmean = np.mean(np.array(mttavg))
    
    return borda, matthewmean
    
