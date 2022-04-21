#---------------------------------------------#
#-------| Written By: Sibt ul Hussain |-------#
#---------------------------------------------#

#---------------Instructions------------------#
# Please read the function documentation before
# proceeding with code writing. 

# For randomizing, you will need to use following functions
# please refer to their documentation for further help.
# 1. np.random.randint
# 2. np.random.random
# 3. np.random.shuffle
# 4. np.random.normal 


# Other Helpful functions: np.atleast_2d, np.squeeze()
# scipy.stats.mode, np.newaxis

#-----------------------------------------------#

# Now, go and look for the missing code sections and fill them.
#-------------------------------------------#
import tree as tr
import numpy as np
import scipy.stats as stats
from numpy import inf
import tools as t



class RandomForest:
    ''' Implements the Random Forest For Classification... '''
    def __init__(self, ntrees=10,treedepth=5,usebagging=False,baggingfraction=0.6,
        weaklearner="Conic",
        nsplits=10,        
        nfeattest=None, posteriorprob=False,scalefeat=True ):        

        self.ntrees=ntrees
        self.treedepth=treedepth
        self.usebagging=usebagging
        self.baggingfraction=baggingfraction

        self.weaklearner=weaklearner
        self.nsplits=nsplits
        self.nfeattest=nfeattest
        
        self.posteriorprob=posteriorprob
        
        self.scalefeat=scalefeat
        

    def findScalingParameters(self,X):

        self.mean=np.mean(X,axis=0)
        self.std=np.std(X,axis=0)

    def applyScaling(self,X):

        X= X - self.mean
        X= X /self.std
        return X

    def train(self,X,Y,vX=None,vY=None):


        nexamples, nfeatures= X.shape

        self.findScalingParameters(X)
        if self.scalefeat:
            X=self.applyScaling(X)

        self.trees=[]
        print "here we will train trees"

        for i in range(self.ntrees):
            
            data=np.zeros((X.shape[0],X.shape[1]+1))
            if Y.ndim==2:
                data[:,-1]=Y[:,0] 
            else:
                data[:,-1]=Y
                
            data[:,:-1]=X
            #putting classes and data together 
            np.random.shuffle(data)

            
            dt=tr.DecisionTree(purity=0.78,maxdepth=self.treedepth,weaklearner=self.weaklearner,exthreshold=11)
            if self.usebagging:
                sample=data[:int(nexamples*self.baggingfraction)]
                
                dt.train(sample[:,:-1],sample[:,-1])
            else:
                dt.train(data[:,:-1],data[:,-1])
                
                
                
        
            self.trees.append(dt)
            

        
    def predict(self, X):
  
        z = []
        
        if self.scalefeat:
            X=self.applyScaling(X)
        for treex in self.trees:
            z.append(treex.predict(X))
            
        arr=np.array(z)
        votes=stats.mode(arr,axis=0)  # to get majority votes
        majority=np.array(votes[0][0])
        

        return majority
        
        

