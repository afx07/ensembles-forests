#---------------------------------------------#
#-------| Written By: Sibt ul Hussain |-------#
#---------------------------------------------#


#---------------Instructions------------------#

# You will be writing a super class named WeakLearner
# and then will be implmenting its sub classes
# RandomWeakLearner and LinearWeakLearner. Remember
# all the overridded functions in Python are by default
# virtual functions and every child classes inherits all the
# properties and attributes of parent class.

# Your task is to  override the train and evaluate functions
# of superclass WeakLearner in each of its base classes.
# For this purpose you might have to write the auxiliary functions as well.

#--------------------------------------------------#

# Now, go and look for the missing code sections and fill them.
#-------------------------------------------#
import random

import numpy as np
import scipy.stats as stats
from numpy import inf

class WeakLearner: # A simple weaklearner you used in Decision Trees...
    """ A Super class to implement different forms of weak learners...


    """
    def __init__(self):
        """
        Input:


        """
        #print "   "
        pass

    def train(self, X, Y):

        nexamples,nfeatures=X.shape

        #-----------------------TODO-----------------------#
        #--------Write Your Code Here ---------------------#

        #---------End of Your Code-------------------------#
        return score, Xlidx,Xridx

    def evaluate(self,node,X):
        """
        Evalute the trained weak learner  on the given example...
        """
        #-----------------------TODO-----------------------#
        #--------Write Your Code Here ---------------------#

        #---------End of Your Code-------------------------#

    def evaluate_numerical_attribute(self,feat, Y):
        
        classes=np.unique(Y)
        nclasses=len(classes)
        sidx=np.argsort(feat)
        
        f=feat[sidx] # sorted features
        sY=Y[sidx] # sorted features class labels...
        # YOUR CODE HERE
        mid_pts=[]
        
        Pc=([np.count_nonzero(Y==x )/(1.0*len(Y)) for x in classes])
        
        dataset_Entropy=stats.entropy(Pc,base=2)
        #print f
        ni=([np.count_nonzero(Y==r ) for r in classes])

        nvi=[]
        
        
        for j in range(len(f)-2):
            if f[j+1]!=f[j]:
                mid=(f[j+1]+f[j])/2.0
                mid_pts.append(mid)
         
                
                
                nvi.append(([len(sY[(sY==a) & (f<=mid)]) for a in classes]))
                
                
        best_v=0
        best_score=0
        gain=[]
        split_score=[]
        Xlidx=[]
        Xridx=[]
        for i in range(len(mid_pts)):
            PDy=[]
            PDn=[]
            
            for j in range(nclasses):
                    PDy.append(nvi[i][j]/(sum(nvi[i])*1.0))
                    
                
                    PDn.append((ni[j]-nvi[i][j])/(1.0*sum([ni[q]-nvi[i][q] for q in range(len(nvi[i]))])))
                    

            HDy=stats.entropy(PDy,base=2)
            HDn=stats.entropy(PDn,base=2)
            
            ny=np.count_nonzero(f<=mid_pts[i])
            n=len(f)*1.0

            HDyDn=ny/n*HDy+(n-ny)/n*HDn
            Xlidx.append(np.where(feat<=mid_pts[i]))
            Xridx.append(np.where(feat>mid_pts[i]))
            split_score.append(HDyDn)

            gain.append(dataset_Entropy-HDyDn)
   
        maxval=max(gain)  
        
        midx=gain.index(maxval)
        

        return mid_pts[midx],split_score[midx],Xlidx[midx],Xridx[midx]
    
    def calculateEntropy(self, Y, mship):

        lexam=Y[mship]
        rexam=Y[np.logical_not(mship)]

        pleft= len(lexam) / float(len(Y))
        pright= 1-pleft

        pl= stats.itemfreq(lexam)[:,1] / float(len(lexam)) + np.spacing(1)
        pr= stats.itemfreq(rexam)[:,1] / float(len(rexam)) + np.spacing(1)

        hl= -np.sum(pl*np.log2(pl))
        hr= -np.sum(pr*np.log2(pr))

        sentropy = pleft * hl + pright * hr

        return sentropy


class RandomWeakLearner(WeakLearner):  # Axis Aligned weak learner....

    def __init__(self, nsplits=+np.inf, nrandfeat=None):

        WeakLearner.__init__(self) # calling base class constructor...
        self.nsplits=nsplits
        self.nrandfeat=nrandfeat


    def train(self, X, Y):

        nexamples,nfeatures=X.shape

        if(not self.nrandfeat):
            self.nrandfeat=int(np.round(np.sqrt(nfeatures)))
        selected_features=random.sample(range(nfeatures),self.nrandfeat)

            
        SPLIT,MINGAIN,XLIDX,XRIDX=0,99,[],[]
        split,mingain,Xlidx,Xridx=0,-1,0,0
        f_idx=0
        for i in selected_features:
            split,mingain,Xlidx,Xridx=self.findBestRandomSplit(X[:,i],Y)

                
            if(mingain<=MINGAIN):
                SPLIT=split;MINGAIN=mingain;XLIDX=Xlidx;XRIDX=Xridx
                f_idx=i
        

        
        

        return SPLIT,MINGAIN,XLIDX,XRIDX,f_idx

        #---------End of Your Code-------------------------#
        #return minscore, bXl,bXr


    def findBestRandomSplit(self,feat,Y):
        splitvalue,minscore,Xlidx,Xridx=0,0,0,0

        frange=np.max(feat)-np.min(feat)
        if(np.isinf(self.nsplits)):
            splitvalue,minscore,Xlidx,Xridx=evaluate_numeric_attribute(feat,Y)
        else:
            selected_splits=random.sample(feat,self.nsplits)
            split_entropy=[]
            for split in selected_splits:
                mship=(feat<=split)
                
                split_entropy.append(self.calculateEntropy(Y,mship))
            min_ent=min(split_entropy)
            midx=split_entropy.index(min_ent)
            best_split=selected_splits[midx]
            Xlidx=np.where(feat<=best_split)
            Xridx=np.where(feat>best_split)
            
            
                
               
        return best_split, min_ent, Xlidx, Xridx

    def evaluate(self,node,X):
        
        if X[node.fidx]<=node.split:
            return True
      
        else:
            return False
            




# build a classifier ax+by+c=0
class LinearWeakLearner(RandomWeakLearner):  # A 2-dimensional linear weak learner....
    """ An Inherited class to implement 2D line based weak learner using
        a random set of features from the given set of features...


    """
    def __init__(self, nsplits=10):
        """
        Input:
            nsplits = How many splits to use for each choosen line set of parameters...

        """
        RandomWeakLearner.__init__(self,nsplits)

        #pass
    

    def train(self,X, Y):

        nexamples,nfeatures=X.shape
        feat=random.sample(range(nfeatures),2)
        split_entropy=[]
        split_values=[]
        mship=[]
        
        for i in range(self.nsplits):
            a=random.uniform(-1,1)
            b=random.uniform(-1,1)
            c=random.uniform(-1,1)
            split_values.append([a,b,c])
            
            arr=(a*X[:,feat[0]])+(b*X[:,feat[1]])+c
            mships=(arr<=0)  
          
            
            mship.append(mships)
            
            
            
            
            split_entropy.append(self.calculateEntropy(Y,mships))

        left=[]
        right=[]
            

            
            
        minscore=min(split_entropy)
        midx=split_entropy.index(minscore)
        split=split_values[midx]
        mships=mship[midx]
        for i in range(len(X)):
            if mships[i]==True:
                left.append(i)
            else:
                right.append(i)
        fidx=feat
        
     
             
            
        
        
        
        return split, minscore, left, right,fidx


    def evaluate(self,node,X):
        
        arr=(node.split[0]*X[node.fidx[0]])+(node.split[1]*X[node.fidx[1]])+node.split[2]
        
        if arr<=0:
            return True
      
        else:
            return False

        


#build a classifier a*x^2+b*y^2+c*x*y+ d*x+e*y+f
class ConicWeakLearner(RandomWeakLearner):  # A 2-dimensional linear weak learner....
    """ An Inherited class to implement 2D Conic based weak learner using 
        a random set of features from the given set of features...


    """
    def __init__(self, nsplits=10):
        """
        Input:
            nsplits = How many splits to use for each choosen line set of parameters...
            
        """
        RandomWeakLearner.__init__(self,nsplits)
        
        #pass
    
    def train(self,X, Y):

        nexamples,nfeatures=X.shape
        feat=random.sample(range(nfeatures),2)
        split_entropy=[]
        split_values=[]
        mship=[]
        
        for i in range(self.nsplits):
            a=random.uniform(-1,1)
            b=random.uniform(-1,1)
            c=random.uniform(-1,1)
            d=random.uniform(-1,1)
            e=random.uniform(-1,1)
            f=random.uniform(-1,1)
            
            
            split_values.append([a,b,c,d,e,f])
            
            arr= (a*np.power(X[:,feat[0]],2))+(b*np.power(X[:,feat[1]],2))+(c*X[:,feat[0]]*X[:,feat[1]])+(d*X[:,feat[0]])
            arr+=(e*X[:,feat[1]])+f
            
            mships=(arr<=0)    
            
            mship.append(mships)
            
            split_entropy.append(self.calculateEntropy(Y,mships))

        left=[]
        right=[]
            

            
            
        minscore=min(split_entropy)
        midx=split_entropy.index(minscore)
        split=split_values[midx]
        mships=mship[midx]
        for i in range(len(X)):
            if mships[i]==True:
                left.append(i)
            else:
                right.append(i)
        fidx=feat
        
     
             
            
        
        
        
        return split, minscore, left, right,fidx


    def evaluate(self,node,X):
        p=node.split
        f=node.fidx
        x=X[f[0]]
        y=X[f[1]]
        x_sqr=x*x
        y_sqr=y*y
        
        
        arr= (p[0]*x_sqr)+(p[1]*y_sqr)+(p[2]*x*y)+(p[3]*x)+(p[4]*y)+p[5]
        
        if arr<=0:
            return True
      
        else:
            return False

        

