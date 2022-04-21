import weakLearner as wl
import numpy as np
import scipy.stats as stats
import scipy
from numpy import inf

#---------------Instructions------------------#

# Here you will have to reproduce the code you have already written in
# your previous assignment.

# However one major difference is that now each node non-terminal node of the
# tree  object will have  an instance of weaklearner...

# Look for the missing code sections and fill them.
#-------------------------------------------#

class Node:
    def __init__(self,purity,klasslabel='',pdistribution=[],score=0,split=[],fidx=-1,wlearner=None):


        self.lchild=None
        self.rchild=None
        self.klasslabel=klasslabel
        self.pdistribution=pdistribution
        self.score=score
        self.wlearner=wlearner
        self.purity = purity
        self.split=split
        self.fidx=fidx

    def set_childs(self,lchild,rchild):
        self.lchild=lchild
        self.rchild=rchild

    def isleaf(self):
        return self.lchild==None and self.rchild==None
    
    def isless_than_eq(self, X):
        return X[self.fidx] <= self.split


    def get_str(self):
        """
            returns a string representing the node information...
        """
        if self.isleaf():
            return 'C(posterior={},class={},Purity={})'.format(self.pdistribution, self.klasslabel,self.purity)
        else:
            return 'I(Fidx={},Score={},Split={})'.format(self.fidx,self.score,self.split)


class DecisionTree:

    def __init__(self, purity, exthreshold=5, maxdepth=10,
     weaklearner="Conic", pdist=False, nsplits=10, nfeattest=None):
        '''
        Input:
        -----------------
            exthreshold: Number of examples to stop splitting, i.e. stop if number examples at a given node are less than exthreshold
            maxdepth: maximum depth of tree upto which we should grow the tree. Remember a tree with depth=10
            has 2^10=1K child nodes.
            weaklearner: weaklearner to use at each internal node.
            pdist: return posterior class distribution or not...
            nsplits: number of splits to use for weaklearner
        '''
        self.purity = purity
        self.maxdepth=maxdepth
        self.exthreshold=exthreshold
        self.weaklearner=weaklearner
        self.nsplits=nsplits
        self.pdist=pdist
        self.nfeattest=nfeattest
        assert (weaklearner in ["Conic", "Linear","Axis-Aligned","Axis-Aligned-Random"])
        #pass

    def getWeakLearner(self):
        if self.weaklearner == "Conic":
            return wl.ConicWeakLearner(self.nsplits)
        elif self.weaklearner== "Linear":
            return wl.LinearWeakLearner(self.nsplits)
        elif self.weaklearner == "Axis-Aligned":
            return wl.WeakLearner()
        else:
            return wl.RandomWeakLearner(self.nsplits,self.nfeattest)

        #pass
        
    def train(self, X, Y):
        
        nexamples,nfeatures=X.shape

        
        self.tree=self.build_tree(X,Y,0)
        
        
    def build_tree(self, X, Y, depth):
        nexamples, nfeatures=X.shape
        classes,class_counts=np.unique(Y,return_counts=True)
        
        n=len(X)
    
        Pc=class_counts/float(nexamples)
        purity=max(Pc)
        
        if purity>=self.purity or n<=self.exthreshold or depth==self.maxdepth:
            midx=np.argmax(np.array(Pc))
            print "C(class=",classes[midx],",Purity=",purity,")"

            return Node(purity=purity,klasslabel=classes[midx])

        
        SPLIT,MINGAIN,XLIDX,XRIDX,f_idx=self.getWeakLearner().train(X,Y)
                

        if(MINGAIN==0 or XLIDX==[] or XRIDX==[]):
            klaslab=classes[np.argmax(class_counts/float(nexamples))]
            print "C(class=",klaslab,",Purity=",purity,")"
            return Node(purity,klasslabel=klaslab)
        else:
            
            DY=X[XLIDX]
            DN=X[XRIDX]
            internal_node=Node(purity=Pc,klasslabel='',score=MINGAIN,split=SPLIT,fidx=f_idx)
            print "Creating Left Child Node With", len(DY) ,"Examples, and Right Node with ",len(DN),"Examples"
            
            lchild=self.build_tree(DY,Y[XLIDX],depth+1)
            rchild=self.build_tree(DN,Y[XRIDX],depth+1)
            internal_node.set_childs(lchild,rchild)
            return internal_node
        

    def test(self, X):
        
  
        nexamples, nfeatures=X.shape
        pclasses=self.predict(X)
        
    
        return pclasses

    


    def predict(self, X):
    
        z=[]
        
        for idx in range(X.shape[0]):
            
            z.append(self._predict(self.tree,X[idx,:]))
            
        
        return z 
    
    def _predict(self,node, X):

        if node.isleaf():
            return node.klasslabel
        
        else:
            if self.getWeakLearner().evaluate(node,X):

                return self._predict(node.lchild,X)
            else:
                return self._predict(node.rchild,X)

        

    def __str__(self):
        
        return self.__print(self.tree)        
        
     
    def find_depth(self):
        
        return self._find_depth(self.tree)
    
    
    def _find_depth(self,node):
        if not node:
            return
        if node.isleaf():
            return 1
        else:
            return max(self._find_depth(node.lchild),self._find_depth(node.rchild))+1
        
    def __print(self,node,depth=0):
        
        ret = ""

        # Print right branch
        if node.rchild:
            ret += self.__print(node.rchild,depth+1)

        # Print own value
        
        ret += "\n" + ("    "*depth) + node.get_str()

        # Print left branch
        if node.lchild:
            ret += self.__print(node.lchild,depth+1)
        
        return ret