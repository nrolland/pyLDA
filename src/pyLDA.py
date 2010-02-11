'''
Created on Jan 31, 2010
@author: nrolland
From Gregor Heinrich's most excellent text 'Parameter estimation for text analysis'
'''
 
import sys, getopt
import collections, array, numpy
from scipy.special import gamma,gammaln
import random, operator


np = 'numpy' in sys.modules
          
def indice(a):
    if np:
        i =numpy.argmax(a)
    else:        
        for i,val in enumerate(a):
            if val > 0:
                break
    return i
 
def indicenbiggest2(i,val, n_biggest, n_biggest_index):
    ret = n_biggest_index
    if val > n_biggest[0]:
        n_biggest[0] = val
        n_biggest_index[0] = i
    else:
        if len(n_biggest)>1:
            sub = n_biggest[1:]
            sub_index = n_biggest_index[1:]
            subret = indicenbiggest2(i,val, sub,sub_index)
            n_biggest_index[1:] = subret[:]
            n_biggest[1:] = sub[:]    
    return ret
 
def indicenbiggest(ar,n):
    ln = min(abs(n),len(ar))
    n_biggest = [float('-Infinity')] *ln
    n_biggest_index = [0] *ln
    
    for i, val in enumerate(ar):
        indicenbiggest2(i,val, n_biggest, n_biggest_index)
 
    return n_biggest_index
 
 
def flatten(x):
    result = []
    for el in x:
        if hasattr(el, "__iter__") and not isinstance(el, basestring):
            result.extend(flatten(el))
        else:
            result.append(el)
    return result
 
def zeros(shape):
    '''
shape : a tuple 
'''
    if np:
        ret = numpy.zeros(shape)
    else:
        ret = [0]*shape[-1]
        for n_for_dim in reversed(shape[:-1]):
            tret = []
            for i in range(n_for_dim):
                tret.append(ret[:])
            ret= tret
            
    return ret

def oneinrow(ar, row_id):
    tar = ar[:]
    tar[row_id] = numpy.ones(len(ar[row_id]))
    return tar

def oneincol(ar, col_id):
    tar = ar[:]
    tar[:,col_id] = numpy.ones(len(ar[:,col_id]))
    return tar

def logdelta(v):
    sigma = 0
    sigmagammaln = 0
    for i, x_i in enumerate(v):
        sigma        += x_i
        sigmagammaln +=  gammaln(x_i)
    return sigmagammaln - gammaln(sigma)

def multinomial(n_add, param, n_dim = 1, normalize = True):
'''
n_add : number of samples to be added for each draw
param : param of multinomial law
n_dim : number of samples
'''
    if np:
        if normalize:
            param /= numpy.sum(param)
        res = numpy.random.multinomial(n_add, param, n_dim)            
    else:
        if normalize:
            s = sum(param)
            param = [param[i] / s for i in range(self.ntopics)]
        res = []
        cdf = [sum(param[:1+end]) for end in range(len(param))]
        zerosample = [0]*len(param)
        for i_dim in range(n_dim):
            sample = zerosample[:]
            for i_add in range(n_add):
                r = random.random()
                for i_cdf, val_cdf in enumerate(cdf):
                    if r < val_cdf : break
                sample[i_cdf] += 1
            res.append(sample)
    
    if n_dim == 1:
        res = res[0]
    return res

 
class BagOfWordDoc(dict):
    '''
A document contains for each term_id, the word count f the term in the document
The vocabulary of the document is all the term_id whose word count is not null
'''
    def __init__(self):
        self._vocabulary = None
        self._wordcount  = None
        self._words = None
    
    def vocabulary(self):
        self._vocabulary = self._vocabulary or self.keys()
        return self._vocabulary
 
    def Nwords(self):
        self._wordcount = self._wordcount or reduce(operator.add, self.values(), 0)
        return self._wordcount
    
    def words(self):
        self._words = self._words or flatten([ [ term_id for i in range(self[term_id]) ]for term_id in self ])
        return self._words
        
class SparseDocCollection(list):
    '''
A class to read collection of documents in a bag of word sparse format.
The format reads like

in docword.filename :
M number of docs
V numbers of words in vocabulary
L numbers of (documents, words) occurence that follows
doc_id word_id doc_word_count

in vocab.filename:
list of words
'''
    def __init__(self):
        self.vocabulary = []
                
    def write(self, commonfilename, directory="."):
        docfile = file(directory+"/docword." + commonfilename, "w")
        docfile.write(str(len(self))  + "\n") #doc number
        docfile.write(str(len(self.vocabulary)) + "\n")#=int(docfile.readline()) #vocab size
        docfile.write(str(9999) + "\n")#=int(docfile.readline()) #wordcount

        for doc_id, doc in enumerate(self):
            for term_id in doc.vocabulary():
                msg = str(doc_id + 1) + " " + str(term_id + 1) + " " + str(doc[term_id]) + "\n"
                docfile.write(msg)
        docfile.close()        
        vocfile = file(directory+"/vocab." + commonfilename,"w")
        for term in enumerate(self.vocabulary):
            vocfile.write(str(term)  + "\n")
        vocfile.close()
        
    def loadtest(self):
        topics = numpy.matrix(zeros((6,25)))
        topics[0] = oneinrow(zeros((5,5)), 0).flatten()/5
        topics[1] = oneinrow(zeros((5,5)), 2).flatten()/5
        topics[2] = oneinrow(zeros((5,5)), 4).flatten()/5
        topics[3] = oneincol(zeros((5,5)), 0).flatten()/5
        topics[4] = oneincol(zeros((5,5)), 2).flatten()/5
        topics[5] = oneincol(zeros((5,5)), 3).flatten()/5
        
        alpha = [1./len(topics)]* len(topics)
        for doc_id in range(100):
            topicsproportions = numpy.random.dirichlet(alpha)
            distrib =  topicsproportions * topics
 
            doc = BagOfWordDoc()
            words = multinomial(50, distrib.tolist()[0], 1)
            for word_id, wordcount in enumerate(words): 
                if wordcount > 0:
                    doc[word_id] = wordcount
            self.append(doc)
            
        vocab = dict()
        for term_id in range(25):
             vocab[term_id] = term_id
        self.vocabulary = [i for i in range(25)]
        

        
    def read(self, commonfilename, directory=".", verbose = False):
        self.vocabulary = file(directory+"/vocab." + commonfilename).readlines()
        docfile = file(directory+"/docword." + commonfilename)
        self.M=int(docfile.readline()) #doc number
        self.V=int(docfile.readline()) #vocab size
        self.L=int(docfile.readline()) #wordcount
        
        last = -1
        maxterm_id = -1
        minterm_id = +999999999
        for line in docfile.readlines():
            #print line, line.split(' ')
            doc_id, term_id, doc_word_count = map(lambda st: int(st), line.split(' '))
            term_id = term_id -1
            doc_id = doc_id -1
            
            if len(self) < doc_id + 1:
                doc = BagOfWordDoc()
                self.append(doc)
            doc = self[-1]
            doc[term_id] = doc_word_count
            
            maxterm_id = max(maxterm_id, term_id)
            minterm_id = min(minterm_id, term_id)
            if verbose and (last != doc_id and doc_id - round(doc_id /1000, 0) *1000 == 0):
                print str((doc_id *100) / self.M) + "%"
                last = doc_id
        if maxterm_id - minterm_id + 1 != self.V:
            print "warning : vocab size != vocab used"

 
class LDAModel():
    def __init__(self):
        #prior among topic in docs
        self.falpha = 0.5
        #prior among topic in words
        self.fbeta = 0.5
        self.ntopics = 6
        self.niters = 1000
        
        self.savestep = 50
        self.tsavemostlikelywords = 20
        self.docs = SparseDocCollection()

        
    def load(self, nameset, directory='.'):
        self.docs.read(nameset,directory)
        self.ntopic_by_doc_topic = zeros((len(self.docs), self.ntopics))
        self.ntopic_by_doc       = zeros((len(self.docs), ))
        self.nterm_by_topic_term = zeros((self.ntopics, len(self.docs.vocabulary)))
        self.nterm_by_topic      = zeros((self.ntopics, ))
        self.z_doc_word          = [ zeros((doc.Nwords(), )) for doc in self.docs]

        
    def add(self,doc_id, term_id, topic, qtty=1.):
        self.ntopic_by_doc_topic[doc_id][topic] += qtty
        self.ntopic_by_doc      [doc_id]   += qtty             
        self.nterm_by_topic_term[topic][term_id] += qtty
        self.nterm_by_topic     [topic] += qtty
        
    def remove(self, doc_id, term_id, topic):
        self.add(doc_id, term_id, topic, -1.)
    
    def printmostfreqtopic(self):
        ndocs_topics = [0]*self.ntopics            
        for doc_id, doc in enumerate(self.docs):
            for topic in xrange(self.ntopics):
                ndocs_topics[topic] += self.ntopic_by_doc_topic[doc_id][topic]
        print "Most topic among docs :",   sum(ndocs_topics) 
             
    def saveit(self, mfw=False, wordspertopic=False, docspertopic=False):
        if mfw:
            totalfreq = [0]*len(self.docs.vocabulary)
            for doc_id, doc in enumerate(self.docs):
                words = doc.words()
                for term_id in doc.vocabulary():
                    totalfreq[term_id] += doc[term_id]
            print "Most frequent words among docs :",  map(lambda i:self.docs.vocabulary[i],indicenbiggest(totalfreq, self.tsavemostlikelywords))
        if wordspertopic:
            for topic in range(self.ntopics):
                index = indicenbiggest(self.nterm_by_topic_term[topic][:], self.tsavemostlikelywords)
                print "MF word for topic", topic, map(lambda i:self.docs.vocabulary[i],index)
        if docspertopic:
            ndocs_topics = [0]*self.ntopics            
            for doc_id, doc in enumerate(self.docs):
                for topic in range(self.ntopics):
                    ndocs_topics[topic] += self.ntopic_by_doc_topic[doc_id][topic]
            print "Docs per topics :",  "(total)", sum( ndocs_topics), ndocs_topics
    def info(self):
        print "# of documents : ", len(self.docs)
        print "# of terms  : ", len(self.docs.vocabulary)
        print "# of words  : ", sum(map(lambda doc:doc.Nwords(), self.docs))
        print "# of topics:  ", self.ntopics

    def loglikelihood(self):
        loglike = 0
        for k in xrange(self.ntopics):
            loglike += logdelta(self.nterm_by_topic_term[k][:] + self.beta)
        loglike -= logdelta(self.beta) * self.ntopics

        for m in xrange(len(self.docs)):
            loglike += logdelta(self.ntopic_by_doc_topic[m][:]+ self.alpha)
        loglike -= logdelta(self.alpha) * len(self.docs)
        return loglike
        
        
    def initialize(self):
        self.alpha = [self.falpha / self.ntopics] * self.ntopics
        self.beta  = [self.fbeta / len(self.docs.vocabulary)] *len(self.docs.vocabulary)
        
        model_init = [1. / self.ntopics] * self.ntopics
        print "initial seed"
        for doc_id, doc in enumerate(self.docs):
            topic_for_words = multinomial(1, model_init,doc.Nwords())
            i_word = 0
            for term_id in doc:
                for i_term_occ in xrange(doc[term_id]):
                    i_topic =  indice(topic_for_words[i_word])
                    self.z_doc_word[doc_id][i_word] = i_topic
                    self.add(doc_id, term_id,i_topic)
                    i_word += 1     

    def iterate(self):
        for doc_id, doc in enumerate(self.docs):
            i_word =0 
            #print doc_id
            for term_id in doc:
                for i_term_occ in xrange(doc[term_id]):
                    self.remove(doc_id, term_id, self.z_doc_word[doc_id][i_word])
                    param = [(self.nterm_by_topic_term[topic][term_id] + self.beta[term_id]) / ( self.nterm_by_topic[topic] + self.fbeta) * \
                            ( self.ntopic_by_doc_topic[doc_id][topic] +  self.alpha[topic]) / ( self.ntopic_by_doc[doc_id] + self.falpha) for topic in range(self.ntopics)]

                    new_topic = indice(multinomial(1, param))
                    self.z_doc_word[doc_id][i_word] = new_topic
                    self.add(doc_id, term_id, new_topic)
                    i_word += 1
            if doc_id - (doc_id / 500)*500== 0:
                #saveit()
                print " doc : ", doc_id ,"/" , len(self.docs)

    def run(self,niters,savestep):
        old_lik = -999999999999
        
        self.initialize()
        for iter in range(niters):
            print "iteration #", iter
            self.iterate()
            if iter - (iter/savestep)*savestep == 0:
                print self.loglikelihood()
                #self.saveit(False, True , False)
            
        
        


class Usage(Exception):
    def __init__(self, msg):
        self.msg = msg

def main(argv=None):
    if argv is None:
        argv = sys.argv
    try:
        try:
            opts, args = getopt.getopt(argv[1:], "htw", ["help"])
        except getopt.error, msg:
            raise Usage(msg)
        
        for o, a in opts:
            if o in ("-h", "--help"):
                raise Usage(__file__ +' -alpha ')
            if o in ("-t", "--test"):
                print >>sys.stdout, 'TEST MODE'
                test = True
            if o in ("-w", "--write test"):
                print >>sys.stdout, 'TEST MODE'
                test = True
        if len(args)<0 :
            raise Usage("arguments missing")

        model = LDAModel()
        
        #model.load('enron.dev.txt','../trainingdocs')
        #model.docs.loadtest()
        #model.docs.write('test.txt')
        model.load('test.txt')
        model.saveit(True, False, False)
        model.info()
        model.run(5000, 10)
        model.saveit(True, True, True)
        
        print "fin"
        
        
    except Usage, err:
        print >>sys.stderr, err.msg
        print >>sys.stderr, "for help use --help"
        return 2
 
 
if __name__ == "__main__":
    sys.exit(main())
 
