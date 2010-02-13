'''
Created on Jan 31, 2010
@author: nrolland
From Gregor Heinrich's most excellent text 'Parameter estimation for text analysis'
'''
 
import sys, getopt
import collections, math, array,numpy
from scipy.special import gamma, gammaln
import Image, ImageOps

import random, operator


np = 'numpy' in sys.modules

def ismultiple(i, n):
    return i - (i/n)*n == 0
        
def indice(a):
    if np:
        i =numpy.argmax(a)
    else:        
        for i,val in enumerate(a):
            if val > 0:
                break
    return i

def righshift(ar, pos = 0):
    ar[pos + 1:] = ar[pos:-1]
    ar[pos] = float('-Infinity')
    return ar
 
def indicenbiggest(ar,n):
    ln = min(abs(n),len(ar))
    n_biggest = [float('-Infinity')] *ln
    n_biggest_index = [0] *ln
    
    for i, val in enumerate(ar):
        for i_biggest in xrange(n):
            if val > n_biggest[i_biggest]:
                n_biggest = righshift(n_biggest, i_biggest)
                n_biggest[i_biggest] = val
                n_biggest_index = righshift(n_biggest_index, i_biggest)
                n_biggest_index[i_biggest] = i
                break
 
    return n_biggest_index

def testindicebiggest():
    ar = [54.0, 50.0, 53.0, 47.0, 50.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0]
    print indicenbiggest(ar,20)
    
def topics2image2(ar, zoom = 1):
    square_size = int(math.sqrt(len(ar[0])))
    topics = Image.new("L", ((square_size*zoom+5)* len(ar), square_size*zoom+10),200)
    imzoom = numpy.ones((zoom,zoom), numpy.uint8)
    maxval = numpy.ones(square_size)

    for topic in range(len(ar)):
        pixels = numpy.zeros((square_size,square_size), numpy.uint8)
        for i in range(square_size):
            pixels[i,:]  =  map(lambda x:255*x, ar[topic][i*square_size:(i+1)*square_size ])
        #print pixels
        pixels = numpy.kron(pixels, imzoom)
        pixels = (255 - pixels)
        img = Image.fromarray(pixels,"L")#.convert("RGB")
        
#        img = ImageOps.convert(img)                
        img = ImageOps.autocontrast(img)        
        img = ImageOps.colorize(img, (0,0,0), (255,255,255))
        topics.paste(img, (1+(square_size*zoom+5)*topic, 5))
    return topics
 
def flatten(x):
    result = []
    for el in x:
        #if isinstance(el, (list, tuple)):
        if hasattr(el, "__iter__") and not isinstance(el, basestring):
            result.extend(flatten(el))
        else:
            result.append(el)
    return result
 
def mat(shape, val=0):
    if np:
        ret = numpy.zeros(shape)
    else:
        ret = [val]*shape[-1]
        for n_for_dim in reversed(shape[:-1]):
            tret = []
            for i in range(n_for_dim):
                tret.append(ret[:])
            ret= tret
            
    return ret

def zeros(shape):
    return mat(shape, 0)

def ones(shape):
    return mat(shape, 1)

def roundmat(mat):
    ret = zeros((len(mat), len(mat[0])))
    for i in range(len(mat)):
        for j in range(len(mat[0])):
            ret[i][j] = round(255 * mat[i][j])
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

def normalize(param):
    if np:
        param /= numpy.sum(param)
    else:
        s = sum(param)
        param = [param[i] / s for i in range(len(param))]
    return param


def multinomial(n_add, param, n_dim = 1, normalizeit = True):
    '''
n_add : number of samples to be added for each draw
param : param of multinomial law
n_dim : number of samples
'''
    if normalizeit:
        param = normalize(param)
    if np:
        res = numpy.random.multinomial(n_add, param, n_dim)           
    else:
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

def gammaln(xx):
    cof =[76.18009172947146,  -86.50532032941677, 24.01409824083091, -1.231739572450155, 0.1208650973866179e-2, -0.5395239384953e-5]
    xx = float(xx)
    y = xx
    x = xx
    tmp = x + 5.5;
    tmp -= (x + 0.5) * math.log(tmp);
    ser = 1.000000000190015;
    for j in xrange(6):
        y += 1
        ser += cof[j] / y;
    return -tmp + math.log(2.5066282746310005 * ser / x)
        
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
        for term in self.vocabulary:
            vocfile.write(str(term)  + "\n")
        vocfile.close()
        
    def loadtest(self, npixels=50, ndocs =500, ntopics=6):#require numpy
        
        topics = numpy.matrix(zeros((ntopics,npixels**2)))

        i_count = 0
        for val in  [0,2,4]: 
            topics[i_count] = normalize(oneinrow(zeros((npixels,npixels)), val).flatten())
            i_count +=1

        for val in [0,2,3]: 
            topics[i_count] = normalize(oneincol(zeros((npixels,npixels)), val).flatten())
            i_count +=1
        
        imtopics = topics2image2(topics.tolist(),10)
        imtopics.save("originaltopics.png")
        
        alpha = [1./ntopics]* ntopics
        for doc_id in range(ndocs):
            topicsproportions = numpy.random.dirichlet(alpha)
            distrib =  topicsproportions * topics

            doc = BagOfWordDoc()
            words = multinomial(npixels**2, distrib.tolist()[0], 1)
            for word_id, wordcount in enumerate(words): 
                if wordcount > 0:
                    doc[word_id] = wordcount
            self.append(doc)
            
        vocab = dict()
        for term_id in range(npixels**2):
             vocab[term_id] = term_id
        self.vocabulary = range(npixels**2)
        

        
    def read(self, commonfilename, directory="."):
        '''
name : common part of the name e.g. xxx for docword.xxx and vocab.xxx
'''
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
            maxterm_id = max(maxterm_id, term_id)
            minterm_id = min(minterm_id, term_id)
            if verbose_read and last != doc_id and ismultiple(doc_id, 1000):
                print str((doc_id *100) / self.M) + "%"
                last = doc_id
            doc[term_id] = doc_word_count
        if maxterm_id - minterm_id + 1 != self.V:
            print "warning : maxterm_id", maxterm_id, "minterm_id", minterm_id, "V size :", self.V
 
class LDAModel():
    def __init__(self):
        #prior among topic in docs
        self.falpha = 0.5
        #prior among topic in words
        self.fbeta = 0.5
        self.ntopics = 6
        self.niters = 1000
        
        self.savestep = 50
        self.tsavemostlikelywords = 5
        self.docs = SparseDocCollection()

        
    def load(self, nameset, directory='.'):
        self.docs.read(nameset,directory)
        self.ntopic_by_doc_topic = zeros((len(self.docs), self.ntopics))
        self.ntopic_by_doc       = zeros((len(self.docs), ))
        self.nterm_by_topic_term = zeros((self.ntopics, len(self.docs.vocabulary)))
        self.nterm_by_topic      = zeros((self.ntopics, ))
        self.z_doc_word          = [ zeros((doc.Nwords(), )) for doc in self.docs]

    def test(self):    
        self.docs.read('enron.dev.txt','../trainingdocs')
        self.docs.write('toto.txt')
        
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


    def phi_theta(self):
        p = ones((self.ntopics,len(self.docs.vocabulary))) 
        th = ones((len(self.docs),self.ntopics))
        tht = zip(*th)
                
        for topic in range(self.ntopics):
            p[topic,:] = normalize(map(operator.add, self.nterm_by_topic_term[topic], self.beta))
            th[:,topic] = normalize(map(lambda x: x + self.alpha[topic], self.ntopic_by_doc_topic[:,topic]))
            
        return p,th
        
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
            
 
            
    def topics2images(self, name="", zoom = 1):
        phi, theta = self.phi_theta()
        topics = topics2image2(phi, zoom)
        topics.save("topics" + name + ".png")
            
    def info(self):
        print "# of documents : ", len(self.docs)
        print "# of terms  : ", len(self.docs.vocabulary)
        print "# of words  : ", sum(map(lambda doc:doc.Nwords(), self.docs))
        print "# of topics:  ", self.ntopics

    def loglikelihood(self):
        loglike = 0
        for k in xrange(self.ntopics):
            loglike += logdelta(map(operator.add, self.nterm_by_topic_term[k][:], self.beta))
        loglike -= logdelta(self.beta) * self.ntopics

        for m in xrange(len(self.docs)):
            loglike += logdelta(map(operator.add, self.ntopic_by_doc_topic[m][:], self.alpha))
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
            for term_id in doc:
                for i_term_occ in xrange(doc[term_id]):
                    self.remove(doc_id, term_id, self.z_doc_word[doc_id][i_word])
                    param = [(self.nterm_by_topic_term[topic][term_id] + self.beta[term_id]) / ( self.nterm_by_topic[topic] + self.fbeta) * \
                             (self.ntopic_by_doc_topic[doc_id][topic] +  self.alpha[topic] ) / ( self.ntopic_by_doc[doc_id] + self.falpha) for topic in range(self.ntopics)]
                    new_topic = indice(multinomial(1, param))
                    self.z_doc_word[doc_id][i_word] = new_topic
                    self.add(doc_id, term_id, new_topic)
                    i_word += 1
            if n_verbose_iterate > -1 and ismultiple(doc_id, n_verbose_iterate):
                print " doc : ", doc_id ,"/" , len(self.docs)

    def run(self,niters,savestep, burnin = 100):
        old_lik = -999999999999
        
        self.initialize()
        for i_iter in range(niters):
            self.iterate()
            new_lik = self.loglikelihood()
            if i_iter - (i_iter/savestep)*savestep == 0:
                if verbose:
                    print "saving image"
                self.topics2images(str(i_iter),10)
                print "Likelihood :", self.loglikelihood(), "iteration #", i_iter
            if (new_lik - old_lik)/old_lik < 1.0/100 and i_iter > burnin:
                print "converged", "iter #:", i_iter
                return
            
        
class Usage(Exception):
    def __init__(self, msg):
        self.msg = msg

verbose =False
verbose_read = False
n_verbose_iterate  = 50

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
        
        if False:
            model.load('enron.dev.txt','../trainingdocs')
        elif False:
            model.docs.loadtest(5, 1000)
            model.docs.write('test.bigger.txt')
        elif True:
            model.load('test.bigger.txt')

        model.saveit(True, False, False)
        model.info()
        model.run(300, 1)
        model.saveit(True, True, True)
        
        print "fin"
        
    except Usage, err:
        print >>sys.stderr, err.msg
        print >>sys.stderr, "for help use --help"
        return 2
  
if __name__ == "__main__":
    sys.exit(main())
