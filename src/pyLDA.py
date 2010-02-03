'''
Created on Jan 31, 2010
@author: nrolland
From Gregor Heinrich's most excellent text 'Parameter estimation for text analysis'
'''
 
import sys, getopt
import collections, array, numpy
import random, operator


def flatten(x):
    result = []
    for el in x:
        #if isinstance(el, (list, tuple)):
        if hasattr(el, "__iter__") and not isinstance(el, basestring):
            result.extend(flatten(el))
        else:
            result.append(el)
    return result
 
def zeros(shape):
    if 'numpy' in sys.modules:
        ret = numpy.zeros(shape)
    else:
        ret = 0
        for n_for_dim in reversed(shape):
            ret = [ret]*n_for_dim
    return ret


def multinomial(n_add, param, n_dim = 1):
    #s = sum(param)
    if 'numpy' in sys.modules:
        if n_dim == 1:
            res = numpy.random.multinomial(n_add, param)
        else:
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
          res = flatten(res)
    return res

 
class BagOfWordDoc():
    '''
A document contains for each term_id, the word count f the term in the document
The vocabulary of the document is all the term_id whose word count is not null
'''
    def __init__(self):
        self.data = collections.defaultdict(lambda: 0)
        self._vocabulary = None
        self._wordcount  = None
        self._wordtoterm = None

    #def __getattr__(self, *args):
    #    return self.data.__getattribute__(*args)

    def __setitem__(self, *args):
        #print "set"
        return self.data.__setitem__(*args)
    def __getitem__(self, *args):
        #print "get"
        return self.data.__getitem__(*args)
    
    def vocabulary(self):
        self._vocabulary = self._vocabulary or self.data.keys()
        return self._vocabulary
 
    def wordcount(self):
        self._wordcount = self._wordcount or reduce(operator.add, self.data.values(), 0)
        return self._wordcount
    
    def wordtoterm(self):
        self._wordtoterm = self._wordtoterm or flatten([[ term_id for i in range(self.data[term_id]) ]for term_id in self.data ])
        return self._wordtoterm 
    
    
class SparseDocCollection(object):
    '''
A class to read collection of documents in a bag of word sparse format:
docword.filename :
M number of docs
V numbers of words in vocabulary
L numbers of (documents, words) occurence that follows
doc_id word_id doc_word_count
vocab.filename:
list of words
'''
    def __init__(self):
        '''
directory : director of files
'''
        pass
        
    def read(self, directory, commonfilename):
        '''
name : common part of the name e.g. xxx for docword.xxx and vocab.xxx
'''
        
        self.vocabulary = file(directory+"/vocab." + commonfilename).readlines()
        docfile = file(directory+"/docword." + commonfilename)
        self.M=int(docfile.readline())
        self.V=int(docfile.readline())
        self.L=int(docfile.readline())
        
        last = -1
        self.documents = list()
        for line in docfile.readlines():
            doc_id, word_id, doc_word_count = map(lambda st: int(st), line.split(' '))
            if len(self.documents) < doc_id:
                doc = BagOfWordDoc()
                self.documents.append(doc)
            doc = self.documents[-1]
            if last != doc_id and doc_id - round(doc_id /1000, 0) *1000 == 0:
                print str((doc_id *100) / self.M) + "%"
                last = doc_id
            #print doc.keys()
            doc[word_id] +=1
            
def indice(a):
    for i,val in enumerate(a):
        if val > 0:
            break
    return i
 
def indicenbiggest2(i,val, acc, acc_index):
    ret = acc_index
    if val > acc[0]:
        acc[0] = val
        acc_index[0] = i
    else:
        if len(acc)>1:
            sub = acc[1:]
            sub_index = acc_index[1:]
            subret = indicenbiggest2(i,val, sub,sub_index)
            acc_index[1:] = sub_index[:]
            acc[1:] = sub[:]
    
    return ret
 
def indicenbiggest(ar,n):
    acc = [float('-Infinity') for i in range(min(abs(n),len(ar)))]
    acc_index = [ 0 for i in range(min(abs(n),len(ar)))]
    
    for i, val in enumerate(ar):
        indicenbiggest2(i,val, acc, acc_index)
 
    return acc_index
 
 
class Usage(Exception):
    def __init__(self, msg):
        self.msg = msg
 
def main(argv=None):
    if argv is None:
        argv = sys.argv
    try:
        try:
            opts, args = getopt.getopt(argv[1:], "ht", ["help"])
        except getopt.error, msg:
            raise Usage(msg)
        
        for o, a in opts:
            if o in ("-h", "--help"):
                raise Usage(__file__ +' -alpha ')
            if o in ("-t", "--test"):
                print >>sys.stdout, 'TEST MODE'
                test = True
        if len(args)<0 :
            raise Usage("arguments missing")
        alpha = 0.5
        beta = 0.5
        ntopics = 10
        niters = 5
        
        
        savestep = 100000
        tsavemostlikelywords = 20
 
        docs = SparseDocCollection()
        docs.read('../trainingdocs', 'enron.dev.txt')

        ntopic_by_doc_topic = zeros((len(docs.documents), ntopics ))
        ntopic_by_doc = zeros((len(docs.documents), ))
        nterm_by_topic_term = zeros((ntopics, len(docs.vocabulary)))
        nterm_by_topic = zeros((ntopics, ))
        z_doc_term = zeros (  (len(docs.documents), len(docs.vocabulary)) )
        
        nulltopiccount= [0]*ntopics
          
  
        model_init = [1. / ntopics for x in range(ntopics)]
        #multinomial = numpy.random.multinomial
 
        def add(doc_id, term_id, qtty=1):
            ntopic_by_doc_topic[doc_id][z_doc_term[doc_id][term_id]] += qtty
            ntopic_by_doc [doc_id] += qtty
            nterm_by_topic_term[z_doc_term[doc_id][term_id]][term_id] += qtty
            nterm_by_topic [z_doc_term[doc_id][term_id]] += qtty
        def remove(doc_id, word_id):
            add(doc_id, word_id, -1)
 
        def sampleparam(nterm_by_topic_forthisword, nterm_by_topic, ntopic_by_doc_topic_forthisdoc, ntopic_by_doc):
            #print "nterm_by_topic_forthisword", nterm_by_topic_forthisword
            #print "nterm_by_topic",nterm_by_topic
            #print "ntopic_by_doc_topic_forthisdoc",ntopic_by_doc_topic_forthisdoc
            #print "ntopic_by_doc",ntopic_by_doc
            param = [(nterm_by_topic_forthisword[topic] + beta) / (nterm_by_topic[topic] + beta) * \
                    (ntopic_by_doc_topic_forthisdoc[topic] + alpha) / (ntopic_by_doc[doc_id] + alpha) for topic in range(ntopics)]
            s = sum(param)
            param = [param[i] / s for i in range(ntopics)]
            if param == nulltopiccount:
                pass
            return param
        
        def printmostfreqtopic():
            ndocs_topics = [0]*ntopics            
            for doc_id, doc in enumerate(docs.documents):
                for topic in range(ntopics):
                    ndocs_topics[topic] += ntopic_by_doc_topic[doc_id][topic]
            print "Most topic among docs :",   sum(ndocs_topics) 
                 
        def saveit():
            totalfreq = [0]*len(docs.vocabulary)
            for doc_id, doc in enumerate(docs.documents):
                words = doc.wordtoterm()
                for term_id in doc.vocabulary():
                    totalfreq[term_id] += doc[term_id]
            print "Most frequent words among docs :",  map(lambda i:docs.vocabulary[i],indicenbiggest(totalfreq, tsavemostlikelywords))

            for topic in range(ntopics):
                index = indicenbiggest(nterm_by_topic_term[topic][:], tsavemostlikelywords)
                print "MF word for topic", topic, map(lambda i:docs.vocabulary[i],index)

            ndocs_topics = [0]*ntopics            
            for doc_id, doc in enumerate(docs.documents):
                for topic in range(ntopics):
                    ndocs_topics[topic] += ntopic_by_doc_topic[doc_id][topic]
            print "Most topic among docs :",   ndocs_topics

                  
        param = 0
        for iter in range(niters):
            for doc_id, doc in enumerate(docs.documents):
                #print doc_id 
                
                words = doc.wordtoterm()
                wordcount = doc.wordcount()
                if iter == 0:
                    topic_for_words = multinomial(1, model_init,wordcount)
                    #print doc.wordcount()
                    #print "topics_for_words", type(topics_for_words), len(topics_for_words)
                    #print "topics_for_words", type(topics_for_words[0]), len(topics_for_word
                    #print "doc", (doc.wordtoterm())[:10]
                    #input()         
                    for i_word, term_id in enumerate(words):
                        #print "doc_id", type(doc_id),doc_id
                        #print "term_id", type(term_id),term_id
                        #print "zdoc_word", type(z_doc_term),len(z_doc_term[doc_id])
                        #print "i_word",i_word
                        #print "term_id",term_id
                        #print "topic", topics_for_words[i_word], indice(topics_for_words[i_word])
                        #print z_doc_term[doc_id][term_id]
                        z_doc_term[doc_id,term_id] = indice(topic_for_words[i_word])
                        if i_word == 0 and doc_id ==0:
                            print  z_doc_term[doc_id,term_id]
                        #print "ok"
                        add(doc_id, term_id)
                else:                    
                    for i_word, term_id in enumerate(words):
                        remove(doc_id, term_id)
                        #print "zip(*nterm_by_topic_term)[term_id]", type(zip(*nterm_by_topic_term)[term_id]), len(zip(*nterm_by_topic_term)[term_id])
                        #print "nterm_by_topic", type(nterm_by_topic), len(nterm_by_topic)
                        #print "zip(*ntopic_by_doc_topic)[doc_id]", type(zip(*ntopic_by_doc_topic)[doc_id]), len(zip(*ntopic_by_doc_topic)[doc_id])
                        #print "ntopic_by_doc", type(ntopic_by_doc), len(ntopic_by_doc)
                        #input()
                        #param = sampleparam(zip(*nterm_by_topic_term)[term_id], nterm_by_topic, ntopic_by_doc_topic[doc_id], ntopic_by_doc)
                        param = sampleparam(nterm_by_topic_term[:,term_id], nterm_by_topic, ntopic_by_doc_topic[doc_id], ntopic_by_doc)
                        #if sum(nterm_by_topic_term[:,term_id]) == 0:
                        #    pass
                        #if doc_id - (doc_id / 100)*100== 0 :
                        #    if i_word - (i_word / 100)*100== 0 :
                        #        print param
                        new_topic = multinomial(1, param)
                        #print doc_id 
                        #input()
                        #if i_word == 0 and doc_id ==0:
                        #    print  z_doc_term[doc_id,term_id]
                        #    input()
                        #print  z_doc_term[doc_id,term_id],param,new_topic, indice(new_topic)
                        z_doc_term[doc_id][term_id] = indice(new_topic)
                        add(doc_id, term_id)
                    if doc_id - (doc_id / 100)*100== 0:
                        #saveit()
                        print "iter", iter, " doc : ", doc_id ,"/" , len(docs.documents)

                        
            print iter
            #if iter - (iter/savestep)*savestep == 0:
            # saveit()
        saveit()
        print param
        
        print "fini"
        
        
    except Usage, err:
        print >>sys.stderr, err.msg
        print >>sys.stderr, "for help use --help"
        return 2
 
 
if __name__ == "__main__":
    sys.exit(main())
 
