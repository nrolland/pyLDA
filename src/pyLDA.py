'''
Created on Jan 31, 2010
 
@author: nrolland
 
From Gregor Heinrich's most excellent text 'Parameter estimation for text analysis'
'''
 
import sys, getopt
import collections
import numpy, operator
from other  import *

def flatten(x):
    result = []
    for el in x:
        #if isinstance(el, (list, tuple)):
        if hasattr(el, "__iter__") and not isinstance(el, basestring):
            result.extend(flatten(el))
        else:
            result.append(el)
    return result
 
 
 
class BagOfWordDoc():
    '''
A document contains for each term_id, the word count f the term in the document
The vocabulary of the document is all the term_id whose word count is not null
'''
    def __init__(self):
        self.data = collections.defaultdict(lambda: 0)
      
    def __getattr__(self, *args):
        return self.data.__getattribute__(*args)
    
    def vocabulary(self):
        return self.data.keys()
 
    def wordcount(self):
        return reduce(operator.add, self.data.values(), 0)
    
    def wordtoterm(self):
        return flatten([[ term_id for i in range(self[term_id]) ]for term_id in self ]);
    
    
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
            sub_index =  acc_index[1:]
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
        beta = 0.1
        ntopics = 10
        niters = 3
        
        savestep = 100000
        tsavemostlikelywords = 20
 
        docs = SparseDocCollection()
        docs.read('../trainingdocs', 'enron.dev.txt')
        
        ntopic_by_doc_topic = numpy.zeros((len(docs.documents), ntopics ))
        ntopic_by_doc = numpy.zeros((len(docs.documents) ))
        nterm_by_topic_term = numpy.zeros((ntopics, len(docs.vocabulary)))
        nterm_by_topic = numpy.zeros((ntopics ))
        z_doc_word = numpy.zeros((len(docs.documents), len(docs.vocabulary) ))
          
         
        model_init = [1. / ntopics for x in range(ntopics)]
        multinomial = numpy.random.multinomial
 
        def add(doc_id, term_id, qtty=1):
            ntopic_by_doc_topic[doc_id,z_doc_word[doc_id,term_id]] += qtty
            ntopic_by_doc [doc_id] += qtty
            nterm_by_topic_term[z_doc_word[doc_id,term_id],term_id] += qtty
            nterm_by_topic [z_doc_word[doc_id,term_id]] += qtty
        def remove(doc_id, word_id):
            add(doc_id, word_id, -1)
 
        def sampleparam(nterm_by_topic_forthisword, nterm_by_topic, ntopic_by_doc_topic_forthisdoc, ntopic_by_doc):
            return [(nterm_by_topic_forthisword[topic] + beta) / (nterm_by_topic[topic] + beta * len(docs.vocabulary)) * \
                    (ntopic_by_doc_topic_forthisdoc[topic] + alpha) / (ntopic_by_doc[doc_id] + alpha * ntopics) for topic in range(ntopics)] 
                        
        def saveit():
            for topic in range(ntopics):
                index = indicenbiggest(nterm_by_topic_term[topic,:], tsavemostlikelywords)
                print map(lambda i:docs.vocabulary[i],index)

                  
        for iter in range(niters):
            for doc_id, doc in enumerate(docs.documents):
                if iter == 0: 
                    topics_for_terms = multinomial(1, model_init, doc.wordcount())
                    
                    for i_word, term_id in enumerate(doc.wordtoterm()):
                        z_doc_word[doc_id,term_id] = indice(topics_for_terms[i_word])
                        add(doc_id, term_id)
                else:
                    for term_id in doc.wordtoterm():
                        remove(doc_id, term_id)
                        new_topic = multinomial(1, sampleparam(nterm_by_topic_term[:,term_id], nterm_by_topic, ntopic_by_doc_topic[doc_id,:], ntopic_by_doc ))
                        z_doc_word[doc_id,term_id] = indice(new_topic)
                        add(doc_id, term_id)
                        
            print iter
            #if iter - (iter/savestep)*savestep == 0:
            #    saveit()
        saveit()
 
        print "fini"
        
        
    except Usage, err:
        print >>sys.stderr, err.msg
        print >>sys.stderr, "for help use --help"
        return 2
 
 
if __name__ == "__main__":
    sys.exit(main())
