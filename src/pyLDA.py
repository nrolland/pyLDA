'''
Created on Jan 31, 2010

@author: nrolland

From Gregor Heinrich's most excellent text 'Parameter estimation for text analysis'
'''

import sys, getopt
import collections
import SparseDoc
import numpy

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
                raise Usage(__file__ +' -alpha  ')
            if o in ("-t", "--test"):
                print >>sys.stdout, 'TEST MODE'
                test = True
        if len(args)<0 :
            raise Usage("arguments missing")        
        alpha =  0.5
        beta =  0.1 
        ntopics =  10 
        niters =  1000
         
        savestep =  100 
        tsavemostlikelywords = 20

        docs = SparseDoc.SparseDocCollection()
        docs.read('../trainingdocs', 'enron.dev.txt')
        
        ntopic_by_doc_topic  = numpy.zeros((len(docs.documents), ntopics ))
        ntopic_by_doc        = numpy.zeros((len(docs.documents)          ))
        nterm_by_topic_term  = numpy.zeros((ntopics, len(docs.vocabulary)))
        nterm_by_topic       = numpy.zeros((ntopics                      ))
        z_doc_word           = numpy.zeros((len(docs.documents), len(docs.vocabulary) ))

        
        model_init = [1 / ntopics for x in range(ntopics)]

        def mutate(model_doc, doc_id, word_id):
            ntopic_by_doc_topic[doc_id][z_doc_word[doc_id][word_id]]   -= 1
            ntopic_by_doc      [doc_id]                                -= 1
            nterm_by_topic_term[z_doc_word[doc_id][word_id]][word_id]  -= 1
            nterm_by_topic     [z_doc_word[doc_id][word_id]]           -= 1

            param = [(nterm_by_topic_term[word_id][topic] + beta) / (nterm_by_topic[topic]  + beta * len(docs.vocabulary)) * \
                     (ntopic_by_doc_topic[doc_id][topic] + alpha) / (ntopic_by_doc[doc_id] + alpha * ntopics)                     for topic in range(ntopics)] 
            
            ntopic_by_doc_topic[doc_id][z_doc_word[doc_id][word_id]]   += 1
            ntopic_by_doc      [doc_id]                                += 1
            nterm_by_topic_term[z_doc_word[doc_id][word_id]][word_id]  += 1
            nterm_by_topic     [z_doc_word[doc_id][word_id]]           += 1

            return param 
            
        
        for i in range(niters):
            for doc_id, doc in enumerate(docs.documents):
                sample_topic_by_word = [numpy.random.multinomial(1, mutate(model_init, doc_id, word_id)) for word_id in doc.wordtoterm() ]
                
                for i in sample_topic_by_word:
                    ntopic_by_doc_topic[doc_id][sample_topic_by_word[i]]              += 1
                    ntopic_by_doc      [doc_id]                                       += 1
                    nterm_by_topic_term[sample_topic_by_word[i]][doc.wordtoterm()[i]] += 1
                    nterm_by_topic     [sample_topic_by_word[i]]                      += 1
                    z_doc_word[doc_id][doc.wordtoterm()[i]]  = sample_topic_by_word[i]

        print "fini"
        
        
    except Usage, err:
        print >>sys.stderr, err.msg
        print >>sys.stderr, "for help use --help"
        return 2


if __name__ == "__main__":
    sys.exit(main())