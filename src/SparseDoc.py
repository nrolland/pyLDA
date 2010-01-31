'''
Created on Jan 31, 2010

@author: nrolland
'''
import  collections, operator


def flatten(x):
    """flatten(sequence) -> list

    Returns a single, flat list which contains all elements retrieved
    from the sequence and all recursively contained sub-sequences
    (iterables).

    Examples:
    >>> [1, 2, [3,4], (5,6)]
    [1, 2, [3, 4], (5, 6)]
    >>> flatten([[[1,2,3], (42,None)], [4,5], [6], 7, MyVector(8,9,10)])
    [1, 2, 3, 42, None, 4, 5, 6, 7, 8, 9, 10]"""

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
        name : common part of the name e.g. xxx for  docword.xxx and vocab.xxx
        '''
        
        self.vocabulary = file(directory+"/vocab." + commonfilename).readlines()
        docfile  = file(directory+"/docword." + commonfilename)
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
            if last != doc_id and  doc_id - round(doc_id /1000, 0) *1000 == 0:
                print str((doc_id *100) / self.M)  + "%"
                last = doc_id
            #print doc.keys()
            doc[word_id] +=1
            