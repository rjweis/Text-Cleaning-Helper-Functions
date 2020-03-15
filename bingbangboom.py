#!/usr/bin/env python
# coding: utf-8

# # Text Cleaning Function

# In[79]:


def clean_docs(doc_lst = [], remove_lst = [], replace_dict = {},
                   tokenize_return = False, stem_docs = False, lemmatize_docs = True):
    ''' Cleans an array of documents according to the arguments specified by the user.
    
    Arguments:
        doc_lst: Series of documents.
        tokenize_return: If TRUE, return doc_array as tokens. 
        stem: If TRUE, stem the doc_array.
        lemmatize: If TRUE, lemmative the doc_array. 
        remove_lst: List of words to be removed from the doc_array, in addition
            to those included in nltk's English stopwords list.
        replace_dict: Dictionary of words to replace in the doc_array.
        
    Returns:
        doc_array: List of freshly cleaned documents.
        
    Step-by-step overview:
        1. Make words lower case
        2. Round 1 of removing stop words.
        3.
        
    Notes:
        doc_lst MUST be either a pandas.Series() or a column of a pandas.DataFrame.
        The code will not execute otherwise.
        
        Ensure there are no null values in the doc_lst. The code will not run otherwise.
        
        remove_lst should include 'xxx', and maybe even numbers ('one', 'two', etc.)
    '''

    #######################
    #######################
    #######################
    
    # define function for removing stopwords
    def remove_stopwords(doc_lst, remove_lst = []):
        import nltk
        from nltk.tokenize import word_tokenize

        stopwords = nltk.corpus.stopwords.words('english') + remove_lst

        n = 0
        for doc in doc_lst:
            tokens = word_tokenize(doc)
            clean_doc = ' '.join(token for token in tokens if token not in stopwords)
            doc_lst[n] = clean_doc
            n += 1

        return(doc_lst)
    
    #######################
    #######################
    #######################
    
    # define function for replacing words
    def word_list_replace(doc_lst, replace_dict):
        import re
        import nltk 
        
        for key in replace_dict:
            regex_words = re.compile(r'\b%s\b' % r'\b|\b'.join(map(re.escape, [key])))
            
            n = 0
            for doc in doc_lst:
                new_words = regex_words.sub(replace_dict[key], doc)
                doc_lst[n] = new_words
                n += 1
        
        return(doc_lst)
        
    #######################
    #######################
    #######################
    
    # define function for lemmatizing words
    def lemmatize(doc_lst):
        from nltk.stem import WordNetLemmatizer
        from nltk.tokenize import word_tokenize
    
        lemmatizer = WordNetLemmatizer()
    
        n = 0
        for doc in doc_lst:
            tokens = word_tokenize(doc)
            lemmatized_words = [lemmatizer.lemmatize(token) for token in tokens]
            lemmatized_doc = ' '.join(lemmatized_words)
            doc_lst[n] = lemmatized_doc
            n += 1
    
        return(doc_lst)

    #######################
    #######################
    #######################
    
    # define function for stemming words
    def stem(doc_lst):
        import nltk
        porter = nltk.stem.porter.PorterStemmer()
        from nltk.tokenize import word_tokenize
    
        n = 0
        for doc in doc_lst:
            tokens = word_tokenize(doc)
            stemmed_words = [porter.stem(token) for token in tokens]
            stemmed_doc = ' '.join(stemmed_words)
            doc_lst[n] = stemmed_doc
            n += 1
    
        return(doc_lst)

    #######################
    #######################
    #######################

    # verify type of doc_lst
    if str(doc_lst.__class__) != "<class 'pandas.core.series.Series'>":
        return('Error: Document list must be pandas.core.series.Series!')
    
    # check for null values
    nulls = doc_lst.isnull().sum()
    if nulls > 0:
        return('Error: Document list contains null values. Remove these and try again.')
    
    # ensure index doesn't have any missing gaps (e.g., from failing to reset the index after dropping null values)
    if doc_lst.index[-1] + 1 < len(doc_lst):
        return('Error: Index is inconsistently numbered and contains gaps. If you have recently dropped null values, try resetting the index.')
    
    # import modules
    
    import re
    import string
    import nltk
    from nltk import word_tokenize
        
    # make all text lowercase
    doc_lst = doc_lst.str.lower()
    
    # remove stopwords
    doc_lst = remove_stopwords(doc_lst, remove_lst = remove_lst)
    
    # remove all punctuation
    doc_lst = doc_lst.str.replace('[^\w\s]', '')
    
    # remove stopwords
    doc_lst = remove_stopwords(doc_lst, remove_lst = remove_lst)
    
    # if some documents are solely stopwords, then they will be NA and need to be dropped
    # doc_lst = doc_lst.dropna()
    # doc_lst = doc_lst.reset_index(drop = True)
    
    # replace words
    if replace_dict != {}:
        doc_lst = word_list_replace(doc_lst, replace_dict = replace_dict)
    
    # stem words if specified by the user
    if stem_docs is True:
        doc_lst = stem(doc_lst)
    
    # lemmatize the words (default for lemmatize_docs = True)
    else:
        doc_lst = lemmatize(doc_lst)
        
    # return tokens from document if specified by the user
    if tokenize_return is True:
        doc_lst = doc_lst.apply(lambda x: word_tokenize(x))
    
    # remove stopwords (round 2)
    doc_lst = remove_stopwords(doc_lst, remove_lst = remove_lst)
    
    return(doc_lst)


# In[60]:


def word_list_replace(doc_lst, replace_dict):
    import re
    import nltk 
    
    for key in replace_dict:
        regex_words = re.compile(r'\b%s\b' % r'\b|\b'.join(map(re.escape, [key])))
        
        n = 0
        for doc in doc_lst:
            new_words = regex_words.sub(replace_dict[key], doc)
            doc_lst[n] = new_words
            n += 1
    
    return(doc_lst)


# In[61]:


def remove_punc(doc_lst):
    import string
    
    punct = string.punctuation
    
    clean_docs = (''.join(letter for letter in word if letter 
                         not in punct) for word in doc_lst)

    return(clean_docs)


# In[62]:


def remove_stopwords(doc_lst, remove_lst = []):
    import nltk
    from nltk.tokenize import word_tokenize

    stopwords = nltk.corpus.stopwords.words('english') + remove_lst

    n = 0
    for doc in doc_lst:
        tokens = word_tokenize(doc)
        clean_doc = ' '.join(token for token in tokens if token not in stopwords)
        doc_lst[n] = clean_doc
        n += 1

    return(doc_lst)


# In[63]:


def stem(doc_lst):
    import nltk
    porter = nltk.stem.porter.PorterStemmer()
    from nltk.tokenize import word_tokenize
    
    n = 0
    for doc in doc_lst:
        tokens = word_tokenize(doc)
        stemmed_words = [porter.stem(token) for token in tokens]
        stemmed_doc = ' '.join(stemmed_words)
        doc_lst[n] = stemmed_doc
        n += 1
    
    return(doc_lst)


# In[64]:


def lemmatize(doc_lst):
    from nltk.stem import WordNetLemmatizer
    from nltk.tokenize import word_tokenize

    lemmatizer = WordNetLemmatizer()

    n = 0
    for doc in doc_lst:
        tokens = word_tokenize(doc)
        lemmatized_words = [lemmatizer.lemmatize(token) for token in tokens]
        lemmatized_doc = ' '.join(lemmatized_words)
        doc_lst[n] = lemmatized_doc
        n += 1

    return(doc_lst)


# In[65]:


def ngram_counter(doc_lst, ngram_size):
    from nltk.util import ngrams
    from nltk.tokenize import word_tokenize
    
    # generate tokens for each doc
    tokens = [word_tokenize(doc) for doc in doc_lst]
    
    # create ngrams (n=ngram_size) for each sublist in tokens, where sublist is tokens[i].
    # i.e., create ngrams for each document
    grams = [list(ngrams(token, ngram_size)) for token in tokens]  
    
    # create list of unique ngrams
    unique_ngrams = list(set([item for sublist in grams for item in sublist]))
                    # [item for sublist in grams for item in sublist] --> turns list of lists into a single list
                    # set[] --> gets unique ngrams from list
                    # list() --> turns set into a subscriptable list so you can do something like print(unique_ngrams[0])                  
    
    # count frequency of every ngram of a unique_ngrams in an array of documents (represented by array of grams)
    cnt_list = []
    for ngram in unique_ngrams:
        cnt = 0
        for gram in grams: # looping through all ngrams, not just the unique set
            cnt += gram.count(ngram) 
        cnt_list.append(cnt)
        
    output = list(zip(unique_ngrams, cnt_list))
    
    return(output)


# In[ ]:

def remove_lst():
    '''Returns a list of the words that are currently being removed from the transcripts in our cleaning process.'''
    
    words2remove = [
        'nt', 'mhm', 'uh', 'uhm', 'um', 'one', 'two', 'three', 'four', 'five',
        'six', 'seven', 'eight', 'nine', 'hm', 'oh', 'h', 'hi', 'wh', 'bye',
        'byebye', 'n', 'g', 'okay', 'hello', 'yes', 'p', 'u', 'yeah', 'f',
        'hey', 'ah', 'xxxnd', 'k', 'kn', 'wa', 'na', 'huh', 'uhhuh', 'j',
        'xxx', 'xxxth', 'xxxrd', 'xxxst', 'im', 'ah', 'uh', 'hm', 'um', 'oh',
        'mhm', 'okay', 'hello', 'hi', 'xxx', 'thank', 'pleas', 'get', 'got',
        'your', 'eh', 'would', 'ye', 'that', 'want', 'theyv', 'name', 'wh',
        'wet', 'rais', 'bro', 'im', 'id', 'gon', 'na', 'go', 'there',
        'yeah', 'uhm', 'much'
    ]
    
    return(words2remove)

###


def repl_dict():
    '''
    Returns a dictionary of word combinations that are being replaced in our cleaning process.
    
        Key: Word to replace
        Value: Word that replaces the Key
    '''

    replace_dict = {
        'eye pad': 'ipad',
        'my pad': 'ipad',
        'i pad': 'ipad',
        'green': 'screen',
        'play': 'pay',
        'played': 'paid',
        'playing': 'paying',
        'tech': 'technical',
        'sherrion': 'asurion',
        'assyrian': 'asurion',
        'agent': 'representative',
        'blue tooth': 'bluetooth',
        'plane': 'plan',
        'check check': 'check',
        'make make': 'make',
        'phone phone': 'phone',
        'know know': 'know',
        'representative representative': 'representative',
        'hi pad': 'ipad',
        'hi pads': 'ipad',
        'hipad': 'ipad',
        'clean': 'claim',
        'assurant': 'asurion',
        '  ': ' ',  # replace double spaces with single spaces
        'cellphone': 'cell phone',
        'miss placed': 'misplaced', 
        'correct screen': 'cracked screen',
        'correct scream': 'correct scream',
        'scream': 'screen',
    }
    
    return(replace_dict)



