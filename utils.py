# Helper Functions adapted from Medium author: Ted Mei 
## Source: https://ted-mei.medium.com/demystify-tf-idf-in-indexing-and-ranking-5c3ae88c3fa0
import numpy as np
from numpy import linalg as LA
import spacy
import scattertext as st
import pandas as pd
import altair as alt
import math

import warnings
warnings.filterwarnings("ignore", category=pd.errors.SettingWithCopyWarning)

dataminingdf = pd.read_csv('../data/mlarticles.csv')

sp = spacy.load('en_core_web_sm')

def lemmatize(instring,title="",lemmaCache = {}):
    parsed = None
    
    if ((title != "") & (title in lemmaCache)):
        parsed = lemmaCache[title]    
    else:
        parsed = sp(instring)

    if (lemmaCache != None):
        lemmaCache[title] = parsed
    sent = [x.text if (x.lemma_ == "-PRON-") else x.lemma_ for x in parsed]
    return(sent)

# compute Term frequency of a specific term in a document
def termFrequency(term, document):
    normalizeDocument = document.lower().split()
    return normalizeDocument.count(term.lower()) / float(len(normalizeDocument))

# IDF of a term
def inverseDocumentFrequency(term, documents):
    count = 0
    for doc in documents:
        if term.lower() in doc.lower().split():
            count += 1
    if count > 0:
        return 1.0 + math.log(float(len(documents))/count)
    else:
        return 1.0
    
    
# tf-idf of a term in a document
def tf_idf(term, document, documents):
    tf = termFrequency(term, document)
    idf = inverseDocumentFrequency(term, documents)
    return tf*idf

def generateVectors(query, documents):
    tf_idf_matrix = np.zeros((len(query.split()), len(documents)))
    for i, s in enumerate(query.lower().split()):
        idf = inverseDocumentFrequency(s, documents)
        for j,doc in enumerate(documents):
            tf_idf_matrix[i][j] = idf * termFrequency(s, doc)
    return tf_idf_matrix
   
def word_count(s):
    counts = dict()
    words = s.lower().split()
    for word in words:
        if word in counts:
            counts[word] += 1
        else:
            counts[word] = 1
    return counts

def build_query_vector(query, documents):
    count = word_count(query)
    vector = np.zeros((len(count),1))
    for i, word in enumerate(query.lower().split()):
        vector[i] = float(count[word])/len(count) * inverseDocumentFrequency(word, documents)
    return vector
    

def consine_similarity(v1, v2):
        return np.dot(v1,v2)/float(LA.norm(v1)*LA.norm(v2))

def compute_relevance(query, documents, tf_idf_matrix,query_vector):
    results = []
    # i does not mean document id.
    for i, doc in enumerate(documents):
        #print(i, doc)
        similarity = consine_similarity(tf_idf_matrix[:,i].reshape(1, len(tf_idf_matrix)), query_vector)
        if np.isnan(similarity[0]):                                        
            results.append((i,0,doc))
        else:
            results.append((i,float(similarity[0]),doc))                       
                    
        #print("query document {}, similarity {}".format(i, float(similarity[0])))
    results = sorted(results, key = lambda x : x[1], reverse = True)

    return results        
  
def drawTilebars(query,normalized=False,sortby='title'):
    # this function takes 
    # query: a string query
    # normalized: an argument about whether to normalize the tilebar (True or False)
    #   if false, the the color of the tile should map to the count
    #   if true, you should decide how you want to normalize (by the max count overall? max count in article?)
    # sortby: a string of either "title" or "score"
    #   if title, the tilebars should be returned based on alphabetical order of the articles
    #   if score, you can decide how you want to rank the articles
    # the function returns: an altair chart
    
    terms = lemmatize(query)
    lem_query = " ".join(terms)
   
    
    print("the lemmatized query terms are: ", terms)
    print("nomalized is ",normalized)
    print("I will sort by", sortby)
    

    ### Pre-processing ### 
    df_target = dataminingdf.copy().drop(dataminingdf.index)
    dataminingdf_cp = dataminingdf.copy()
    
    # originally wanting to lemmatize the entire text to help with the counts, but takes too long to load.
    #dataminingdf_cp['text'] = dataminingdf_cp['text'].apply(lemmatize).apply(lambda x: " ".join(x))
    
    # get documents
    documents = dataminingdf['title'].unique()
    
    # generate counts
    for document in documents:
        for term in terms:
            df_temp = dataminingdf_cp[dataminingdf_cp['title'] == document]
            df_temp['term'] = term
            df_temp['count'] = df_temp['text'].str.count(' ' + term + ' ')
            df_target = pd.concat([df_target, df_temp])

    # filter on documents where relevant words are found
    drop_titles_df = df_target.groupby('title')['count'].sum()    
    inscope = drop_titles_df[drop_titles_df > 0].index
    mask = df_target.title.isin(inscope)
   
    # sort by parameter
    if sortby == 'title':
        documents = sorted(documents.tolist())
        sort_param = None

    else: 
        # Apply scoring algorithm tf-idf

        df_ranking = dataminingdf_cp.copy() # if want do a full-scan

        mask2 = df_ranking.title.isin(inscope)
        df_ranking = df_ranking[mask2]

        df_ranking['full_text'] = df_ranking[['docid','text','title']].groupby(['docid'])['text'].transform(lambda x: ' '.join(x))    
        df_ranking = df_ranking[['docid','title','full_text']].drop_duplicates()
        documents_text =  list(df_ranking['full_text'])
        
        # a list of tuples and documentid
        #documents_text =  list(zip(df_ranking.docid,df_ranking.full_text))
        
        temp = df_ranking[['docid','title','full_text']].to_dict('split')
    
        # key is text, value is title
        mapper = {}
        for i in temp['data']:
            mapper[i[2]] = i[1]
   

        # Perform tf_idf and ranking
        tf_idf_matrix = generateVectors(lem_query, documents_text)
        query_vector = build_query_vector(lem_query, documents_text)
        order = compute_relevance(lem_query, documents_text, tf_idf_matrix,query_vector)
        rank_by_document = [i[2] for i in order]
        
        # rank_by_title
        sort_param = [mapper[key] for key in rank_by_document]


    # Apply normalization using z scoring
      # For each term/observation - the average expected term and divide by standard deviation
    if normalized:
        mean = df_target['count'].mean()
        sd = df_target['count'].std()
        df_target['count'] = (df_target['count'] - mean)/ sd
        scheme_option = "blues"
    
    else:
        scheme_option = 'orangered'
        
    

    ### Build Chart ### 
    end = max(df_target['lineid'])
    domainx = [i for i in range(end+1)]
    

    # Building the chart: 
    
    # data source is filtered on relevant documents
    chart = alt.Chart(df_target[(mask) & df_target['count'] > 0]).mark_rect().encode(
                                                x = alt.X('lineid:O', scale = alt.Scale(domain = domainx), title = ""),
                                                y = alt.Y('term:N', title = ""),
                                                color = alt.Color('count:Q', legend = None, scale = alt.Scale(scheme = scheme_option)),
                                                tooltip = ['term','count','lineid']

                                            ).facet(facet = alt.Facet('title:N' , sort = sort_param, title = None, header = alt.Header(labelAnchor = 'middle', labelBaseline = 'top', labelOrient = 'right', labelAngle = 0, labelPadding = 10)), columns = 1) 

    return chart

    