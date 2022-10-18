import pymongo
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

client = pymongo.MongoClient("mongodb+srv://fbfighter:fbfighter@fb-topic.ixbkp2u.mongodb.net/?retryWrites=true&w=majority")

db = client["fakenews"]

col = db["fbpost"]

data = col.find()

corpus = []

# corpus = ['The sun is the largest celestial body in the solar system',
#           'The solar system consists of the sun and eight revolving planets',
#           'Ra was the Egyptian Sun God',
#           'The Pyramids were the pinnacle of Egyptian architecture',
#           'The quick brown fox jumps over the lazy dog']

for x in data[:5]:
    corpus.append(x['text'])

# tfidf_vectorizer = TfidfVectorizer()
#
# tfidf_matrix = tfidf_vectorizer.fit_transform(corpus)
#
# cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

