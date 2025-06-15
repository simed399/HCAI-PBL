import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

nltk.download('stopwords')
nltk.download('wordnet')

STOP = set(stopwords.words('english'))
LEMMA = WordNetLemmatizer()

def clean_text(text):
    text = text.lower()
    text = re.sub(r'<.*?>', ' ', text)
    text = re.sub(r'[^a-z\s]', ' ', text)
    tokens = [LEMMA.lemmatize(tok) for tok in text.split() if tok not in STOP]
    return ' '.join(tokens)
