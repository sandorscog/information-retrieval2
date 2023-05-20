import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import SnowballStemmer
from nltk.corpus import stopwords
import string

nltk.download('stopwords', quiet=True)
nltk.download('punkt', quiet=True)


def pre_process(text: str) -> list:
    snowball = SnowballStemmer(language='english')
    stop_words = set(stopwords.words('english'))
    text = text.lower()

    tokens = word_tokenize(text)
    tokens = [token for token in tokens if token not in string.punctuation]

    clean_tokens = []
    for token in tokens:
        clean_token = ''.join(e for e in token if e.isalnum())
        clean_tokens.append(clean_token)

    tokens = [token for token in clean_tokens if token not in stop_words]
    tokens = [snowball.stem(token) for token in tokens]
    tokens = [
        token.replace('/', '_')
        .replace('`', '')
        .replace("'s", '')
        .replace('s', '')
        .replace("'", '')
        .replace(',', '')
        .replace('#', '')
        .replace('!', '')
        .replace('', '')
        for token in tokens
    ]

    tokens = [token for token in tokens if token != '']

    # print(tokens)
    return tokens


