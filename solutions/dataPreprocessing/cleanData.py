from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer, word_tokenize
from nltk.stem.lancaster import LancasterStemmer
from nltk.stem.porter import PorterStemmer



def cleanText(text, stemmingMethod=None):
    stopWords = stopwords.words('english')
    punctuationTokeniser = RegexpTokenizer(r'\w+')
    cleanedText = punctuationTokeniser.tokenize(text)
    cleanedText = [word for word in cleanedText if not word.lower() in stopWords]
    cleanedText = stemText(cleanedText, stemmingMethod)    
    return ' '.join(cleanedText)


def stemText(cleanedText, stemmingMethod):
    global lastError
    try: lastError
    except NameError: lastError = None

    if stemmingMethod:
        stemmingMethod = stemmingMethod.lower()
        errorText = '\nERROR: Stemming Method \'{0}\' not recongnised. Acceptable inputs are \'lancaster\', \'l\', \'poter\', or \'p\'.\nExecution has continued without word stemming.\n'.format(stemmingMethod)

        stemmer = None
        if stemmingMethod == 'lancaster' or stemmingMethod == 'l':
            stemmer = LancasterStemmer()
        elif stemmingMethod == 'porter' or stemmingMethod == 'p':
            stemmer = PorterStemmer()

        if stemmer != None:
            for i in range(len(cleanedText)):
                cleanedText[i] = stemmer.stem(cleanedText[i])
        
        elif lastError != errorText:
            lastError = errorText
            print(errorText)
        
    return cleanedText