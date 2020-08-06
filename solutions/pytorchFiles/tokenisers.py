import spacy

def defaultSpacyTokeniser(string):
    if not globals().get('spacyTokens'):
        globals()['spacyTokens'] = spacy.load('en', disable=['parser', 'tagger', 'ner'])
    
    global spacyTokens

    return [word.text for word in spacyTokens(string)]

