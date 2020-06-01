def Word2Int(labels):
    for i, label in enumerate(labels):
        if label == 'propaganda':
            labels[i] = 1
        else:
            labels[i] = 0
    return labels


def Int2Word(i):
    if i == 0:
        return 'non-propaganda'
    elif i == 1:
        return 'propaganda'
    else:
        return 'ERROR binaryTranslation_Int2Word function was passed value: {0}'.format(i)