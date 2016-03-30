
if __name__ == '__main__':
    path = 'dat/subj.txt'
    f = open(path, 'r')
    positive = open('dat/postive.yml', 'w')
    negative = open('dat/negative.yml', 'w')
    neutral = open('dat/neutral.yml', 'w')

    for line in f:
        split = line.split(' ')
        typ = split[0]
        word = split[2]
        polar = split[5]
        if 'type=' in typ and 'word1=' in word and 'priorpolarity=' in polar:
            typ = typ[5:]
            if typ == 'strongsubj': typ = 'strong'
            else:                   typ = 'weak'
            word = word[6:]
            polar = polar[14:-1]
            
            if polar == 'positive':
                positive.write(str(word) + ": [positive, " + str(typ) + "]\n")
            elif polar == 'negative':
                negative.write(str(word) + ": [negative, " + str(typ) + "]\n")
            elif polar == 'neutral':
                neutral.write(str(word) + ": [neutral]\n")
    positive.close()
    negative.close()
    neutral.close()
    f.close()
