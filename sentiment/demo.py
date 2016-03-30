import sentiment

if __name__ == '__main__':
    files = ['dat/positive.yml', 'dat/negative.yml', 'dat/dec.yml', 'dat/inc.yml', 'dat/neutral.yml']    
    d = sentiment.gen_dict(files)
    
    s1 = """Earth has not anything to show
    more fair: Dull would he be of soul who could pass by A sight 
    so touching in its majesty: This City now doth, like a garment, wear 
    The beauty of the morning; silent, bare, Ships, towers, domes, theatres, and
    temples lie Open unto the fields, and to the sky; All bright and glittering in the 
    smokeless air. Never did sun more beautifully steep In his first splendour, valley, rock,
    or hill; Ne'er saw I, never felt, a calm so deep! The river glideth at his own sweet will: Dear
    God! the very houses seem asleep; And all that mighty heart is lying still!"""
    sentiment.demo_analyze(s1, d)
    print "\n\n\n\n"

    
    s2 = """Eomer. Take your Eored down the left flank. Gamling, follow the King's banner down the center. Grimbold, take your company right, after you pass the wall. Forth, and fear no darkness! Arise! Arise, Riders of Theoden! Spears shall be shaken, shields shall be splintered! A sword day... a red day... ere the sun rises! Ride now! Ride now! Ride! Ride for ruin and the world's ending! Death! Forth Eorlingas!"""
    sentiment.demo_analyze(s2, d)    
