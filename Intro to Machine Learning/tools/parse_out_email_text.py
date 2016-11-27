#!/usr/bin/python

from nltk.stem.snowball import SnowballStemmer
import string

def parseOutText(f):
    """ given an opened email file f, parse out all text below the
        metadata block at the top
        (in Part 2, you will also add stemming capabilities)
        and return a string that contains all the words
        in the email (space-separated) 
        
        example use case:
        f = open("email_file_name.txt", "r")
        text = parseOutText(f)
        
        """


    f.seek(0)  ### go back to beginning of file (annoying)
    all_text = f.read()

    ### split off metadata
    content = all_text.split("X-FileName:")
    words = ""
    if len(content) > 1:
        ### remove punctuation
        text_string = content[1].translate(string.maketrans("", ""), string.punctuation)
##        print("Initial text =============")
##        print(text_string)
        ### project part 2: comment out the line below

        #words = text_string

        ### split the text string into individual words, stem each word,
        ### and append the stemmed word to words (make sure there's a single
        ### space between each stemmed word)
        
        from nltk.stem.snowball import SnowballStemmer
        stemmer = SnowballStemmer("english")
        #print(text_string)

        split_texts = text_string.split(" ")
##        print("Text after split =============")
##        print(split_texts)
        #print(split_test)
        #stemmed_texts = [stemmer.stem(split_text) for split_text in split_texts]

        for elt in split_texts:
##            print("'{}' ==> '{}'".format(elt, stemmer.stem(elt)))
            if elt == "\n" or elt == "\\n" or elt == " " or elt == "\\n\n":
                split_texts.remove(elt)

            
        for elt in split_texts:
##            print("'{}' ==> '{}'".format(elt, stemmer.stem(elt)))
            elt = elt.replace("\n", " ")
            elt = elt.replace("\\n", " ")
            elt = elt.replace("\n\n", " ")
            elt = elt.replace("  ", " ")
            words += stemmer.stem(elt)+ " "
##            if elt != "\\n" and elt != " " and elt != "\\n\n":
##                words += stemmer.stem(elt)+ " "
        """
        test = stemmer.stem("Everyone")
        print("test: ", test)
        print("debut =============")
        print(text_string)
        print("fin =============")
        """



    return words

    

def main():
    ff = open("../text_learning/test_email.txt", "r")
    text = parseOutText(ff)
##    print("debut =============")
##    print text
##    print("fin =============")



if __name__ == '__main__':
    main()

