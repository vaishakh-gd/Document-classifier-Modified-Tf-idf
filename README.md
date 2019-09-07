# Document-classifier-Modified-Tf-idf
Attempted to classify documents using a modified method of tf-idf

this methodology considers the relative distance betweem the words.(number of words between 2 words as ) also as a parameter to classify documents.
(this approach is also  explored  using Convoluted neural networks in Text classification )

TF(t) = (Number of times term t appears in a document) / (Total number of terms in the document).

IDF(t) = log_e(Total number of documents / Number of documents with term t in it)

in this approach we attempt to introduce this formula

Let no= number_of_words_between_word_and_previous_after_normalisation

Summation((1/no)*Tf((+),word)*Tf((+),previous)-1/no*Tf((-),word)*Tf((-),previous)) (after extensively removing the stop words)

This is computed for every paragraph.(as number of words between the entire text isnt accounted for)

 based on the turn out of the number(whether it's positive or negative paragraph) the paragraph classification is obtained. After that the paragraph number, multiplied by idf is used for
 
 final classification.
 
 
Improvisation is possible for higher accuracy.





