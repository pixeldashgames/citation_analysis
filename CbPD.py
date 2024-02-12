# Importing necessary libraries
import spacy
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from nltk.tokenize import word_tokenize
import nltk
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
    nltk.download('wordnet')
    nltk.download('stopwords')
    nltk.download('averaged_perceptron_tagger')
    

# Loading the English language model from spaCy
nlp = spacy.load('en_core_web_sm')


# Function to clean the input text
def clean_string(text):
    # Creating a Doc object
    doc = nlp(text)
    # Lemmatizing the words, removing stop words and joining them back into a string
    cleaned_text = " ".join(token.lemma_ for token in doc if not token.is_stop)
    return cleaned_text


# Function to extract citations from the text
def extract_citations(text):
    # Creating a Doc object
    doc = nlp(text)
    # Extracting entities that are either works of art or organizations
    citations = [ent.text for ent in doc.ents if ent.label_ in ['WORK_OF_ART', 'ORG']]
    # Removing citations that are too short
    citations = [citation for citation in citations if len(citation.split()) > 2]
    return citations


# Function to calculate the similarity between two texts
def calculate_similarity(text1, text2):
    # Preparing the data for the Doc2Vec model
    tagged_data = [TaggedDocument(words=word_tokenize(_d.lower()), tags=[str(i)]) for i, _d in
                   enumerate([text1, text2])]
    # Training the Doc2Vec model
    model = Doc2Vec(tagged_data, vector_size=20, window=2, min_count=1, epochs=100)
    # Inferring vectors for the input texts
    vector1 = model.infer_vector(word_tokenize(text1.lower()))
    vector2 = model.infer_vector(word_tokenize(text2.lower()))
    # Calculating the similarity between the inferred vectors
    return model.docvecs.similarity_unseen_docs(model, vector1, vector2)


# Function to detect plagiarism between two texts
def detect_plagiarism(text1, text2):
    # Cleaning the input texts
    cleaned_text1 = clean_string(text1)
    cleaned_text2 = clean_string(text2)
    # Extracting citations from the cleaned texts
    citations1 = extract_citations(cleaned_text1)
    citations2 = extract_citations(cleaned_text2)
    # Calculating the similarity between the citations
    citation_similarity = calculate_similarity(' '.join(citations1), ' '.join(citations2))
    # Returning the similarity score
    return citation_similarity


import PyPDF2

if __name__ == "__main__":
    # Open the first PDF file in read-binary mode
    with open('test1.pdf', 'rb') as file:
        # Create a PDF file reader object
        reader = PyPDF2.PdfReader(file)
        # Read the content of the PDF file and remove newline characters and other special symbols
        text1 = ' '.join([reader.pages[i].extract_text()
                         .replace('\n', ' ')
                         .replace('\r', ' ')
                          for i in range(len(reader.pages))])

    # Open the second PDF file in read-binary mode
    with open('test2.pdf', 'rb') as file:
        # Create a PDF file reader object
        reader = PyPDF2.PdfReader(file)
        # Read the content of the PDF file and remove newline characters and other special symbols
        text2 = ' '.join([reader.pages[i].extract_text()
                         .replace('\n', ' ')
                         .replace('\r', ' ')
                          for i in range(len(reader.pages))])
    print(detect_plagiarism(text1, text2))