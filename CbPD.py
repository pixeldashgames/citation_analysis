# Importing necessary libraries
import spacy
from spacy.tokens import Doc

# Loading the English language model from spaCy
nlp = spacy.load('en_core_web_sm')


# Function to clean the input text
def tokenize_text(text: str) -> Doc:
    # Creating a Doc object
    return nlp(text)

# Function to extract citations from the text
def extract_citations(doc: Doc)->str:
    """
    This function takes a Doc object as input and returns a string of citations.
    The citations are extracted from the entities in the Doc object that are either works of art or organizations.
    Each citation is separated by a comma and a space.
    """
    # Extracting entities that are either works of art or organizations
    citations = [ent.text for ent in doc.ents if ent.label_ in ['WORK_OF_ART', 'ORG']]
    if len(citations)==0:
        return ""
    # Removing citations that are too short
    citations = [citation for citation in citations if len(citation.split()) > 2]
    # Joining the citations into a single string
    citations_str = ', '.join(citations)
    return citations_str


# Function to calculate the similarity between two texts
def calculate_similarity(text1:str, text2:str)->float:
    """
    This function takes two strings as input and returns the cosine similarity between them.
    The similarity is calculated using the spaCy library.
    """
    # Creating Doc objects for the input texts
    doc1 = nlp(text1.lower())
    doc2 = nlp(text2.lower())
    # Calculating the similarity between the inferred vectors
    return doc1.similarity(doc2)


# Function to detect plagiarism between two texts
def detect_plagiarism(text1:str, text2:str):
    '''
    This is the main function of this module, it takes two strings as an input and return the cosine similarity bewteen
    the citations of each one using the spaCy library
    '''
    doc1 = tokenize_text(text1)
    doc2 = tokenize_text(text2)
    # Extracting citations from the cleaned texts
    citations1 = extract_citations(doc1)
    citations2 = extract_citations(doc2)
    if len(citations1)==0 or len(citations2)==0:
        return 0
    # Calculating the similarity between the citations
    citation_similarity = calculate_similarity(citations1,citations2)
    # Returning the similarity score
    return citation_similarity


import PyPDF2
import sys
if __name__ == "__main__":
    if len(sys.argv)!=3:
        print("Usage: python script.py pdf1.pdf pdf2.pdf")
    # Open the first PDF file in read-binary mode
    with open(sys.argv[1], 'rb') as file:
        # Create a PDF file reader object
        reader = PyPDF2.PdfReader(file)
        # Read the content of the PDF file and remove newline characters and other special symbols
        text1 = ' '.join([reader.pages[i].extract_text()
                         .replace('\n', ' ')
                         .replace('\r', ' ')
                          for i in range(len(reader.pages))])

    # Open the second PDF file in read-binary mode
    with open(sys.argv[2], 'rb') as file:
        # Create a PDF file reader object
        reader = PyPDF2.PdfReader(file)
        # Read the content of the PDF file and remove newline characters and other special symbols
        text2 = ' '.join([reader.pages[i].extract_text()
                         .replace('\n', ' ')
                         .replace('\r', ' ')
                          for i in range(len(reader.pages))])
    print(detect_plagiarism(tokenize_text(text1), tokenize_text(text2)))