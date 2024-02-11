import spacy
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from nltk.tokenize import word_tokenize

nlp = spacy.load('en_core_web_lm')


def clean_string(text):
    doc = nlp(text)
    cleaned_text = " ".join(token.lemma_ for token in doc if not token.is_stop)
    return cleaned_text


def extract_citations(text):
    doc = nlp(text)
    # searching for titles of works and organizations
    citations = [ent.text for ent in doc.ents if ent.label_ in ['WORK_OF_ART', 'ORG']]

    # Post-processing: remove citations that are too short
    citations = [citation for citation in citations if len(citation.split()) > 2]
    return citations


def calculate_similarity(text1, text2):
    tagged_data = [TaggedDocument(words=word_tokenize(_d.lower()), tags=[str(i)]) for i, _d in enumerate([text1, text2])]
    model = Doc2Vec(tagged_data, vector_size=20, window=2, min_count=1, epochs=100)
    vector1 = model.infer_vector(word_tokenize(text1.lower()))
    vector2 = model.infer_vector(word_tokenize(text2.lower()))
    return model.docvecs.similarity_unseen_docs(model, vector1, vector2)


def detect_plagiarism(text1, text2):
    cleaned_text1 = clean_string(text1)
    cleaned_text2 = clean_string(text2)

    citations1 = extract_citations(cleaned_text1)
    citations2 = extract_citations(cleaned_text2)

    citation_similarity = calculate_similarity(' '.join(citations1), ' '.join(citations2))

    return citation_similarity