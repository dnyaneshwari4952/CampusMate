import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from src.config import MAX_INPUT_LENGTH

# Dictionary of common contractions to expand
CONTRACTIONS = {
    "can't": "cannot",
    "won't": "will not",
    "n't": " not",
    "i'm": "i am",
    "i've": "i have",
    "i'll": "i will",
    "i'd": "i would",
    "he's": "he is",
    "she's": "she is",
    "it's": "it is",
    "that's": "that is",
    "who's": "who is",
    "what's": "what is",
    "where's": "where is",
    "when's": "when is",
    "why's": "why is",
    "how's": "how is",
    "we're": "we are",
    "they're": "they are",
    "you're": "you are",
    "you've": "you have",
    "you'll": "you will",
    "you'd": "you would",
    "here's": "here is",
}

# Common chat abbreviations and student acronyms
CHAT_ABBREVIATIONS = {
    "r": "are",
    "u": "you",
    "ur": "your",
    "pls": "please",
    "plz": "please",
    "abt": "about",
    "bro": "",
    "dude": "",
    "sir": "",
    "cse": "computer science",
    "cs": "computer science",
    "docs": "documents",
    "doc": "document",
    "clg": "college",
    "univ": "university",
    "dept": "department",
}

# Domain spelling corrections
SPELLING_CORRECTIONS = {
    r"\badmision\b": "admission",
    r"\badmisson\b": "admission",
    r"\badmisssion\b": "admission",
    r"\baddmission\b": "admission",
    r"\baddmisssion\b": "admission",
    r"\bunivercity\b": "university",
    r"\buniversaty\b": "university",
    r"\bunversity\b": "university",
    r"\bplacment\b": "placement",
    r"\bplacemnt\b": "placement",
    r"\blibary\b": "library",
    r"\blibrry\b": "library",
    r"\blibry\b": "library",
    r"\bhostal\b": "hostel",
    r"\bhosstel\b": "hostel",
    r"\bdocumnts\b": "document",
    r"\bdocumant\b": "document",
    r"\bdocumants\b": "document",
    r"\bscholrship\b": "scholarship",
    r"\bscholarshp\b": "scholarship",
    r"\bfeess\b": "fees",
    r"\bfess\b": "fees",
    r"\bcours\b": "course",
    r"\bcoures\b": "course",
    r"\bwich\b": "which",
    r"\bavalable\b": "available",
    r"\bdetals\b": "details",
    r"\bprograming\b": "programming",
    r"\bcurriculam\b": "curriculum",
    r"\bcomputr\b": "computer",
}

# Domain synonyms normalization
DOMAIN_SYNONYMS = {
    r"\buni\b": "university",
    r"\bcollege\b": "university",
    r"\bprogramme\b": "program",
    r"\bprograms\b": "program",
    r"\bprogrammes\b": "program",
    r"\binfo\b": "information",
}

_nltk_resources_initialized = False

def download_nltk_resources():
    """
    Safely checks and initializes required NLTK resources quietly.
    """
    global _nltk_resources_initialized
    if _nltk_resources_initialized:
        return

    resources = {
        "punkt": "tokenizers/punkt",
        "punkt_tab": "tokenizers/punkt_tab",
        "wordnet": "corpora/wordnet",
        "stopwords": "corpora/stopwords"
    }

    for name, path in resources.items():
        try:
            nltk.data.find(path)
        except LookupError:
            try:
                nltk.download(name, quiet=True)
            except Exception:
                pass

    _nltk_resources_initialized = True

def clean_text(text: str) -> str:
    """
    Performs initial deterministic cleaning: length checking, lowercase conversion,
    URL/HTML removal, contraction/abbreviation expansion, repeated letter normalization,
    and punctuation handling.
    """
    if not isinstance(text, str):
        return ""

    if len(text) > MAX_INPUT_LENGTH:
        raise ValueError(f"Input text exceeds maximum length of {MAX_INPUT_LENGTH} characters.")

    # 1. Lowercase
    text = text.lower()

    # 2. URL removal
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)

    # 3. HTML tag removal
    text = re.sub(r"<.*?>", " ", text)

    # 4. Expand contractions
    for contraction, expansion in sorted(CONTRACTIONS.items(), key=lambda x: len(x[0]), reverse=True):
        text = re.sub(r"\b" + re.escape(contraction) + r"\b", expansion, text)

    # 5. Expand chat abbreviations
    for abbr, expansion in CHAT_ABBREVIATIONS.items():
        text = re.sub(r"\b" + re.escape(abbr) + r"\b", expansion, text)

    # 6. Apply spelling corrections
    for typo_pattern, correction in SPELLING_CORRECTIONS.items():
        text = re.sub(typo_pattern, correction, text)

    # 7. Normalize repeated characters (3 or more: 'heyyyy' -> 'hey', 'hellooo' -> 'hello')
    text = re.sub(r"(.)\1{2,}", r"\1", text)

    # 8. Normalize punctuation to spaces
    text = re.sub(r"[^a-zA-Z0-9\s]", " ", text)

    # 9. Clean duplicate whitespace
    text = " ".join(text.split())

    return text

def preprocess_text(text: str) -> str:
    """
    Executes full deterministic NLP preprocessing pipeline:
    clean_text -> domain normalization -> tokenization -> stopword filtering -> lemmatization.

    Returns space-separated processed tokens.
    """
    download_nltk_resources()

    cleaned = clean_text(text)
    if not cleaned:
        return ""

    # Apply domain synonym standardizations
    for term_pattern, norm in DOMAIN_SYNONYMS.items():
        cleaned = re.sub(term_pattern, norm, cleaned)

    # Tokenize
    try:
        tokens = word_tokenize(cleaned)
    except Exception:
        tokens = cleaned.split()

    # Stopword management
    try:
        stop_words = set(stopwords.words("english"))
    except Exception:
        stop_words = set()

    # Preserve question and negation keywords for intent discernment
    preserved_keywords = {
        "what", "when", "where", "why", "how", "who", "which",
        "not", "no", "cannot", "never"
    }
    filtered_stopwords = stop_words - preserved_keywords

    # Lemmatize tokens
    lemmatizer = WordNetLemmatizer()
    processed_tokens = []

    for token in tokens:
        if token not in filtered_stopwords:
            try:
                lemma = lemmatizer.lemmatize(token, pos='n')
                lemma = lemmatizer.lemmatize(lemma, pos='v')
            except Exception:
                lemma = token
            processed_tokens.append(lemma)

    return " ".join(processed_tokens)
