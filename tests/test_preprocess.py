import pytest
from src.config import MAX_INPUT_LENGTH
from src.preprocess import clean_text, preprocess_text

def test_lowercase_conversion():
    assert clean_text("HELLO WORLD") == "hello world"
    assert clean_text("mIxEd CaSe") == "mixed case"

def test_url_removal():
    assert clean_text("visit http://google.com for info") == "visit for info"
    assert clean_text("check www.example.org now") == "check now"

def test_html_tag_removal():
    assert clean_text("hello <b>world</b>") == "hello world"
    assert clean_text("<p>some text</p>") == "some text"

def test_contraction_expansion():
    assert clean_text("when's the exam?") == "when is the exam"
    assert clean_text("i'm going, you're not") == "i am going you are not"
    assert clean_text("can't do it") == "cannot do it"

def test_chat_abbreviations():
    assert clean_text("what r u doing") == "what are you doing"
    assert clean_text("tell me abt ur hostel pls") == "tell me about your hostel please"

def test_spelling_corrections():
    assert clean_text("admision in univercity") == "admission in university"
    assert clean_text("wich courses and fess") == "which courses and fees"
    assert clean_text("libary and hostal") == "library and hostel"

def test_repeated_characters():
    assert clean_text("heyyyy") == "hey"
    assert clean_text("hellooo") == "hello"
    assert clean_text("good") == "good"
    assert clean_text("sheet") == "sheet"

def test_punctuation_handling():
    assert clean_text("hello!!! how are you?") == "hello how are you"
    assert clean_text("text-with-hyphen") == "text with hyphen"

def test_empty_and_whitespace_edge_cases():
    assert clean_text("") == ""
    assert clean_text("   ") == ""
    assert clean_text("😊😊😊") == ""
    assert clean_text("???!!!") == ""

def test_numbers():
    assert clean_text("class 123 is open") == "class 123 is open"

def test_max_input_length():
    long_string = "a" * (MAX_INPUT_LENGTH + 1)
    with pytest.raises(ValueError):
        clean_text(long_string)

def test_preprocess_text_pipeline():
    processed = preprocess_text("when are the exams scheduled?")
    assert "when" in processed
    assert "exam" in processed
    assert "are" not in processed
    assert "the" not in processed
