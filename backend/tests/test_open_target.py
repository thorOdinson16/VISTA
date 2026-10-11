"""Open-target parsing: arbitrary nouns and word-boundary vocabulary match."""

from backend.app.perception.vocabulary import query_terms
from backend.app.retrieval.query_parser import QueryParser, extract_target_noun


def test_unknown_noun_becomes_target():
    plan = QueryParser().parse("Where is my charger?")
    assert plan.query_type == "locate"
    assert plan.target_category == "charger"


def test_color_words_not_part_of_noun():
    assert extract_target_noun("where is the red flask right now") == "flask"


def test_generic_pronouns_are_not_objects():
    assert extract_target_noun("where is everything") is None


def test_vocabulary_match_respects_word_boundaries():
    parser = QueryParser(vocabulary=query_terms(extra_labels=["pen"]))
    assert parser.parse("where is the open door").target_category == "open door"
    assert parser.parse("where is the pen").target_category == "pen"


def test_compound_noun_is_not_collapsed_to_known_term():
    parser = QueryParser(vocabulary=query_terms(extra_labels=["phone"]))
    assert parser.parse("where is my phone charger").target_category == (
        "phone charger"
    )
    assert parser.parse("where is the phone").target_category == "phone"


def test_known_synonym_phrase_still_maps_to_canonical():
    assert QueryParser().parse("where is the coffee cup").target_category == "mug"
