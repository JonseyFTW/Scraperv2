"""Tests for card_name_parser.parse_product_name.

Run from repo root:   pytest tests/test_card_name_parser.py -v
"""
from card_name_parser import parse_product_name


def test_all_fields():
    r = parse_product_name("Joe Burrow #30 [Yellow Pyramids] /185")
    assert r.player_name   == "Joe Burrow"
    assert r.card_number   == "30"
    assert r.print_run     == 185
    assert r.variant_label == "Yellow Pyramids"


def test_name_and_variant_only():
    r = parse_product_name("Joe Burrow [Silver]")
    assert r.player_name   == "Joe Burrow"
    assert r.variant_label == "Silver"
    assert r.card_number   is None
    assert r.print_run     is None


def test_alphanumeric_card_number():
    r = parse_product_name("Topps Chrome Refractor #RPA-25")
    assert r.player_name   == "Topps Chrome Refractor"
    assert r.card_number   == "RPA-25"
    assert r.print_run     is None
    assert r.variant_label is None


def test_print_run_no_card_number():
    r = parse_product_name("Base /99")
    assert r.player_name   == "Base"
    assert r.card_number   is None
    assert r.print_run     == 99
    assert r.variant_label is None


def test_empty_input():
    r = parse_product_name("")
    assert r == parse_product_name(None)
    assert r.player_name is None
    assert r.card_number is None
    assert r.print_run is None
    assert r.variant_label is None


def test_bracket_before_hash():
    # Split on whichever comes first — '[' here.
    r = parse_product_name("Joe Burrow [Silver] #30")
    assert r.player_name   == "Joe Burrow"
    assert r.variant_label == "Silver"
    assert r.card_number   == "30"


def test_plain_name():
    r = parse_product_name("Joe Burrow")
    assert r.player_name   == "Joe Burrow"
    assert r.card_number   is None
    assert r.print_run     is None
    assert r.variant_label is None


def test_upper_deck_one_of_one():
    # "#1/1000" should parse as card_number=1, print_run=1000 per SCP convention.
    r = parse_product_name("Upper Deck Legend #1/1000")
    assert r.card_number == "1"
    assert r.print_run   == 1000


def test_whitespace_around_tokens():
    r = parse_product_name("  Joe Burrow  # 30  [ Silver ] / 185 ")
    assert r.player_name   == "Joe Burrow"
    assert r.card_number   == "30"
    assert r.print_run     == 185
    assert r.variant_label == "Silver"
