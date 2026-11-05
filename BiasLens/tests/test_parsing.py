from biaslens.parsing import parse_choice, parse_confidence, parse_mcq, parse_rating, parse_yesno

def test_parse_choice_explicit():
    assert parse_choice("Answer: A") == "A"
    assert parse_choice("Option B") == "B"
    assert parse_choice("I choose B.", ("A", "B")) == "B"

def test_parse_choice_bare():
    assert parse_choice("A") == "A"
    assert parse_choice("b") == "B"
    assert parse_choice("nonsense") is None

def test_parse_mcq_four():
    assert parse_mcq("Answer: C", ("A", "B", "C", "D")) == "C"
    assert parse_mcq("D is correct", ("A", "B", "C", "D")) == "D"

def test_parse_confidence_formats():
    assert parse_confidence("Confidence: 85") == 0.85
    assert parse_confidence("I'm 90% sure") == 0.90
    assert parse_confidence("confidence 0.7") == 0.7
    assert parse_confidence("no number here") is None

def test_parse_confidence_clamped():
    assert parse_confidence("Confidence: 150") == 1.0

def test_parse_rating():
    assert parse_rating("Rating: 7/10") == 7.0
    assert parse_rating("I'd rate 8 out of 10") == 8.0
    assert parse_rating("Rating: 12/10") == 10.0  # clamped
    assert parse_rating("no digits") is None

def test_parse_yesno():
    assert parse_yesno("Yes, hire them") is True
    assert parse_yesno("No.") is False
    assert parse_yesno("maybe") is None
