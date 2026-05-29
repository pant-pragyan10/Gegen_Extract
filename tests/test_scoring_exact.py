from gegenextract.scoring.evaluator import Evaluator


def test_exact_string_match():
    ev = Evaluator()
    pred = {"name": "Alice"}
    gold = {"name": "Alice"}
    res = ev.score(pred, gold)
    assert res["name"].precision == 1.0
    assert res["name"].recall == 1.0
    assert res["name"].f1 == 1.0
