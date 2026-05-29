from gegenextract.scoring.evaluator import Evaluator


def test_nested_objects():
    ev = Evaluator()
    pred = {"person": {"name": "Alice", "age": 30}}
    gold = {"person": {"name": "Alice", "age": 30}}
    res = ev.score(pred, gold)
    assert res["person.name"].f1 == 1.0
    assert res["person.age"].f1 == 1.0
