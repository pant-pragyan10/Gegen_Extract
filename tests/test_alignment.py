from gegenextract.scoring.alignment import align_arrays


def test_greedy_alignment_simple():
    pred = [{"id": 1}, {"id": 2}]
    gold = [{"id": 1}, {"id": 2}]
    matches = align_arrays(pred, gold)
    assert len(matches) == 2
    assert matches[0][2] == 1.0


def test_greedy_alignment_unmatched():
    pred = [{"id": 1}]
    gold = [{"id": 2}]
    matches = align_arrays(pred, gold)
    assert matches[0][0] is not None
    assert matches[0][2] == 0.0
