from gegenextract.scoring.semantic import SemanticComparator, InMemorySemanticCache


def test_jaccard_similarity_cache():
    cache = InMemorySemanticCache()
    comp = SemanticComparator(cache)
    s1 = "Alice in wonderland"
    s2 = "Alice wonder"
    sim1 = comp.similarity(s1, s2)
    sim2 = comp.similarity(s1, s2)
    assert sim1 == sim2
    assert 0.0 <= sim1 <= 1.0
