import pagerank

damping_factor = 0.85


def test_transition_model():
    corpus = {"1.html": {"2.html", "3.html"}, "2.html": {"3.html"}, "3.html": {"2.html"}}
    expected = {"1.html": 0.05, "2.html": 0.475, "3.html": 0.475}
    assert pagerank.transition_model(corpus, "1.html", damping_factor) == expected


def test_transition_model_no_outgoing_links():
    corpus = {"1.html": {}, "2.html": {"1.html"}}
    expected = {"1.html": 0.5, "2.html": 0.5}
    assert pagerank.transition_model(corpus, "1.html", damping_factor) == expected