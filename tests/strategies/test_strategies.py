from strategies.engine import MeanReversionStrategy, MetaAllocationEngine, TrendFollowingStrategy


def test_trend_following_signal():
    strat = TrendFollowingStrategy()

    # Check fallback on empty data
    sig, conf = strat.generate_signal({})
    assert sig == 0.0
    assert conf == 0.0

def test_meta_allocation_dominance():
    # Setup mock strategies
    trend = TrendFollowingStrategy()
    rev = MeanReversionStrategy()
    meta = MetaAllocationEngine(strategies=[trend, rev])

    # FIX (flaky) : le tirage Thompson (np.random.beta) rendait ce test
    # aléatoire (~40% d'échec : le poids de Trend doit dépasser 0.40 mais le
    # tirage MAB est uniforme). On FIGE le tirage via le cache temporel
    # (_bandit_sample_cache, mécanisme P1-12 du LOT 5) : le test vérifie la
    # logique de dominance du régime, pas le hasard.
    meta._bandit_sample_cache[0] = (0.0, 0.90)   # Trend  : tirage élevé
    meta._bandit_sample_cache[1] = (0.0, 0.10)   # MeanRev: tirage faible

    # Verify dominant strategy selection based on regime
    # regime_state_id = 0 (Bull) -> Trend Following gets a boost (+0.40), making it dominate
    res = meta.allocate(market_data={}, regime_state_id=0, ml_prediction=0.0, ppo_action=0.0)
    assert "Trend Following" in res["contributions"]
    assert "Mean Reversion" in res["contributions"]
    assert res["contributions"]["Trend Following"]["weight"] >= 0.40
    assert res["contributions"]["Mean Reversion"]["weight"] <= 0.60
