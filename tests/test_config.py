import config


def test_repeat_configuration():
    assert config.N_REPEATS == 10
    assert config.REPEAT_SEEDS == list(range(42, 52))


def test_split_fractions():
    total = (
        config.TRAIN_FRAC
        + config.CALIB_FRAC
        + config.TEST_FRAC
    )

    assert total == 1.0
    assert config.TRAIN_FRAC == 0.60
    assert config.CALIB_FRAC == 0.20
    assert config.TEST_FRAC == 0.20


def test_experiment_dimensions():
    assert len(config.COHORTS) == 4
    assert len(config.ALL_MODELS) == 9
    assert len(config.ALL_METHODS) == 5


def test_evaluation_configuration():
    assert config.PRIMARY_METRIC == "brier"
    assert config.ECE_PRIMARY_N_BINS == 5
    assert config.ECE_SENSITIVITY_N_BINS == 10
    assert config.CONFIDENCE_LEVEL == 0.95

def test_nested_cross_validation_configuration():
    assert config.N_REPEATS == 10
    assert config.N_OUTER_FOLDS == 5
    assert config.N_INNER_FOLDS == 5

    assert len(
        config.REPEAT_SEEDS
    ) == config.N_REPEATS

    assert len(
        set(config.REPEAT_SEEDS)
    ) == config.N_REPEATS