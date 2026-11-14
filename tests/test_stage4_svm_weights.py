from run_stage4_svm_weights import configured_search


def test_class_weight_only_changes_requested_parameter():
    weighted = configured_search({0: 1, 1: 100})
    unweighted = configured_search(None)
    assert weighted.estimator.get_params()['svm__class_weight'] == {0: 1, 1: 100}
    assert unweighted.estimator.get_params()['svm__class_weight'] is None
    assert weighted.param_grid == unweighted.param_grid
    assert weighted.cv.n_splits == unweighted.cv.n_splits == 3
    assert weighted.scoring == unweighted.scoring == 'average_precision'
    assert [name for name, _ in weighted.estimator.steps] == [name for name, _ in unweighted.estimator.steps]
