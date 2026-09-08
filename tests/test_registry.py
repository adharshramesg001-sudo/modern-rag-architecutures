from rag_compare.registry import ALL_SPECS, BY_KEY

EXPECTED_KEYS = {"classic", "hybrid", "graph", "agentic", "corrective", "multimodal"}


def test_all_expected_architectures_registered():
    assert {spec.key for spec in ALL_SPECS} == EXPECTED_KEYS
    assert set(BY_KEY.keys()) == EXPECTED_KEYS


def test_every_spec_has_required_fields():
    for spec in ALL_SPECS:
        assert spec.name
        assert spec.pipeline, f"{spec.key} has an empty pipeline"
        assert spec.good_points and spec.bad_points
        assert spec.default_query
        assert spec.compare_row


def test_compare_rows_share_the_same_dimensions():
    dimension_sets = [frozenset(spec.compare_row.keys()) for spec in ALL_SPECS]
    assert len(set(dimension_sets)) == 1, "All architectures must report the same comparison dimensions"
