import numpy as np

from experiments.v8_nested_validation import connected_groups, make_folds, representation


def test_union_preserves_both_relations_and_does_not_merge_empty_scaffolds():
    identities = ["a", "b", "b", "c", "d", "d", "e"]
    scaffolds = ["s1", "s1", "s2", "s2", "", "", ""]
    groups = connected_groups(identities, scaffolds)
    assert len(set(groups[:4])) == 1
    assert groups[4] == groups[5]
    assert groups[6] not in (groups[0], groups[4])


def test_frozen_splits_keep_identity_and_component_groups_inside_folds():
    identities = np.repeat(np.arange(40), 2)
    components = identities // 2
    schemes = make_folds(identities, components)
    assert schemes == make_folds(identities, components)
    for name, outer in schemes.items():
        coverage = np.zeros(len(identities), int)
        for fold in outer:
            tr, va = np.asarray(fold["train"]), np.asarray(fold["valid"])
            coverage[va] += 1
            assert set(tr).isdisjoint(va)
            assert set(tr) | set(va) == set(range(len(identities)))
            for f in [fold, *fold["inner"]]:
                a, b = f["train"], f["valid"]
                assert set(identities[a]).isdisjoint(identities[b])
                if name == "scaffold":
                    assert set(components[a]).isdisjoint(components[b])
            assert all(set(f["train"]) | set(f["valid"]) == set(tr) for f in fold["inner"])
        np.testing.assert_array_equal(coverage, 1)


def test_preprocessing_fit_does_not_depend_on_query_embeddings():
    rng = np.random.default_rng(9)
    bits = rng.integers(0, 2, (18, 8)).astype(float)
    embedding = rng.normal(size=(18, 40))
    tr = np.arange(12)
    before = representation(bits, embedding, tr)
    embedding[12:] += 1e5
    after = representation(bits, embedding, tr)
    for a, b in zip(before, after):
        np.testing.assert_allclose(a[tr], b[tr], atol=1e-10)
