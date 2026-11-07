import numpy as np
from rdkit import Chem

from experiments.v4_mmp_transfer import MMPExpert, fragments


def test_single_cut_core_constraints_and_maps():
    smiles = "CCOc1ccccc1"
    parts = fragments(smiles)
    assert parts
    original_size = Chem.MolFromSmiles(smiles).GetNumHeavyAtoms()
    for core, variable in parts:
        c, v = Chem.MolFromSmiles(core), Chem.MolFromSmiles(variable)
        assert len(Chem.GetMolFrags(c)) == len(Chem.GetMolFrags(v)) == 1
        assert c.GetNumHeavyAtoms() * 2 >= original_size
        assert v.GetNumHeavyAtoms() <= 8
        assert c.GetNumHeavyAtoms() + v.GetNumHeavyAtoms() == original_size
        for mol in (c, v):
            dummy = [a for a in mol.GetAtoms() if a.GetAtomicNum() == 0]
            assert len(dummy) == 1 and dummy[0].GetAtomMapNum() == 1


def test_training_only_transform_labels_and_real_transfer():
    # Two repeated methyl->hydroxyl substitutions on different aromatic cores.
    smiles = ["Cc1ccccc1", "Oc1ccccc1", "Cc1ccncc1", "Oc1ccncc1", "Cc1ccoc1"]
    ctx = np.array([["vaccinia", "vero"]] * 5)
    model = MMPExpert().fit([fragments(s) for s in smiles], ctx,
                            np.array([10., 20., 30., 40., 50.]), [0, 1, 2, 3, 4], "mmp-raw")
    labels = {i for records in model.transforms.values() for pair in records for i in pair}
    assert labels <= {0, 1, 2, 3, 4} and 99 not in labels
    pred, fired, support = model.predict([fragments("Oc1ccoc1")], ctx[:1], [7.])
    assert fired[0] and support[0] >= 2 and pred[0] > 50
    # Unknown context must not borrow a transformation from a different assay.
    pred, fired, _ = model.predict([fragments("Oc1ccoc1")], np.array([["other", "vero"]]), [7.])
    assert not fired[0] and pred[0] == 7
