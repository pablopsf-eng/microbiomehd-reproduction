"""Tests for mbhd.classifier (M4).

Synthetic known-answer
tests for the classifier mechanics (perfect separability -> AUC 1.0,
label-independent features -> AUC ~0.5), a folder-grouping regression test
(real data, skip-guarded), a leakage assertion on the underlying
train/test-split mechanism, and a self-check that the permutation null's own
mean AUC is ~0.5 by construction.
"""

from __future__ import annotations

import pathlib

import numpy as np
import pandas as pd
import pytest
import yaml
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.preprocessing import StandardScaler

from mbhd.classifier import (
    DegenerateFoldError,
    _pipeline,
    _tuned_pipeline,
    leave_one_group_out_auc,
    leave_one_group_out_auc_dynamic_features,
    permutation_null_auc,
    pooled_feature_matrix,
    within_study_cv_auc,
)
from mbhd.compositional import clr, multiplicative_replacement
from mbhd.datasets import load_dataset_info
from mbhd.differential import NoSurvivingSamplesError

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
DATA_RAW = _REPO_ROOT / "data" / "raw"
DATASET_INFO = DATA_RAW / "dataset_info.yaml"
SEED = 42


def _require(*paths: pathlib.Path) -> None:
    missing = [p.name for p in paths if not p.exists()]
    if missing:
        pytest.skip(f"data not fetched: {', '.join(missing)}")


# --------------------------------------------------------------------------
# pooled_feature_matrix: folder-grouping (both synthetic and real-data).
# --------------------------------------------------------------------------


def _build_shared_folder_archive(tmp_path: pathlib.Path) -> tuple[dict, pathlib.Path]:
    """Two dataset IDs (cdi_schubert, noncdi_schubert -- real CASE_CONTROL
    entries) sharing one physical folder and one control ("H") sample pool,
    exactly the shared-folder shape of the real archive
    (cdi_schubert/noncdi_schubert, nash_zhu/ob_zhu)."""
    data_root = tmp_path / "data_root"
    folder = "cdi_schubert_results"
    study_dir = data_root / folder
    (study_dir / "RDP").mkdir(parents=True)

    # Counts are large enough to clear differential_abundance's default
    # 100-reads-per-sample floor -- pooled_feature_matrix uses filter_table's
    # own defaults, unlike the synthetic archives in tests/test_differential.py
    # (which pass min_sample_reads=0 explicitly to differential_abundance).
    otu = pd.DataFrame(
        {
            "H1": [100, 0],
            "H2": [120, 0],
            "C1": [0, 100],
            "C2": [0, 120],
            "N1": [20, 80],
            "N2": [30, 90],
        },
        index=[
            "k__B;p__X;c__X;o__X;f__X;g__Foo;s__;d__denovo1",
            "k__B;p__X;c__X;o__X;f__X;g__Bar;s__;d__denovo2",
        ],
    )
    otu_filename = "cdi_schubert.otu_table.100.denovo.rdp_assigned"
    otu.to_csv(study_dir / "RDP" / otu_filename, sep="\t")
    (study_dir / "summary_file.txt").write_text(
        "DATASET_ID\tcdi_schubert\n"
        "#16S_start\n"
        f"OTU_TABLE_RDP\t{otu_filename}\n"
        "PROCESSED\tTrue\n"
        "#16S_end\n"
    )
    metadata = pd.DataFrame(
        {
            "DiseaseState": {
                "H1": "H",
                "H2": "H",
                "C1": "CDI",
                "C2": "CDI",
                "N1": "nonCDI",
                "N2": "nonCDI",
            }
        }
    )
    metadata.to_csv(study_dir / "cdi_schubert.metadata.txt", sep="\t")

    dataset_info_path = data_root / "dataset_info.yaml"
    dataset_info_path.write_text(
        yaml.dump(
            {
                "cdi_schubert": {"folder": folder},
                "noncdi_schubert": {"folder": folder},
            }
        )
    )
    index = load_dataset_info(dataset_info_path)
    return index, data_root


def test_pooled_feature_matrix_shared_folder_dataset_ids_get_the_same_group(
    tmp_path: pathlib.Path,
) -> None:
    index, data_root = _build_shared_folder_archive(tmp_path)
    genera = [
        "k__B;p__X;c__X;o__X;f__X;g__Foo",
        "k__B;p__X;c__X;o__X;f__X;g__Bar",
    ]
    x, y, groups = pooled_feature_matrix(
        ["cdi_schubert", "noncdi_schubert"], genera, index, data_root
    )
    assert set(groups.unique()) == {"cdi_schubert_results"}
    assert list(x.columns) == genera
    # cdi_schubert: 2 H (0) + 2 CDI (1); noncdi_schubert: 2 H (0) + 2 nonCDI (1).
    assert len(x) == 8
    assert y.sum() == 4


def _build_par_scheperjans_archive(tmp_path: pathlib.Path) -> tuple[dict, pathlib.Path]:
    """Single-dataset archive, H vs PAR, counts large enough to clear
    pooled_feature_matrix's default read-count filters -- g__Foo is 100% of
    the H group's relative abundance and entirely absent from PAR (and vice
    versa for g__Bar), a clean, hand-verifiable asymmetry."""
    data_root = tmp_path / "data_root"
    folder = "par_scheperjans_results"
    study_dir = data_root / folder
    (study_dir / "RDP").mkdir(parents=True)
    otu = pd.DataFrame(
        {"S1": [100, 0], "S2": [120, 0], "S3": [0, 100], "S4": [0, 120]},
        index=[
            "k__B;p__X;c__X;o__X;f__X;g__Foo;s__;d__denovo1",
            "k__B;p__X;c__X;o__X;f__X;g__Bar;s__;d__denovo2",
        ],
    )
    otu_filename = "par_scheperjans.otu_table.100.denovo.rdp_assigned"
    otu.to_csv(study_dir / "RDP" / otu_filename, sep="\t")
    (study_dir / "summary_file.txt").write_text(
        "DATASET_ID\tpar_scheperjans\n"
        "#16S_start\n"
        f"OTU_TABLE_RDP\t{otu_filename}\n"
        "PROCESSED\tTrue\n"
        "#16S_end\n"
    )
    pd.DataFrame(
        {"DiseaseState": {"S1": "H", "S2": "H", "S3": "PAR", "S4": "PAR"}}
    ).to_csv(study_dir / "par_scheperjans.metadata.txt", sep="\t")
    dataset_info_path = data_root / "dataset_info.yaml"
    dataset_info_path.write_text(yaml.dump({"par_scheperjans": {"folder": folder}}))
    index = load_dataset_info(dataset_info_path)
    return index, data_root


def test_pooled_feature_matrix_abundance_transform_default_matches_omitting_it(
    tmp_path: pathlib.Path,
) -> None:
    index, data_root = _build_par_scheperjans_archive(tmp_path)
    genera = ["k__B;p__X;c__X;o__X;f__X;g__Foo", "k__B;p__X;c__X;o__X;f__X;g__Bar"]

    without_kwarg = pooled_feature_matrix(["par_scheperjans"], genera, index, data_root)
    with_default = pooled_feature_matrix(
        ["par_scheperjans"], genera, index, data_root, abundance_transform=None
    )
    pd.testing.assert_frame_equal(without_kwarg[0], with_default[0])
    pd.testing.assert_series_equal(without_kwarg[1], with_default[1])
    pd.testing.assert_series_equal(without_kwarg[2], with_default[2])


def test_pooled_feature_matrix_applies_clr_transform(tmp_path: pathlib.Path) -> None:
    """g__Foo is 100% of every H sample's relative abundance (raw value
    1.0) -- under CLR it must not be, proving abundance_transform actually
    ran rather than being silently ignored."""
    index, data_root = _build_par_scheperjans_archive(tmp_path)
    genera = ["k__B;p__X;c__X;o__X;f__X;g__Foo", "k__B;p__X;c__X;o__X;f__X;g__Bar"]

    raw_x, _, _ = pooled_feature_matrix(["par_scheperjans"], genera, index, data_root)
    clr_x, _, _ = pooled_feature_matrix(
        ["par_scheperjans"],
        genera,
        index,
        data_root,
        abundance_transform=lambda t: clr(t, pseudocount_fn=multiplicative_replacement),
    )

    foo = genera[0]
    h_keys = [k for k in raw_x.index if k.endswith(":S1") or k.endswith(":S2")]
    assert len(h_keys) == 2
    np.testing.assert_allclose(raw_x.loc[h_keys, foo].to_numpy(), 1.0)
    assert not np.allclose(clr_x.loc[h_keys, foo].to_numpy(), 1.0)


def test_pooled_feature_matrix_raises_when_a_group_has_no_surviving_samples(
    tmp_path: pathlib.Path,
) -> None:
    data_root = tmp_path / "data_root"
    folder = "par_scheperjans_results"
    study_dir = data_root / folder
    (study_dir / "RDP").mkdir(parents=True)
    otu = pd.DataFrame(
        {"S1": [50, 5], "S2": [40, 6]},
        index=[
            "k__B;p__X;c__X;o__X;f__X;g__Foo;s__;d__denovo1",
            "k__B;p__X;c__X;o__X;f__X;g__Bar;s__;d__denovo2",
        ],
    )
    otu_filename = "par_scheperjans.otu_table.100.denovo.rdp_assigned"
    otu.to_csv(study_dir / "RDP" / otu_filename, sep="\t")
    (study_dir / "summary_file.txt").write_text(
        f"OTU_TABLE_RDP\t{otu_filename}\n#16S_start\n#16S_end\n"
    )
    pd.DataFrame({"DiseaseState": {"S1": "H", "S2": "H"}}).to_csv(
        study_dir / "par_scheperjans.metadata.txt", sep="\t"
    )
    dataset_info_path = data_root / "dataset_info.yaml"
    dataset_info_path.write_text(yaml.dump({"par_scheperjans": {"folder": folder}}))
    index = load_dataset_info(dataset_info_path)

    with pytest.raises(NoSurvivingSamplesError, match="par_scheperjans"):
        pooled_feature_matrix(
            ["par_scheperjans"],
            ["k__B;p__X;c__X;o__X;f__X;g__Foo"],
            index,
            data_root,
        )


def test_folder_grouping_regression_on_real_dataset_info() -> None:
    """The two real shared-folder pairs (cdi/noncdi_schubert; nash_zhu/ob_zhu)
    must always resolve to the same folder -- the fact
    pooled_feature_matrix's/leave_one_group_out_auc's whole leakage-avoidance
    design depends on."""
    _require(DATASET_INFO)
    index = load_dataset_info(DATASET_INFO)
    assert index["cdi_schubert"].folder == index["noncdi_schubert"].folder
    assert index["nash_zhu"].folder == index["ob_zhu"].folder


# --------------------------------------------------------------------------
# leave_one_group_out_auc: synthetic known-answer tests.
# --------------------------------------------------------------------------


def _synthetic_groups(n_groups: int, n_per_class: int) -> pd.Series:
    keys = [f"g{g}:s{i}" for g in range(n_groups) for i in range(2 * n_per_class)]
    values = [f"g{g}" for g in range(n_groups) for _ in range(2 * n_per_class)]
    return pd.Series(values, index=keys)


def test_perfect_separability_gives_auc_one() -> None:
    """A feature identical to the label, with no group-specific offset, must
    let a fold trained on other groups perfectly rank an unseen group."""
    n_groups, n_per_class = 5, 4
    groups = _synthetic_groups(n_groups, n_per_class)
    y = pd.Series(
        ([0] * n_per_class + [1] * n_per_class) * n_groups, index=groups.index
    )
    x = pd.DataFrame({"f": y.to_numpy(dtype=float)}, index=groups.index)

    result = leave_one_group_out_auc(x, y, groups, seed=SEED)
    assert len(result) == n_groups
    np.testing.assert_allclose(result["auc_roc"].to_numpy(), 1.0)
    np.testing.assert_allclose(result["auc_pr"].to_numpy(), 1.0)


def test_label_independent_features_give_chance_level_auc() -> None:
    """A feature drawn independent of the label (fixed seed) must give a
    LOFO AUC-ROC nowhere near 1.0 or 0.0 on average across folds."""
    rng = np.random.default_rng(SEED)
    n_groups, n_per_class = 6, 15
    groups = _synthetic_groups(n_groups, n_per_class)
    y = pd.Series(
        ([0] * n_per_class + [1] * n_per_class) * n_groups, index=groups.index
    )
    x = pd.DataFrame(
        {"f": rng.normal(size=len(groups))},
        index=groups.index,
    )

    result = leave_one_group_out_auc(x, y, groups, seed=SEED)
    assert 0.3 < result["auc_roc"].mean() < 0.7


# --------------------------------------------------------------------------
# leave_one_group_out_auc_dynamic_features: the per-fold, leak-free feature
# selection variant added after review flagged the fixed-global-pool design
# as a feature-selection leakage path (M4, docs/reproduction-m4.md, Part B).
# --------------------------------------------------------------------------


def test_dynamic_features_uses_a_different_feature_set_per_fold() -> None:
    """Mechanical proof that features_for_fold's return value actually
    determines what each fold is trained/scored on: an "informative" column
    (== label, works in every group) and a "noise" column (independent of
    label everywhere). Folds told to use "informative" must score ~1.0;
    folds told to use "noise" must not."""
    n_groups, n_per_class = 4, 6
    groups = _synthetic_groups(n_groups, n_per_class)
    y = pd.Series(
        ([0] * n_per_class + [1] * n_per_class) * n_groups, index=groups.index
    )
    rng = np.random.default_rng(SEED)
    x = pd.DataFrame(
        {
            "informative": y.to_numpy(dtype=float),
            "noise": rng.normal(size=len(groups)),
        },
        index=groups.index,
    )

    calls: list[str] = []

    def features_for_fold(held_out_group: str) -> list[str]:
        calls.append(held_out_group)
        return ["noise"] if held_out_group == "g0" else ["informative"]

    result = leave_one_group_out_auc_dynamic_features(
        x, y, groups, seed=SEED, features_for_fold=features_for_fold
    )
    assert sorted(calls) == sorted(f"g{g}" for g in range(n_groups))

    by_group = result.set_index("group")
    assert by_group.loc["g0", "n_features"] == 1
    assert by_group.loc["g0", "auc_roc"] != pytest.approx(1.0)
    for g in ("g1", "g2", "g3"):
        assert by_group.loc[g, "auc_roc"] == pytest.approx(1.0)


def test_dynamic_features_degenerate_fold_raises_named_error() -> None:
    groups = pd.Series(
        ["a", "a", "b", "b", "c", "c"], index=["s1", "s2", "s3", "s4", "s5", "s6"]
    )
    y = pd.Series([0, 1, 0, 0, 0, 1], index=groups.index)  # "b" is all-control
    x = pd.DataFrame({"f": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]}, index=groups.index)

    with pytest.raises(DegenerateFoldError, match="'b'"):
        leave_one_group_out_auc_dynamic_features(
            x, y, groups, seed=SEED, features_for_fold=lambda _g: ["f"]
        )


def test_dynamic_features_pipeline_fn_default_matches_omitting_it() -> None:
    n_groups, n_per_class = 4, 5
    groups = _synthetic_groups(n_groups, n_per_class)
    y = pd.Series(
        ([0] * n_per_class + [1] * n_per_class) * n_groups, index=groups.index
    )
    x = pd.DataFrame({"f": y.to_numpy(dtype=float)}, index=groups.index)

    without_kwarg = leave_one_group_out_auc_dynamic_features(
        x, y, groups, seed=SEED, features_for_fold=lambda _g: ["f"]
    )
    with_default = leave_one_group_out_auc_dynamic_features(
        x,
        y,
        groups,
        seed=SEED,
        features_for_fold=lambda _g: ["f"],
        pipeline_fn=_pipeline,
    )
    pd.testing.assert_frame_equal(without_kwarg, with_default)
    assert without_kwarg["selected_C"].isna().all()


def test_tuned_pipeline_records_selected_c_and_plain_pipeline_does_not() -> None:
    """_tuned_pipeline's LogisticRegressionCV exposes a fitted C_ per fold,
    recorded as selected_C -- the plain _pipeline's LogisticRegression has
    no such attribute, so its rows must report None instead (M4 follow-up:
    class_weight="balanced" + nested-CV-tuned C)."""
    n_groups, n_per_class = 5, 15
    groups = _synthetic_groups(n_groups, n_per_class)
    y = pd.Series(
        ([0] * n_per_class + [1] * n_per_class) * n_groups, index=groups.index
    )
    rng = np.random.default_rng(SEED)
    x = pd.DataFrame(
        {
            "f1": y.to_numpy(dtype=float) * 2.0
            + rng.normal(size=len(groups), scale=0.5),
            "f2": rng.normal(size=len(groups)),
        },
        index=groups.index,
    )

    tuned = leave_one_group_out_auc_dynamic_features(
        x,
        y,
        groups,
        seed=SEED,
        features_for_fold=lambda _g: ["f1", "f2"],
        pipeline_fn=_tuned_pipeline,
    )
    assert tuned["selected_C"].notna().all()
    assert (tuned["selected_C"] > 0).all()


def test_degenerate_fold_raises_named_error() -> None:
    """A held-out group with only one class present must fail loudly,
    naming the group, rather than silently propagate a NaN AUC. A third,
    mixed-class group keeps every *other* fold's training data non-
    degenerate, isolating the failure to the held-out side (module
    docstring: this cannot arise on real data, where every dataset already
    has >=1 control and >=1 case)."""
    groups = pd.Series(
        ["a", "a", "b", "b", "c", "c"], index=["s1", "s2", "s3", "s4", "s5", "s6"]
    )
    y = pd.Series([0, 1, 0, 0, 0, 1], index=groups.index)  # group "b" is all-control
    x = pd.DataFrame({"f": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]}, index=groups.index)

    with pytest.raises(DegenerateFoldError, match="'b'"):
        leave_one_group_out_auc(x, y, groups, seed=SEED)


# --------------------------------------------------------------------------
# Leakage: the underlying LeaveOneGroupOut + fit-on-train-only mechanism
# leave_one_group_out_auc relies on.
# --------------------------------------------------------------------------


def test_scaler_fit_inside_a_fold_never_sees_the_held_out_groups_data() -> None:
    """group C's own feature values are shifted far from A/B's -- a scaler
    fit on the training fold alone must land at the training-only mean, and
    that must differ from the full-pooled mean whenever the held-out
    group's data differs from it, proving the held-out fold's values were
    never included in the fit."""
    x = pd.DataFrame(
        {"f": [0.0, 1.0, 0.0, 1.0, 100.0, 101.0]},
        index=["a1", "a2", "b1", "b2", "c1", "c2"],
    )
    y = pd.Series([0, 1, 0, 1, 0, 1], index=x.index)
    groups = pd.Series(["A", "A", "B", "B", "C", "C"], index=x.index)

    x_arr, y_arr, groups_arr = x.to_numpy(), y.to_numpy(), groups.to_numpy()
    for train_idx, test_idx in LeaveOneGroupOut().split(x_arr, y_arr, groups_arr):
        held_out = groups_arr[test_idx][0]
        if held_out != "C":
            continue
        scaler = StandardScaler().fit(x_arr[train_idx])
        train_only_mean = x_arr[train_idx].mean()
        full_pool_mean = x_arr.mean()
        assert scaler.mean_[0] == pytest.approx(train_only_mean)
        assert scaler.mean_[0] != pytest.approx(full_pool_mean)


# --------------------------------------------------------------------------
# Permutation null: self-check that its own mean AUC is ~0.5 by construction.
# --------------------------------------------------------------------------


def test_permutation_null_mean_auc_is_approximately_half() -> None:
    n_groups, n_per_class = 4, 10
    groups = _synthetic_groups(n_groups, n_per_class)
    rng = np.random.default_rng(SEED)
    y = pd.Series(
        ([0] * n_per_class + [1] * n_per_class) * n_groups, index=groups.index
    )
    # A real, strong signal -- the null must destroy it via shuffling alone.
    x = pd.DataFrame(
        {"f": y.to_numpy(dtype=float) * 5.0 + rng.normal(size=len(groups), scale=0.1)},
        index=groups.index,
    )

    null = permutation_null_auc(x, y, groups, seed=SEED, n_repeats=30)
    assert len(null) == 30
    assert 0.35 < null.mean() < 0.65


# --------------------------------------------------------------------------
# within_study_cv_auc: secondary comparison.
# --------------------------------------------------------------------------


def test_within_study_cv_auc_separable_feature() -> None:
    y = pd.Series([0, 0, 0, 0, 1, 1, 1, 1])
    x = pd.DataFrame({"f": [0.0, 0.1, 0.0, 0.1, 10.0, 10.1, 10.0, 10.1]})
    mean_auc, actual_n_splits = within_study_cv_auc(x, y, seed=SEED, n_splits=4)
    assert actual_n_splits == 4
    assert mean_auc == pytest.approx(1.0)


def test_within_study_cv_auc_caps_n_splits_to_minority_class_size() -> None:
    y = pd.Series([0, 0, 1, 1, 1, 1, 1, 1])  # minority class ("0") has 2 members
    x = pd.DataFrame({"f": [0.0, 0.1, 5.0, 5.1, 5.2, 5.3, 5.4, 5.5]})
    _, actual_n_splits = within_study_cv_auc(x, y, seed=SEED, n_splits=5)
    assert actual_n_splits == 2


def test_within_study_cv_auc_raises_when_minority_class_too_small() -> None:
    y = pd.Series([0, 1, 1, 1, 1])  # minority class has 1 member
    x = pd.DataFrame({"f": [0.0, 1.0, 2.0, 3.0, 4.0]})
    with pytest.raises(DegenerateFoldError, match="1 sample"):
        within_study_cv_auc(x, y, seed=SEED)
