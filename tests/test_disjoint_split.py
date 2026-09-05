"""Tests for the disjoint train/test split (the correctness keystone)."""

import pytest

from toolrl.data.schema_generator import (
    Column,
    ForeignKey,
    Schema,
    Table,
    assert_disjoint,
    disjoint_split,
    normalized_fingerprint,
    structural_fingerprint,
)


def test_intersection_is_empty(generator):
    schemas = []
    for i, domain in enumerate(("ecommerce", "logistics", "hr")):
        schemas.extend(generator.generate_many(domain, 15, seed=100 + i))
    train, test = disjoint_split(schemas, test_ratio=0.3, seed=42)
    assert len(train) > 0 and len(test) > 0
    # exact structural intersection must be empty
    assert set(map(structural_fingerprint, train)) & set(map(structural_fingerprint, test)) == set()
    # near-duplicate intersection must be empty
    assert set(map(normalized_fingerprint, train)) & set(map(normalized_fingerprint, test)) == set()
    # the helper itself must not raise
    assert_disjoint(train, test)


def test_same_structure_never_split_across_sides(generator):
    # two schemas with the same structure but different seed data must end up
    # on the same side of the split.
    same_structure = [
        generator.generate("ecommerce", seed=100),
        generator.generate("ecommerce", seed=101),
    ]
    # force identical structure by comparing; if the sampled structure differs,
    # just pick the first one and clone it with different seed rows.
    s1 = generator.generate("ecommerce", seed=200)
    s2 = Schema(
        name="clone", domain=s1.domain, tables=s1.tables,
        seed_rows=s1.seed_rows, seed=999,
    )
    train, test = disjoint_split([s1, s2], test_ratio=0.5, seed=1)
    assert structural_fingerprint(s1) == structural_fingerprint(s2)
    # both are in the same list
    assert (s1 in train and s2 in train) or (s1 in test and s2 in test)


def test_near_duplicate_detected():
    # `customers(id, name)` vs `clients(uid, label)` are near-duplicates.
    a = Schema(
        name="a", domain="ecommerce",
        tables=(
            Table(
                name="customers",
                columns=(Column("id", "INTEGER"), Column("name", "TEXT")),
                primary_key="id",
            ),
        ),
        seed_rows={"customers": []},
    )
    b = Schema(
        name="b", domain="ecommerce",
        tables=(
            Table(
                name="clients",
                columns=(Column("uid", "INTEGER"), Column("label", "TEXT")),
                primary_key="uid",
            ),
        ),
        seed_rows={"clients": []},
    )
    assert structural_fingerprint(a) != structural_fingerprint(b)
    assert normalized_fingerprint(a) == normalized_fingerprint(b)


def test_assert_disjoint_raises_on_overlap(generator):
    s = generator.generate("ecommerce", seed=1)
    with pytest.raises(ValueError):
        assert_disjoint([s], [s])


def test_assert_disjoint_raises_on_near_duplicate():
    a = Schema(
        name="a", domain="ecommerce",
        tables=(
            Table(
                name="customers",
                columns=(Column("id", "INTEGER"), Column("name", "TEXT")),
                primary_key="id",
            ),
        ),
        seed_rows={"customers": []},
    )
    b = Schema(
        name="b", domain="ecommerce",
        tables=(
            Table(
                name="clients",
                columns=(Column("uid", "INTEGER"), Column("label", "TEXT")),
                primary_key="uid",
            ),
        ),
        seed_rows={"clients": []},
    )
    with pytest.raises(ValueError):
        assert_disjoint([a], [b])


def test_disjoint_split_empty():
    train, test = disjoint_split([], test_ratio=0.2)
    assert train == [] and test == []


def test_disjoint_split_rejects_bad_ratio(generator):
    with pytest.raises(ValueError):
        disjoint_split([generator.generate("ecommerce", seed=1)], test_ratio=1.5)
