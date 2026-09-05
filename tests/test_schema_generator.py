"""Tests for schema generation (SPEC 2.1)."""

from toolrl.data.schema_generator import SchemaGenerator, structural_fingerprint


def test_generates_all_domains(generator):
    for domain in ("ecommerce", "logistics", "hr"):
        schema = generator.generate(domain, seed=1)
        assert schema.domain == domain
        assert len(schema.tables) >= 3
        for table in schema.tables:
            assert table.primary_key
            assert len(table.columns) >= 1
            assert any(c.name == table.primary_key for c in table.columns)
        # every required table has seed rows
        for table in schema.tables:
            assert len(schema.seed_rows.get(table.name, [])) > 0


def test_deterministic(generator):
    a = generator.generate("ecommerce", seed=123)
    b = generator.generate("ecommerce", seed=123)
    assert structural_fingerprint(a) == structural_fingerprint(b)
    assert a.seed_rows == b.seed_rows


def test_varied_structures(generator):
    schemas = generator.generate_many("ecommerce", 30, seed=10)
    fingerprints = {structural_fingerprint(s) for s in schemas}
    assert len(fingerprints) >= 2


def test_foreign_keys_are_valid(generator):
    for domain in ("ecommerce", "logistics", "hr"):
        schema = generator.generate(domain, seed=5)
        table_names = {t.name for t in schema.tables}
        for table in schema.tables:
            col_names = {c.name for c in table.columns}
            for fk in table.foreign_keys:
                assert fk.column in col_names
                assert fk.ref_table in table_names
                ref = schema.table(fk.ref_table)
                assert any(c.name == fk.ref_column for c in ref.columns)


def test_seed_rows_satisfy_foreign_keys(generator):
    for domain in ("ecommerce", "logistics", "hr"):
        schema = generator.generate(domain, seed=7)
        pks = {
            t.name: {r[t.primary_key] for r in schema.seed_rows.get(t.name, [])}
            for t in schema.tables
        }
        for table in schema.tables:
            for fk in table.foreign_keys:
                if fk.ref_table == table.name:
                    continue  # self-FK (manager_id) may legitimately be NULL
                for row in schema.seed_rows.get(table.name, []):
                    assert row[fk.column] in pks[fk.ref_table]


def test_schema_serialization_roundtrip(generator):
    schema = generator.generate("hr", seed=9)
    restored = type(schema).from_dict(schema.to_dict())
    assert structural_fingerprint(restored) == structural_fingerprint(schema)
    assert restored.seed_rows == schema.seed_rows
