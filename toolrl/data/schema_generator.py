"""Schema generation and the disjoint train/test split (SPEC 2.1).

This module is the *correctness keystone* of the project. It produces varied,
structurally-plausible database schemas parameterized by domain (e-commerce,
logistics, HR) and provides a held-out split that is enforced at the
*schema-structure* level — test tasks use schemas whose structure was never
seen in training, not merely different queries over the same schema.

What "disjoint" means here
--------------------------
A schema is identified by its *structure*: the multiset of tables, each table's
columns (name, type, nullability, uniqueness), its primary key, and its
foreign-key edges. The seed rows (data) are *not* part of the identity — two
schemas with the same structure but different rows are the same schema for the
purposes of the split, because what the policy learns to navigate is the
structure.

The split guarantees (and `assert_disjoint` verifies):

1. **Exact disjointness** — the canonical structural fingerprint of any train
   schema never equals that of any test schema (provable by construction: the
   split partitions the *set of fingerprints*, so the two sides are disjoint
   by definition).
2. **Near-duplicate disjointness** — a relaxed fingerprint that canonicalizes
   table/column *identifiers* (so `customers` vs `clients` are recognized as
   the same structure) is also disjoint. This guards against schemas that are
   structurally identical up to renaming sneaking across the split.

The provable guarantee comes from the fact that we partition fingerprints, not
schema instances: every fingerprint lands on exactly one side, so no structure
can appear on both.
"""

from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Iterable, Sequence

try:  # pragma: no cover - exercised via config loading in tests
    import yaml
except ImportError:  # pragma: no cover
    yaml = None


# --------------------------------------------------------------------------- #
# Schema data model
# --------------------------------------------------------------------------- #

_VALID_TYPES = {"INTEGER", "TEXT", "REAL", "BOOLEAN", "TIMESTAMP"}


@dataclass(frozen=True)
class Column:
    name: str
    type: str
    nullable: bool = False
    unique: bool = False
    values: tuple[str, ...] = ()
    min: float | None = None
    max: float | None = None

    def __post_init__(self) -> None:
        if self.type not in _VALID_TYPES:
            raise ValueError(f"invalid column type {self.type!r}")

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {"name": self.name, "type": self.type}
        if self.nullable:
            d["nullable"] = True
        if self.unique:
            d["unique"] = True
        if self.values:
            d["values"] = list(self.values)
        if self.min is not None:
            d["min"] = self.min
        if self.max is not None:
            d["max"] = self.max
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Column":
        return cls(
            name=d["name"],
            type=d["type"],
            nullable=bool(d.get("nullable", False)),
            unique=bool(d.get("unique", False)),
            values=tuple(d.get("values", ())),
            min=d.get("min"),
            max=d.get("max"),
        )


@dataclass(frozen=True)
class ForeignKey:
    column: str
    ref_table: str
    ref_column: str

    def to_dict(self) -> dict[str, str]:
        return {
            "column": self.column,
            "ref_table": self.ref_table,
            "ref_column": self.ref_column,
        }

    @classmethod
    def from_dict(cls, d: dict[str, str]) -> "ForeignKey":
        return cls(d["column"], d["ref_table"], d["ref_column"])


@dataclass(frozen=True)
class Table:
    name: str
    columns: tuple[Column, ...]
    primary_key: str
    foreign_keys: tuple[ForeignKey, ...] = ()

    def column(self, name: str) -> Column:
        for c in self.columns:
            if c.name == name:
                return c
        raise KeyError(f"table {self.name!r} has no column {name!r}")

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "columns": [c.to_dict() for c in self.columns],
            "primary_key": self.primary_key,
            "foreign_keys": [fk.to_dict() for fk in self.foreign_keys],
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Table":
        return cls(
            name=d["name"],
            columns=tuple(Column.from_dict(c) for c in d["columns"]),
            primary_key=d["primary_key"],
            foreign_keys=tuple(ForeignKey.from_dict(f) for f in d.get("foreign_keys", ())),
        )


@dataclass
class Schema:
    """A database schema: structure + deterministic seed data.

    ``seed_rows`` maps table name -> list of row dicts (keys are column names,
    values are Python scalars). It is deterministic given ``seed``, which is
    what makes replay (SPEC 2.5) possible.
    """

    name: str
    domain: str
    tables: tuple[Table, ...]
    seed_rows: dict[str, list[dict[str, Any]]] = field(default_factory=dict)
    seed: int = 0

    def table(self, name: str) -> Table:
        for t in self.tables:
            if t.name == name:
                return t
        raise KeyError(f"schema {self.name!r} has no table {name!r}")

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "domain": self.domain,
            "seed": self.seed,
            "tables": [t.to_dict() for t in self.tables],
            "seed_rows": {k: [dict(r) for r in v] for k, v in self.seed_rows.items()},
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Schema":
        return cls(
            name=d["name"],
            domain=d["domain"],
            tables=tuple(Table.from_dict(t) for t in d["tables"]),
            seed_rows={k: [dict(r) for r in v] for k, v in d.get("seed_rows", {}).items()},
            seed=int(d.get("seed", 0)),
        )


# --------------------------------------------------------------------------- #
# Structural fingerprints
# --------------------------------------------------------------------------- #


def _canonical_table(t: Table) -> str:
    """Exact structural signature of a single table (identifiers included)."""
    parts = [f"table {t.name}"]
    for c in t.columns:
        parts.append(
            f"  col {c.name}:{c.type}:{'N' if c.nullable else ''}"
            f"{'U' if c.unique else ''}"
        )
    parts.append(f"  pk {t.primary_key}")
    for fk in sorted(t.foreign_keys, key=lambda f: (f.column, f.ref_table, f.ref_column)):
        parts.append(f"  fk {fk.column}->{fk.ref_table}.{fk.ref_column}")
    return "\n".join(parts)


def structural_fingerprint(schema: Schema) -> str:
    """Canonical hash of the full schema structure (identifiers included)."""
    parts = [_canonical_table(t) for t in sorted(schema.tables, key=lambda t: t.name)]
    return _sha256("\n".join(parts))


def _column_shape(c: Column) -> tuple[str, bool, bool]:
    return (c.type, c.nullable, c.unique)


def normalized_fingerprint(schema: Schema) -> str:
    """Relaxed structural fingerprint that ignores table/column *names*.

    Two schemas are considered near-duplicates iff their structures are
    identical up to renaming of identifiers. We canonicalize by:

    1. Replacing each table with its *shape*: the sorted multiset of column
       shapes plus its primary-key column shape.
    2. Ordering tables by shape (deterministic), then re-labeling them T0..Tn.
    3. Rewriting foreign-key edges in terms of these canonical indices.

    Column names are dropped entirely (only type/nullability/uniqueness and the
    PK flag survive), so ``customers(id, name)`` and ``clients(uid, label)``
    with the same column shapes hash identically.
    """
    tables = list(schema.tables)
    shapes: dict[str, tuple[tuple[str, bool, bool], ...]] = {}
    pk_shape: dict[str, tuple[str, bool, bool]] = {}
    for t in tables:
        shapes[t.name] = tuple(sorted(_column_shape(c) for c in t.columns))
        pk_shape[t.name] = _column_shape(t.column(t.primary_key))

    # Order tables by (shape, pk-shape, name) for a stable canonical ordering.
    ordered = sorted(
        tables, key=lambda t: (shapes[t.name], pk_shape[t.name], t.name)
    )
    index = {t.name: i for i, t in enumerate(ordered)}

    edges: list[tuple[int, tuple[str, bool, bool], int, tuple[str, bool, bool]]] = []
    for t in ordered:
        for fk in sorted(t.foreign_keys, key=lambda f: (f.column, f.ref_table, f.ref_column)):
            src_shape = _column_shape(t.column(fk.column))
            ref_table = schema.table(fk.ref_table)
            ref_shape = _column_shape(ref_table.column(fk.ref_column))
            edges.append((index[t.name], src_shape, index[fk.ref_table], ref_shape))

    parts = []
    for i, t in enumerate(ordered):
        parts.append(f"T{i} {shapes[t.name]} pk={pk_shape[t.name]}")
    for e in sorted(edges):
        parts.append(f"edge {e}")
    return _sha256("\n".join(parts))


def _sha256(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


# --------------------------------------------------------------------------- #
# Disjoint split
# --------------------------------------------------------------------------- #


def disjoint_split(
    schemas: Sequence[Schema],
    test_ratio: float = 0.2,
    seed: int | None = None,
) -> tuple[list[Schema], list[Schema]]:
    """Split schemas into disjoint train/test sets at the *structure* level.

    Grouping is done by ``normalized_fingerprint`` (near-duplicate-invariant),
    so every structurally-equivalent set of schemas lands entirely on one side.
    Because the groups are partitioned, the train and test *structures* are
    disjoint by construction; ``assert_disjoint`` verifies the guarantee.

    Returns ``(train_schemas, test_schemas)``.
    """
    if not schemas:
        return [], []
    if not (0.0 < test_ratio < 1.0):
        raise ValueError("test_ratio must be in (0, 1)")

    groups: dict[str, list[Schema]] = {}
    for s in schemas:
        groups.setdefault(normalized_fingerprint(s), []).append(s)

    rng = random.Random(seed)
    group_keys = sorted(groups.keys())  # deterministic ordering
    rng.shuffle(group_keys)

    n_test = max(1, int(round(len(group_keys) * test_ratio)))
    test_keys = set(group_keys[:n_test])

    train: list[Schema] = []
    test: list[Schema] = []
    for key in group_keys:
        (test if key in test_keys else train).extend(groups[key])

    assert_disjoint(train, test)
    return train, test


def assert_disjoint(train: Sequence[Schema], test: Sequence[Schema]) -> None:
    """Raise ``ValueError`` if any train/test structure (or near-duplicate) overlaps.

    This is the *provable zero-overlap* guarantee. It checks both the exact
    fingerprint and the identifier-canonicalized (near-duplicate) fingerprint.
    """
    train_exact = {structural_fingerprint(s) for s in train}
    test_exact = {structural_fingerprint(s) for s in test}
    exact_overlap = train_exact & test_exact
    if exact_overlap:
        raise ValueError(f"train/test schemas overlap (exact): {sorted(exact_overlap)}")

    train_norm = {normalized_fingerprint(s) for s in train}
    test_norm = {normalized_fingerprint(s) for s in test}
    norm_overlap = train_norm & test_norm
    if norm_overlap:
        raise ValueError(
            f"train/test schemas overlap (near-duplicate): {sorted(norm_overlap)}"
        )


# --------------------------------------------------------------------------- #
# Schema generator
# --------------------------------------------------------------------------- #

_DEFAULT_CONFIG = Path(__file__).resolve().parents[2] / "configs" / "schema_domains.yaml"


class SchemaGenerator:
    """Generates varied, deterministic schemas parameterized by domain.

    Domain definitions are loaded from ``configs/schema_domains.yaml`` (or a
    caller-supplied path/dict). Each call to :meth:`generate` samples the
    optional tables/columns with a seeded RNG, then fills seed rows with a
    separate deterministic RNG, so ``generate(domain, seed)`` always returns
    the same :class:`Schema`.
    """

    def __init__(self, config: dict[str, Any] | str | Path | None = None):
        if config is None:
            config = _DEFAULT_CONFIG
        if isinstance(config, (str, Path)):
            if yaml is None:
                raise ImportError("PyYAML is required to load schema_domains.yaml")
            with open(config, "r", encoding="utf-8") as fh:
                config = yaml.safe_load(fh)
        self.config = config or {}
        self.defaults = self.config.get("defaults", {})
        self.domains: dict[str, dict[str, Any]] = self.config.get("domains", {})

    @property
    def domain_names(self) -> list[str]:
        return list(self.domains.keys())

    def generate(self, domain: str, seed: int | None = None) -> Schema:
        if domain not in self.domains:
            raise ValueError(
                f"unknown domain {domain!r}; available: {sorted(self.domains)}"
            )
        seed = seed if seed is not None else random.getrandbits(31)
        domain_def = self.domains[domain]
        tables = self._select_tables(domain_def, seed)
        schema = Schema(
            name=f"{domain}_{seed}",
            domain=domain,
            tables=tuple(tables),
            seed=seed,
        )
        schema.seed_rows = self._generate_seed_rows(schema, seed)
        return schema

    def generate_many(
        self, domain: str, n: int, seed: int | None = None
    ) -> list[Schema]:
        if seed is None:
            seed = random.getrandbits(31)
        return [self.generate(domain, seed + i) for i in range(n)]

    # -- structure selection ------------------------------------------------- #

    def _select_tables(self, domain_def: dict[str, Any], seed: int) -> list[Table]:
        rng = random.Random(f"{seed}:structure")
        tables: list[Table] = []
        for name, spec in domain_def.get("tables", {}).items():
            tables.append(self._build_table(name, spec, rng))
        for name, spec in domain_def.get("optional_tables", {}).items():
            if rng.random() < 0.5:
                tables.append(self._build_table(name, spec, rng))
        return tables

    @staticmethod
    def _build_table(name: str, spec: dict[str, Any], rng: random.Random) -> Table:
        columns = [Column(**c) for c in spec.get("columns", [])]
        for c in spec.get("optional_columns", []):
            if rng.random() < 0.5:
                columns.append(Column(**c))
        fks = tuple(ForeignKey(**f) for f in spec.get("foreign_keys", []))
        return Table(name=name, columns=tuple(columns), primary_key=spec["primary_key"], foreign_keys=fks)

    # -- seed data generation ------------------------------------------------- #

    def _generate_seed_rows(self, schema: Schema, seed: int) -> dict[str, list[dict[str, Any]]]:
        # Topological order: parents (referenced tables) before children.
        order = _topological_order(schema.tables)
        row_counts = self._row_counts(schema, seed)
        pks: dict[str, list[Any]] = {}
        seed_rows: dict[str, list[dict[str, Any]]] = {}

        for table in order:
            count = row_counts[table.name]
            rows = self._generate_table_rows(schema, table, count, seed, pks)
            seed_rows[table.name] = rows
            pks[table.name] = [r[table.primary_key] for r in rows]
        return seed_rows

    def _row_counts(self, schema: Schema, seed: int) -> dict[str, int]:
        rng = random.Random(f"{seed}:rowcounts")
        counts: dict[str, int] = {}
        for t in schema.tables:
            spec = self._table_spec(schema.domain, t.name)
            lo, hi = spec.get("row_count", [5, 20])
            counts[t.name] = rng.randint(lo, hi)
        return counts

    def _table_spec(self, domain: str, name: str) -> dict[str, Any]:
        d = self.domains[domain]
        for section in ("tables", "optional_tables"):
            if name in d.get(section, {}):
                return d[section][name]
        raise KeyError(f"no spec for table {name!r} in domain {domain!r}")

    def _generate_table_rows(
        self,
        schema: Schema,
        table: Table,
        count: int,
        seed: int,
        pks: dict[str, list[Any]],
    ) -> list[dict[str, Any]]:
        rng = random.Random(f"{seed}:rows:{table.name}")
        rows: list[dict[str, Any]] = []
        fk_by_col = {fk.column: fk for fk in table.foreign_keys}

        for i in range(1, count + 1):
            row: dict[str, Any] = {}
            for col in table.columns:
                if col.name == table.primary_key:
                    row[col.name] = i
                elif col.name in fk_by_col:
                    fk = fk_by_col[col.name]
                    row[col.name] = self._fk_value(fk, table, i, rng, pks, col)
                else:
                    row[col.name] = self._value_for(col, table, seed, i)
            rows.append(row)
        return rows

    def _fk_value(
        self,
        fk: ForeignKey,
        table: Table,
        i: int,
        rng: random.Random,
        pks: dict[str, list[Any]],
        col: Column,
    ) -> Any:
        if fk.ref_table == table.name:
            # Self-referencing FK (e.g. employees.manager_id): make a valid,
            # acyclic hierarchy — reference an earlier row, or NULL if allowed.
            if col.nullable and i == 1:
                return None
            if col.nullable and rng.random() < 0.15:
                return None
            return rng.randint(1, i - 1) if i > 1 else None
        candidates = pks.get(fk.ref_table)
        if not candidates:
            return None
        return candidates[rng.randrange(len(candidates))]

    def _value_for(self, col: Column, table: Table, seed: int, i: int) -> Any:
        rng = random.Random(f"{seed}:value:{table.name}:{col.name}")
        if col.type == "INTEGER":
            lo = int(col.min) if col.min is not None else 1
            hi = int(col.max) if col.max is not None else 100
            return rng.randint(lo, hi)
        if col.type == "REAL":
            lo = float(col.min) if col.min is not None else 1.0
            hi = float(col.max) if col.max is not None else 100.0
            return round(rng.uniform(lo, hi), 2)
        if col.type == "BOOLEAN":
            return bool(rng.random() < 0.5)
        if col.type == "TIMESTAMP":
            start = datetime.fromisoformat(
                self.defaults.get("timestamp_start", "2020-01-01 00:00:00")
            )
            end = datetime.fromisoformat(
                self.defaults.get("timestamp_end", "2024-12-31 23:59:59")
            )
            delta = (end - start).total_seconds()
            dt = start + timedelta(seconds=rng.uniform(0, delta))
            return dt.strftime("%Y-%m-%d %H:%M:%S")
        # TEXT
        if col.values:
            return rng.choice(col.values)
        return f"{table.name}_{col.name}_{i}"


def _topological_order(tables: Iterable[Table]) -> list[Table]:
    """Order tables so referenced tables come before referencing tables."""
    by_name = {t.name: t for t in tables}
    visited: set[str] = set()
    order: list[Table] = []

    def visit(name: str) -> None:
        if name in visited or name not in by_name:
            return
        visited.add(name)
        t = by_name[name]
        for fk in t.foreign_keys:
            if fk.ref_table in by_name:
                visit(fk.ref_table)
        order.append(t)

    for t in sorted(by_name.values(), key=lambda x: x.name):
        visit(t.name)
    return order
