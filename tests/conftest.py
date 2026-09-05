"""Shared pytest fixtures."""

import pytest

from toolrl.data.schema_generator import SchemaGenerator
from toolrl.env.backend import SQLiteBackend


@pytest.fixture(scope="session")
def generator() -> SchemaGenerator:
    return SchemaGenerator()


@pytest.fixture()
def backend() -> SQLiteBackend:
    b = SQLiteBackend()
    yield b
    b.close()
