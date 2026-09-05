"""Multi-step task generation on top of schemas (SPEC 2.1).

Each :class:`Task` is a natural-language goal that *requires 2+ dependent tool
calls* to solve — later queries depend on values produced by earlier ones (e.g.
"find the top customer, then look up their most recent order"). Ground truth is
computed by actually running the reference ("gold") SQL against the schema's
deterministic seed data, so the answer the environment grades against is
exactly what the DB would return — no hand-written answers, no drift.

Task templates are domain-specific and only reference *required* tables and
columns (which are always present regardless of the sampled structural
variant), which keeps generation robust across the varied schemas produced by
:class:`~toolrl.data.schema_generator.SchemaGenerator`.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any, Callable

from toolrl.data.schema_generator import Schema
from toolrl.env.backend import Backend, BackendHandle, SQLiteBackend


@dataclass
class Task:
    task_id: str
    schema_name: str
    domain: str
    instruction: str
    ground_truth: Any
    min_steps: int
    seed: int
    gold_sql: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


# A template is a function (schema, run) -> (instruction, gold_sql, answer),
# where run(sql) -> list[tuple] executes SQL against the materialized schema.
Template = Callable[[Schema, Callable[[str], list[tuple]]], tuple[str, list[str], Any]]


def _scalar(rows: list[tuple]) -> Any:
    return rows[0][0] if rows and rows[0] else None


def _lit(value: Any) -> str:
    if value is None:
        return "NULL"
    if isinstance(value, (int, float)):
        return str(value)
    return "'" + str(value).replace("'", "''") + "'"


# --------------------------------------------------------------------------- #
# E-commerce templates
# --------------------------------------------------------------------------- #


def _ecom_top_customer_recent_order(schema: Schema, run) -> tuple[str, list[str], Any]:
    s1 = (
        "SELECT customer_id FROM orders GROUP BY customer_id "
        "ORDER BY SUM(total) DESC, customer_id ASC LIMIT 1"
    )
    top = _scalar(run(s1))
    s2 = (
        f"SELECT order_id FROM orders WHERE customer_id = {_lit(top)} "
        "ORDER BY created_at DESC, order_id DESC LIMIT 1"
    )
    order_id = _scalar(run(s2))
    instruction = (
        "Find the customer who has spent the most in total across all their "
        "orders, then return the order ID of that customer's most recent order."
    )
    return instruction, [s1, s2], order_id


def _ecom_top_customer_recent_order_status(schema: Schema, run) -> tuple[str, list[str], Any]:
    s1 = (
        "SELECT customer_id FROM orders GROUP BY customer_id "
        "ORDER BY SUM(total) DESC, customer_id ASC LIMIT 1"
    )
    top = _scalar(run(s1))
    s2 = (
        f"SELECT order_id FROM orders WHERE customer_id = {_lit(top)} "
        "ORDER BY created_at DESC, order_id DESC LIMIT 1"
    )
    order_id = _scalar(run(s2))
    s3 = f"SELECT status FROM orders WHERE order_id = {_lit(order_id)}"
    status = _scalar(run(s3))
    instruction = (
        "Find the customer who has spent the most overall, then find their most "
        "recent order, and finally return the status of that order."
    )
    return instruction, [s1, s2, s3], status


def _ecom_most_orders_region(schema: Schema, run) -> tuple[str, list[str], Any]:
    s1 = (
        "SELECT customer_id FROM orders GROUP BY customer_id "
        "ORDER BY COUNT(*) DESC, customer_id ASC LIMIT 1"
    )
    top = _scalar(run(s1))
    s2 = f"SELECT region FROM customers WHERE customer_id = {_lit(top)}"
    region = _scalar(run(s2))
    instruction = (
        "Find the customer who placed the most orders, then return the region "
        "that customer belongs to."
    )
    return instruction, [s1, s2], region


def _ecom_top_product_category(schema: Schema, run) -> tuple[str, list[str], Any]:
    s1 = (
        "SELECT product_id FROM order_items GROUP BY product_id "
        "ORDER BY SUM(quantity) DESC, product_id ASC LIMIT 1"
    )
    top = _scalar(run(s1))
    s2 = f"SELECT category FROM products WHERE product_id = {_lit(top)}"
    category = _scalar(run(s2))
    instruction = (
        "Find the product with the highest total quantity sold across all "
        "orders, then return that product's category."
    )
    return instruction, [s1, s2], category


# --------------------------------------------------------------------------- #
# Logistics templates
# --------------------------------------------------------------------------- #


def _logi_busiest_warehouse_delayed(schema: Schema, run) -> tuple[str, list[str], Any]:
    s1 = (
        "SELECT warehouse_id FROM shipments GROUP BY warehouse_id "
        "ORDER BY COUNT(*) DESC, warehouse_id ASC LIMIT 1"
    )
    wid = _scalar(run(s1))
    s2 = f"SELECT COUNT(*) FROM shipments WHERE warehouse_id = {_lit(wid)} AND status = 'delayed'"
    count = _scalar(run(s2))
    instruction = (
        "Find the warehouse that has handled the most shipments, then return "
        "how many of that warehouse's shipments are delayed."
    )
    return instruction, [s1, s2], count


def _logi_heaviest_shipment_carrier(schema: Schema, run) -> tuple[str, list[str], Any]:
    s1 = "SELECT shipment_id FROM shipments ORDER BY weight DESC, shipment_id ASC LIMIT 1"
    sid = _scalar(run(s1))
    s2 = f"SELECT carrier FROM shipments WHERE shipment_id = {_lit(sid)}"
    carrier = _scalar(run(s2))
    instruction = "Find the single heaviest shipment, then return its carrier."
    return instruction, [s1, s2], carrier


def _logi_most_vehicles_city(schema: Schema, run) -> tuple[str, list[str], Any]:
    s1 = (
        "SELECT warehouse_id FROM vehicles GROUP BY warehouse_id "
        "ORDER BY COUNT(*) DESC, warehouse_id ASC LIMIT 1"
    )
    wid = _scalar(run(s1))
    s2 = f"SELECT city FROM warehouses WHERE warehouse_id = {_lit(wid)}"
    city = _scalar(run(s2))
    instruction = (
        "Find the warehouse with the most vehicles, then return the city where "
        "that warehouse is located."
    )
    return instruction, [s1, s2], city


# --------------------------------------------------------------------------- #
# HR templates
# --------------------------------------------------------------------------- #


def _hr_biggest_dept_top_salary(schema: Schema, run) -> tuple[str, list[str], Any]:
    s1 = (
        "SELECT department_id FROM employees GROUP BY department_id "
        "ORDER BY COUNT(*) DESC, department_id ASC LIMIT 1"
    )
    did = _scalar(run(s1))
    s2 = (
        f"SELECT employee_id FROM employees WHERE department_id = {_lit(did)} "
        "ORDER BY salary DESC, employee_id ASC LIMIT 1"
    )
    emp = _scalar(run(s2))
    instruction = (
        "Find the department with the most employees, then return the employee "
        "ID of the highest-paid employee in that department."
    )
    return instruction, [s1, s2], emp


def _hr_longest_tenure_department(schema: Schema, run) -> tuple[str, list[str], Any]:
    s1 = "SELECT employee_id FROM employees ORDER BY hire_date ASC, employee_id ASC LIMIT 1"
    emp = _scalar(run(s1))
    s2 = (
        "SELECT d.name FROM employees e JOIN departments d "
        f"ON e.department_id = d.department_id WHERE e.employee_id = {_lit(emp)}"
    )
    dept = _scalar(run(s2))
    instruction = (
        "Find the employee with the longest tenure (earliest hire date), then "
        "return the name of the department they work in."
    )
    return instruction, [s1, s2], dept


def _hr_highest_paid_department(schema: Schema, run) -> tuple[str, list[str], Any]:
    s1 = "SELECT employee_id FROM employees ORDER BY salary DESC, employee_id ASC LIMIT 1"
    emp = _scalar(run(s1))
    s2 = (
        "SELECT d.name FROM employees e JOIN departments d "
        f"ON e.department_id = d.department_id WHERE e.employee_id = {_lit(emp)}"
    )
    dept = _scalar(run(s2))
    instruction = (
        "Find the highest-paid employee in the company, then return the name of "
        "their department."
    )
    return instruction, [s1, s2], dept


_TEMPLATES: dict[str, list[Template]] = {
    "ecommerce": [
        _ecom_top_customer_recent_order,
        _ecom_top_customer_recent_order_status,
        _ecom_most_orders_region,
        _ecom_top_product_category,
    ],
    "logistics": [
        _logi_busiest_warehouse_delayed,
        _logi_heaviest_shipment_carrier,
        _logi_most_vehicles_city,
    ],
    "hr": [
        _hr_biggest_dept_top_salary,
        _hr_longest_tenure_department,
        _hr_highest_paid_department,
    ],
}


class TaskGenerator:
    """Generates multi-step tasks with computed ground truth for a schema."""

    def __init__(self, backend: Backend | None = None):
        self.backend = backend if backend is not None else SQLiteBackend()

    def generate_tasks(
        self, schema: Schema, n: int = 1, seed: int | None = None
    ) -> list[Task]:
        if schema.domain not in _TEMPLATES:
            raise ValueError(f"no task templates for domain {schema.domain!r}")
        templates = _TEMPLATES[schema.domain]
        rng = random.Random(seed if seed is not None else schema.seed)

        namespace = f"taskgen_{schema.name}"
        handle: BackendHandle = self.backend.spawn(schema, namespace)

        def run(sql: str) -> list[tuple]:
            res = self.backend.execute(handle, sql)
            if res.status != "ok":
                raise RuntimeError(f"gold SQL failed: {res.error}\n{sql}")
            return res.rows

        tasks: list[Task] = []
        try:
            for i in range(n):
                template = templates[i % len(templates)]
                instruction, gold_sql, answer = template(schema, run)
                tasks.append(
                    Task(
                        task_id=f"{schema.name}_{template.__name__}_{i}",
                        schema_name=schema.name,
                        domain=schema.domain,
                        instruction=instruction,
                        ground_truth=answer,
                        min_steps=len(gold_sql),
                        seed=schema.seed,
                        gold_sql=gold_sql,
                    )
                )
        finally:
            self.backend.teardown(handle)

        return tasks
