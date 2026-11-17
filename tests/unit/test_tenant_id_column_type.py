"""Pin the ``tenant_id`` column type: native Postgres ``uuid``, Python-side ``str``.

Two independent facts are asserted, because getting either one wrong is silent:

* **The DDL type must be ``uuid``.** That is the fleet ruling and the whole point of
  migration ``0010_tenant_id_uuid``. A regression to ``String(255)`` would compile,
  pass every other test, and re-open the door to storing a tenant NAME.
* **``as_uuid`` must be False.** This is the load-bearing half. With
  ``as_uuid=True`` SQLAlchemy returns ``uuid.UUID`` objects, and
  ``uuid.UUID(t) == t`` is ``False`` for every ``str`` ``t``. This service compares
  tenant strings in ~20 places — ``_get_tenant_id`` returns the canonical
  ``str(uuid)`` and it is threaded through the ``JobManager`` / ``StyleManager`` /
  ``BatchManager`` scope filters, the ``(tenant_id, slug)`` style-description cache
  key, and the ``ClaimedJob`` snapshot the worker re-binds identity from. None of
  those is a TYPE error, so ``ty`` and ``ruff`` stay green while ownership guards
  404 the owner on their own rows and scope predicates match nothing.

The second assertion is not theoretical: it reproduces the exact comparison the
service performs, so the test fails for the same reason production would.
"""

from __future__ import annotations

import uuid

import pytest
from sqlalchemy import Uuid
from sqlalchemy.dialects import postgresql

from src.database.models import Base, BatchJob, CustomStyle, RedesignJob, RedesignUsage

#: Every model carrying a tenant. If a fifth is added, it belongs here too.
TENANT_MODELS: tuple[type[Base], ...] = (RedesignJob, RedesignUsage, CustomStyle, BatchJob)


@pytest.mark.parametrize("model", TENANT_MODELS, ids=lambda m: m.__tablename__)
def test_tenant_id_is_a_native_uuid_column(model: type[Base]) -> None:
    """The DDL emitted for ``tenant_id`` is ``UUID``, never ``VARCHAR``."""
    column = model.__table__.columns["tenant_id"]
    assert isinstance(column.type, Uuid), (
        f"{model.__tablename__}.tenant_id is {column.type!r}, not a uuid type. "
        "Migration 0010_tenant_id_uuid made this column a native Postgres uuid; "
        "reverting the model to String would silently diverge create_all-built test "
        "databases from prod and re-allow storing a tenant NAME."
    )
    compiled = column.type.compile(dialect=postgresql.dialect())
    assert compiled == "UUID", f"{model.__tablename__}.tenant_id compiles to {compiled}, not UUID"


@pytest.mark.parametrize("model", TENANT_MODELS, ids=lambda m: m.__tablename__)
def test_tenant_id_is_returned_as_str_not_uuid_object(model: type[Base]) -> None:
    """``as_uuid=False`` — the Python-side value stays a ``str``.

    ``as_uuid`` is the flag SQLAlchemy actually branches on when building the result
    processor, so reading it is reading the behaviour, not a spelling. It is checked
    here rather than round-tripped through a live connection deliberately: this must
    stay a unit test that runs with no Postgres, and the live round-trip is already
    proven by the migration verification (the claim resolver returns a real ``uuid``
    column and the ORM reads it back as a ``str``).
    """
    column_type = model.__table__.columns["tenant_id"].type
    assert isinstance(column_type, Uuid), (
        f"{model.__tablename__}.tenant_id is {column_type!r}, not a uuid type"
    )
    assert column_type.as_uuid is False, (
        f"{model.__tablename__}.tenant_id uses as_uuid=True. That returns uuid.UUID "
        "objects, and `uuid.UUID(t) == t` is False for every str t — every "
        "`== tenant_id` comparison in this service would silently become False "
        "without any type checker noticing. See models.TENANT_ID_TYPE."
    )


def test_uuid_object_does_not_equal_its_own_string_form() -> None:
    """The premise of the choice above, asserted rather than assumed.

    If this ever stops holding, the ``as_uuid=False`` pin can be revisited — until
    then it is the only thing standing between a green test suite and a service
    that 404s every owner on their own rows.
    """
    tenant = "019de842-c92a-7e37-8fdd-eb0d8c5396f9"
    assert uuid.UUID(tenant) != tenant
    assert str(uuid.UUID(tenant)) == tenant
