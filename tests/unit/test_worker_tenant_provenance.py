"""The detached render worker must carry the enqueue request's tenant PROVENANCE.

WHY THIS FILE EXISTS. Every LLM call this service makes goes through
``WMClient``, which bakes ``X-Tenant-Id: Bearer <DOWNSTREAM_LICENSE_TOKEN>``,
and ``worker_management`` is a metered destination — so each call is a
*delegated* hop. Since w2v_common 0.67.0, ``_apply_outbound_tenant`` refuses to
delegate (``UnvouchedTenantError``) unless the ``tenant_source`` ContextVar
holds a member of ``VOUCHED_TENANT_SOURCES``.

``set_identity_context(tenant_source=...)`` defaults to ``None``, so dropping the
provenance is NOT a type error and ships perfectly green — it only shows up in
production as every background render dying at the WM call, or (on a tolerant
lane) as cost attributed to nobody. A sibling service shipped exactly that.

The detached worker loop inherits no ContextVars, so the ONLY place the source
can be read is ``_capture_identity`` at enqueue time. These tests pin both ends
of that thread and the fail-closed behaviour in between.
"""

from __future__ import annotations

import pytest
from w2v_common.auth import (
    clear_identity_context,
    get_tenant_source_from_context,
    set_identity_context,
)
from w2v_common.tenant import VOUCHED_TENANT_SOURCES, TenantSource

from src.services.job_worker import _capture_identity, _restore_tenant_source
from tests.tenants import TENANT_1


@pytest.fixture(autouse=True)
def _clean_identity():
    clear_identity_context()
    yield
    clear_identity_context()


class TestCaptureCarriesProvenance:
    @pytest.mark.parametrize("source", sorted(VOUCHED_TENANT_SOURCES, key=str))
    def test_a_vouched_source_is_snapshotted(self, source: TenantSource) -> None:
        set_identity_context(TENANT_1, "lic", "sub", tenant_source=source)
        assert _capture_identity()["tenant_source"] == source.value

    def test_an_unvouched_source_is_snapshotted_verbatim_not_upgraded(self) -> None:
        """The snapshot records what happened. Promoting an unvouched source to a
        vouched one here would forge the vouch — the whole point of the ladder."""
        set_identity_context(TENANT_1, "lic", "sub", tenant_source=TenantSource.UNVOUCHED_HEADER)
        captured = _capture_identity()
        assert captured["tenant_source"] == TenantSource.UNVOUCHED_HEADER.value
        assert TenantSource(captured["tenant_source"]) not in VOUCHED_TENANT_SOURCES

    def test_no_identity_snapshots_an_empty_source_not_a_default(self) -> None:
        captured = _capture_identity()
        assert captured["tenant_source"] == ""
        assert captured["tenant_id"] == ""

    def test_the_whole_snapshot_stays_json_serialisable(self) -> None:
        """job_options is jsonb: a bare TenantSource/UUID would fail on INSERT."""
        import json

        set_identity_context(TENANT_1, "lic", "sub", tenant_source=TenantSource.LICENSE_TOKEN)
        json.dumps(_capture_identity())  # must not raise


class TestRestoreFailsClosed:
    @pytest.mark.parametrize("source", list(TenantSource))
    def test_every_known_member_round_trips(self, source: TenantSource) -> None:
        assert _restore_tenant_source(source.value) is source

    @pytest.mark.parametrize("bad", ["", None, "not-a-source", 7, {"a": 1}])
    def test_absent_or_unknown_degrades_to_none_never_to_a_vouched_default(
        self, bad: object
    ) -> None:
        """A pre-upgrade job has no persisted source. ``None`` is unvouched, so the
        delegation gate refuses it — the correct, loud outcome. Substituting
        LICENSE_TOKEN to 'keep old jobs working' would let this process spend an
        arbitrary tenant's credentials on provenance it cannot stand behind."""
        restored = _restore_tenant_source(bad)
        assert restored is None
        assert restored not in VOUCHED_TENANT_SOURCES


class TestRoundTripThroughSetIdentityContext:
    def test_the_worker_re_establishes_a_delegatable_provenance(self) -> None:
        """End to end: enqueue-scope source -> job_options -> worker context."""
        set_identity_context(TENANT_1, "lic", "sub", tenant_source=TenantSource.LICENSE_TOKEN)
        ident = _capture_identity()
        clear_identity_context()
        assert get_tenant_source_from_context() is None, "worker starts with no context"

        set_identity_context(
            tenant_id=TENANT_1,
            license_id=ident["license_id"],
            subject_id=ident["subject_id"],
            tenant_source=_restore_tenant_source(ident["tenant_source"]),
        )
        restored = get_tenant_source_from_context()
        assert restored is TenantSource.LICENSE_TOKEN
        assert restored in VOUCHED_TENANT_SOURCES, (
            "if this is not vouched, every WM call from the render worker raises "
            "UnvouchedTenantError"
        )
