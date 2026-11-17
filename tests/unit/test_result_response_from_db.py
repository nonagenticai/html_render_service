"""``GET /jobs/{id}/result`` serves from a single builder off the authoritative
DB row.

There used to be a second, hand-maintained serializer on the Redis result-cache
hit path. It silently diverged from the DB serialization — it dropped the
completion-gating ``verification_status``/``verification_report`` and defaulted
``style`` to "modern" — so once a completed job's result was cached, ``/result``
returned ``verification_status=null`` and the orchestrator's web_design_v2 verify
stage failed the whole run (``W2V_WEB_DESIGN_REQUIRE_VERIFICATION`` — refusing to
complete an unverified design). The result cache was removed entirely (it was a
once-per-job endpoint whose cache was never invalidated); there is now exactly
one serialization, so cache and DB cannot drift. This asserts the gating fields
come straight from the row.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from src.api.routes.jobs import get_job_result
from src.services.job_manager import JobManager
from tests.tenants import TENANT_1


@pytest.mark.asyncio
async def test_result_sources_gating_fields_from_db(db_session) -> None:
    job = await JobManager.create_job(
        db_session, tenant_id=TENANT_1, url="", scope="landing", style="assembly"
    )
    await JobManager.set_verification(
        db_session, job.job_id, "passed", {"status": "passed", "findings": []}
    )
    await JobManager.mark_completed(db_session, job.job_id, redesigned_html_path="/x/redesign.html")
    await db_session.commit()

    request = MagicMock()
    request.state.tenant_id = TENANT_1
    request.app.state.storage_service = MagicMock()

    resp = await get_job_result(job.job_id, request, db=db_session)

    # The completion-gating fields and style come from the DB row, verbatim.
    assert resp.verification_status == "passed"
    assert resp.verification_report == {"status": "passed", "findings": []}
    assert resp.style == "assembly"
    assert resp.redesigned_html_path == "/x/redesign.html"


def _result_request() -> MagicMock:
    request = MagicMock()
    request.state.tenant_id = TENANT_1
    request.app.state.storage_service = MagicMock()
    return request


@pytest.mark.asyncio
async def test_result_surfaces_concept_image_degraded(db_session) -> None:
    """A requested-but-failed concept-image generation is flagged, job still completed."""
    job = await JobManager.create_job(
        db_session, tenant_id=TENANT_1, url="", scope="landing", style="modern"
    )
    await JobManager.mark_completed(
        db_session,
        job.job_id,
        redesigned_html_path="/x/redesign.html",
        concept_image_degraded=True,
        concept_image_reason="submit failed: boom",
    )
    await db_session.commit()

    resp = await get_job_result(job.job_id, _result_request(), db=db_session)

    # Core job completed; concept-image failure is distinguishable from "none requested".
    assert resp.concept_image_degraded is True
    assert resp.concept_image_error == "submit failed: boom"


@pytest.mark.asyncio
async def test_result_not_degraded_when_concept_images_ok(db_session) -> None:
    """No concept-image failure → no degraded flag, no error (the common case)."""
    job = await JobManager.create_job(
        db_session, tenant_id=TENANT_1, url="", scope="landing", style="modern"
    )
    await JobManager.mark_completed(db_session, job.job_id, redesigned_html_path="/x/redesign.html")
    await db_session.commit()

    resp = await get_job_result(job.job_id, _result_request(), db=db_session)

    assert resp.concept_image_degraded is False
    assert resp.concept_image_error is None
