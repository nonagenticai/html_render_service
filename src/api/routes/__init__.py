"""Redesign routes package.

Composes the redesign APIRouter from focused submodules and re-exports the
public symbols (route handlers, helpers, JobManager/StyleManager, background
tasks) so that ``from src.domains.redesign.routes import X`` and patches like
``src.domains.redesign.routes.JobManager`` continue to work for tests and
other code that originally imported from the monolithic ``routes.py`` module.

All redesign route paths have distinct literal-or-suffix structures (no
``/jobs/literal`` collides with ``/jobs/{job_id}``), so submodule registration
order does not affect FastAPI matching. Order below mirrors the original
declaration order for readability.
"""

from __future__ import annotations

import asyncio
import logging

from fastapi import APIRouter
from w2v_common.auth import LICENSE_GATE

# JobManager and StyleManager are imported at the package level so test
# patches such as ``patch("src.domains.redesign.routes.JobManager...")`` work
# even though the actual implementations live in submodules.
from src.services.job_manager import JobManager
from src.services.style_manager import StyleManager

from . import (
    artifacts,
    batch,
    jobs,
    refinement,
    styles,
    variants,
    webhooks,
)
from ._helpers import (
    _background_tasks,
    _build_visual_assets,
    _concept_image_assets_from_payload,
    _get_redesign_service,
    _get_session_factory,
    _get_storage,
    _get_tenant_id,
    _preferred_concept_image_url,
    _process_batch_background,
    _process_job_background,
    _process_refinement_background,
)
from .artifacts import (
    download_comparison,
    download_component_html,
    download_diff,
    download_html,
    download_original_screenshot,
    download_screenshot,
)
from .batch import (
    cancel_batch,
    create_batch,
    get_batch_results,
    get_batch_status,
    list_batches,
    retry_failed_batch_pages,
)
from .jobs import (
    cancel_job,
    create_redesign,
    get_job_result,
    get_job_status,
    list_jobs,
)
from .refinement import get_iteration_history, refine_redesign
from .styles import (
    create_style,
    delete_style,
    get_style,
    list_styles,
    preview_style,
    update_style,
)
from .variants import compare_variants, download_variant_html
from .webhooks import get_webhook_deliveries

logger = logging.getLogger(__name__)

# Canonical surface: /api/designs (any web design — see scope vocabulary).
#
# ENTITLEMENT GATE (w2v_common.auth.entitlement, >=0.75.0), mounted router-wide.
# Until now this service authenticated the licence token for tenant IDENTITY
# (AuthMiddleware answers "who are you") and never asked "may you" — a tenant
# whose licence is expired or revoked still carries a syntactically valid token
# and was served. LICENSE_GATE closes that: none/expired/revoked -> 402
# license_<state>, an unestablishable verdict -> 503 license_unavailable +
# Retry-After, active/expiring_soon -> pass.
#
# EVERY route under this prefix is gated, and that is correct: /api/designs is
# the entire tenant-scoped surface — job creation, job/batch reads, artifact
# downloads (HTML/PDF/screenshots/diffs), refinement, variants, custom styles
# and the webhook DELIVERY LOG (an outbound-delivery read for the tenant, not an
# inbound callback). There is no internal-callback or peer-to-peer route here,
# so nothing under /api/designs has a reason to be exempt.
#
# DELIBERATELY NOT GATED, each for a stated reason:
#   * the health router (/health, /health/live, /health/ready and aliases),
#     mounted separately in asgi.py. A 402/503 on a probe CrashLoops the pod and
#     an unlicensed tenant cannot be billed by killing our own deployment. The
#     gate refuses these by path too (is_license_exempt_path), so this is belt
#     and braces, not a single point of failure.
#   * /metrics and /docs|/openapi.json, added by w2v_common's create_app: a
#     Prometheus scrape and the schema carry no tenant and do no tenant work.
#
# DORMANT ON MERGE: require_license is a no-op unless W2V_LICENSE_GATE_ENABLED
# is true, read per-call from os.environ. It defaults to False, so this change
# alters no behaviour until gitops arms the flag. Preserve that default.
router = APIRouter(prefix="/api/designs", tags=["designs"], dependencies=list(LICENSE_GATE))

# POST "" is registered directly because FastAPI rejects empty path +
# empty subrouter prefix during include_router.
router.add_api_route(
    "",
    create_redesign,
    methods=["POST"],
    operation_id="create_design",
    responses={422: {"description": "Validation error"}},
)
router.include_router(jobs.router)  # /jobs/{job_id}, /jobs/{job_id}/result, cancel, list
router.include_router(artifacts.router)
router.include_router(variants.router)
router.include_router(refinement.router)
router.include_router(webhooks.router)
router.include_router(styles.router)
router.include_router(batch.router)


__all__ = [
    "JobManager",
    "StyleManager",
    "_background_tasks",
    "_build_visual_assets",
    "_concept_image_assets_from_payload",
    "_get_redesign_service",
    "_get_session_factory",
    "_get_storage",
    "_get_tenant_id",
    "_preferred_concept_image_url",
    "_process_batch_background",
    "_process_job_background",
    "_process_refinement_background",
    "asyncio",
    "cancel_batch",
    "cancel_job",
    "compare_variants",
    "create_batch",
    "create_redesign",
    "create_style",
    "delete_style",
    "download_comparison",
    "download_component_html",
    "download_diff",
    "download_html",
    "download_original_screenshot",
    "download_screenshot",
    "download_variant_html",
    "get_batch_results",
    "get_batch_status",
    "get_iteration_history",
    "get_job_result",
    "get_job_status",
    "get_style",
    "get_webhook_deliveries",
    "list_batches",
    "list_jobs",
    "list_styles",
    "logger",
    "preview_style",
    "refine_redesign",
    "retry_failed_batch_pages",
    "router",
    "update_style",
]
