"""SECURITY DEFINER claim resolver for the cross-tenant lease queue (single source).

The durable job queue is drained by a worker pool that claims the next runnable
``redesign_jobs`` row across ALL tenants (``FOR UPDATE SKIP LOCKED``). Under Phase D
RLS enforce the runtime is a non-bypass role with no ``app.current_tenant`` bound, so
the claim SELECT would fail-closed to 0 rows and NOTHING would ever render. The claim
therefore runs through this SECURITY DEFINER function (owned by the table owner →
bypasses ENABLE), which atomically does the exact same thing ``JobManager.claim_next_job``
did in Python: pick one queue-owned pending/expired-lease row, fail poison jobs past
``max_attempts``, else move it to ``processing`` with a fresh lease + incremented
attempt, and return ``(job_id, tenant_id, job_options)``. The worker then binds the
returned tenant for all subsequent per-job DB work (see rls_events.after_begin).

EXECUTE is REVOKEd from PUBLIC here and GRANTed to the app role out-of-band in Stage A
(the app role does not exist in CI). Single-sourced so BOTH the migration (prod) and
the ``create_all`` ``after_create`` event (tests) create an identical function.

``tenant_id`` IS DECLARED ``uuid``, AND THAT DECLARATION IS THE WHOLE POINT OF
``0010_tenant_id_uuid`` TOUCHING THIS FILE. ``ALTER COLUMN ... TYPE`` does NOT
revalidate function bodies, so a resolver left declaring ``tenant_id varchar`` keeps
existing after the flip and then dies on every single call with "structure of query
does not match function result type" — which is the claim loop, i.e. NOTHING renders,
with no schema error to point at. ``CREATE OR REPLACE`` cannot change a return type
either, so the migration DROPs and re-creates it. The Python seam
(``JobManager.claim_next_job``) reads this through a raw ``text()`` query, where
psycopg hands back a ``uuid.UUID`` rather than a ``str``; it coerces with ``str(...)``
there, deliberately, so ``ClaimedJob.tenant_id`` stays the ``str`` every downstream
comparison expects.
"""

from __future__ import annotations

from sqlalchemy import Connection, MetaData, event, text

CLAIM_FN_SQL = """
CREATE OR REPLACE FUNCTION claim_next_render_job(
    p_worker_id text, p_lease_seconds int, p_max_attempts int
) RETURNS TABLE(job_id varchar, tenant_id uuid, job_options jsonb)
  LANGUAGE plpgsql SECURITY DEFINER SET search_path FROM CURRENT AS $fn$
  DECLARE
    v_id bigint; v_job_id varchar; v_tenant uuid; v_options jsonb; v_attempts int;
  BEGIN
    SELECT rj.id, rj.job_id, rj.tenant_id, rj.job_options, rj.attempts
      INTO v_id, v_job_id, v_tenant, v_options, v_attempts
    FROM redesign_jobs rj
    WHERE rj.job_options IS NOT NULL
      AND (
        rj.status = 'pending'
        OR (rj.status = 'processing'
            AND (rj.lease_expires_at IS NULL OR rj.lease_expires_at < now()))
      )
    ORDER BY rj.created_at
    LIMIT 1
    FOR UPDATE SKIP LOCKED;

    IF v_id IS NULL THEN
      RETURN;  -- empty queue
    END IF;

    IF COALESCE(v_attempts, 0) >= p_max_attempts THEN
      UPDATE redesign_jobs
        SET status = 'failed',
            error_message = 'exceeded max claim attempts (' || p_max_attempts
                            || ') — failed as poison job',
            completed_at = now()
      WHERE id = v_id;
      -- The parent's verdict is terminal, so its unfinished variants are too.
      -- Aliased, and every column qualified: the function's OUT parameters are
      -- named ``job_id``/``tenant_id``, so a bare ``job_id`` here is ambiguous
      -- between the PL/pgSQL variable and the table column and the whole claim
      -- errors out — i.e. the queue stops draining.
      UPDATE redesign_variants rv
        SET status = 'failed',
            error_message = 'parent job failed as poison before this variant finished',
            completed_at = now()
      WHERE rv.job_id = v_job_id AND rv.status IN ('pending', 'processing');
      RETURN;  -- poison job skipped
    END IF;

    -- RE-CLAIM: retire the previous attempt's unfinished variants.
    --
    -- ``create_variant`` only ever INSERTs, and a re-claim replays the whole
    -- render from scratch, so the rows the interrupted attempt left behind are
    -- never written again by anyone: no code path resets or completes them, and
    -- ``get_variants`` returns every row for the job. Without this the variant
    -- list shows a permanent ``pending`` alongside the new attempt's rows — the
    -- same "no reset path" defect as an unreaped batch, one table down.
    --
    -- Safe to run unconditionally: on a FIRST claim the job has no variants yet
    -- (they are created inside the dispatch), so this matches zero rows.
    UPDATE redesign_variants rv
      SET status = 'failed',
          error_message = 'superseded: the parent job was re-claimed after an '
                          'interrupted attempt and re-runs from scratch',
          completed_at = now()
    WHERE rv.job_id = v_job_id AND rv.status IN ('pending', 'processing');

    UPDATE redesign_jobs
      SET status = 'processing',
          claimed_by = p_worker_id,
          started_at = COALESCE(started_at, now()),
          lease_expires_at = now() + make_interval(secs => p_lease_seconds),
          attempts = COALESCE(attempts, 0) + 1
    WHERE id = v_id;

    RETURN QUERY SELECT v_job_id, v_tenant, v_options;
  END
  $fn$
"""


#: Stale-batch reaper — the SAME lease sweep, extended to the one table that was
#: left out of it.
#:
#: ``batch_jobs`` has no lease columns and is never claimed: a batch is driven by
#: an in-process ``asyncio.create_task`` (``_process_batch_background``). So a pod
#: restart, an OOMKill, or any exception outside that task's own ``except`` leaves
#: the row ``pending``/``processing`` FOREVER — nothing advances it (only
#: ``BatchManager`` does, from the task that is gone), nothing reclaims it (the
#: lease reclaim in ``0002_job_lease`` is ``redesign_jobs``-only), and nothing
#: prunes it (``CleanupService`` has no caller and ``RETENTION_ENABLED`` defaults
#: false). The batch status endpoint is then wedged for that client permanently.
#:
#: This RECLAIMS rather than deletes: the row reaches a terminal ``failed`` with a
#: recorded reason, so the client gets an answer and the history survives. Batches
#: are not re-runnable from a persisted bundle the way queue jobs are — there is no
#: ``job_options`` equivalent on ``batch_jobs`` — so "terminal with a reason" is the
#: honest reclaim here, not a re-dispatch.
#:
#: SECURITY DEFINER for exactly the reason ``claim_next_render_job`` is: the sweep
#: is cross-tenant and the worker binds no ``app.current_tenant``, so under Phase D
#: RLS enforce a plain UPDATE would fail closed to 0 rows and the reaper would
#: silently do nothing in production while passing every test.
#:
#: Staleness is a NO-PROGRESS WINDOW, not a wall-clock ceiling on the batch. It
#: is measured from ``COALESCE(last_progress_at, started_at, created_at)``, and
#: ``last_progress_at`` is bumped by ``increment_completed``/``increment_failed``
#: on every committed page outcome.
#:
#: A pure wall-clock ceiling was WRONG and provably reaped healthy work.
#: ``started_at`` is written once by ``mark_processing`` and never again, so an
#: age measured from it is the age of the whole batch — and the longest
#: legitimate batch here is 20 pages (``BatchRedesignRequest.pages`` max_length)
#: at ``batch_max_concurrency`` 3 = 7 sequential waves, with NO wall-clock cap on
#: an in-process batch child at all (``max_job_lifetime_seconds`` bounds only the
#: queue path). Any ceiling had to be a guess above that product, and a guess
#: below it kills a live render. The progress clock removes the guess: the window
#: only has to exceed the longest single PAGE, so it no longer scales with how
#: many pages the client sent.
#:
#: Unlike a queue job — where an unstarted ``pending`` row is merely backlogged
#: and MUST NOT be aged out — an unstarted batch is owned by nobody: if its task
#: died before ``mark_processing``, no worker will ever pick it up, so ``pending``
#: is just as dead as ``processing``. Hence ``created_at`` as the last fallback.
#:
#: Child rows: only ``job_options IS NULL`` children are touched. A queue-owned
#: child belongs to the worker pool and its own lease/poison path must remain the
#: only thing that ends it. Variants are retired before their parents so the
#: subquery still sees the pre-update child status.
REAP_BATCH_FN_SQL = """
CREATE OR REPLACE FUNCTION reap_stale_batch_jobs(p_max_lifetime_seconds int)
  RETURNS integer
  LANGUAGE plpgsql SECURITY DEFINER SET search_path FROM CURRENT AS $fn$
  DECLARE
    v_batch_ids varchar[];
    v_reason text;
    v_count int;
    v_secs int;
    -- Hard floor on the no-progress window, enforced HERE rather than only in
    -- the caller's config. This function is SECURITY DEFINER: it bypasses RLS
    -- and terminates work across every tenant, and the restricted app role can
    -- call it with any argument it likes. A call with ``1`` would fail every
    -- in-flight batch in the fleet in one statement. Config is a preference; a
    -- privileged function that ends other tenants' work needs its own bound.
    --
    -- 2100 is DERIVED, not picked, and the derivation was WRONG once already.
    -- The clock is bumped after every SECTION, so the longest legitimate gap is
    -- the one BEFORE the first section: job.started webhook (<=~60s) + page
    -- fetch (browser lease + 30s goto) + the first WM call. One WM call is
    -- 3 attempts x 600s + 1.0s + 2.0s backoff = 1803s (w2v_common
    -- ``_MAX_RETRIES``/``_RETRY_DELAY``; a read timeout IS retried), giving
    -- ~1893s. A previous revision put this floor at 1800 off a 1202s figure
    -- computed from ``wm.max_retries``/``wm.retry_delay`` — two config fields
    -- nothing ever read, now deleted. That floor sat BELOW the single gap it
    -- exists to bound: harmless at the shipped default, live the moment an
    -- operator lowered REDESIGN_BATCH_NO_PROGRESS_SECONDS.
    --
    -- 2100 clears 1893 with margin and stays below the 3600s shipped default.
    --
    -- It is emphatically NOT ``max_job_lifetime_seconds``: that bounds only the
    -- QUEUE path. An in-process batch child has no wall-clock cap at all, so a
    -- page legitimately runs longer than the queue path would ever allow a job
    -- to — which is exactly why the reaper measures PROGRESS and not age.
    --
    -- Clamps UP rather than raising: the safe direction here is a LONGER window
    -- (a late reap wedges a status endpoint; an early reap kills paid work), so
    -- refusing the call outright would leave the abandoned batches immortal
    -- again. It is loud about it.
    c_min_seconds constant int := 2100;
  BEGIN
    IF p_max_lifetime_seconds <= 0 THEN
      RETURN 0;  -- reaper disabled
    END IF;

    v_secs := GREATEST(p_max_lifetime_seconds, c_min_seconds);
    IF v_secs <> p_max_lifetime_seconds THEN
      RAISE WARNING 'reap_stale_batch_jobs: no-progress window % s is below the % s '
                    'floor and would reap live batches; clamped up to % s',
                    p_max_lifetime_seconds, c_min_seconds, v_secs;
    END IF;

    v_reason := 'batch abandoned: no page completed or failed for ' || v_secs
                || 's — its in-process orchestrator died (pod restart or an unhandled '
                || 'error) and nothing else advances a batch. Reaped by the lease sweep.';

    SELECT array_agg(s.batch_id) INTO v_batch_ids FROM (
      SELECT batch_id
        FROM batch_jobs
       WHERE status IN ('pending', 'processing')
         AND COALESCE(last_progress_at, started_at, created_at)
             < now() - make_interval(secs => v_secs)
       ORDER BY created_at
       FOR UPDATE SKIP LOCKED
    ) s;

    IF v_batch_ids IS NULL THEN
      RETURN 0;
    END IF;

    UPDATE batch_jobs
      SET status = 'failed', error_message = v_reason, completed_at = now()
    WHERE batch_id = ANY(v_batch_ids);
    GET DIAGNOSTICS v_count = ROW_COUNT;

    UPDATE redesign_variants v
      SET status = 'failed',
          error_message = 'parent batch was reaped as abandoned',
          completed_at = now()
    WHERE v.status IN ('pending', 'processing')
      AND v.job_id IN (
        SELECT j.job_id FROM redesign_jobs j
         WHERE j.batch_id = ANY(v_batch_ids)
           AND j.job_options IS NULL
           AND j.status IN ('pending', 'processing')
      );

    UPDATE redesign_jobs j
      SET status = 'failed', error_message = v_reason, completed_at = now()
    WHERE j.batch_id = ANY(v_batch_ids)
      AND j.job_options IS NULL
      AND j.status IN ('pending', 'processing');

    RETURN v_count;
  END
  $fn$
"""

#: Same ACL treatment as the claim resolver: not world-executable, granted to the
#: restricted runtime role when that role exists (it does not in CI).
REVOKE_REAP_BATCH_FN = "REVOKE EXECUTE ON FUNCTION reap_stale_batch_jobs(int) FROM PUBLIC"

GRANT_REAP_BATCH_FN = """
DO $$
BEGIN
  IF EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'w2v_html_render_app') THEN
    GRANT EXECUTE ON FUNCTION reap_stale_batch_jobs(int) TO w2v_html_render_app;
  END IF;
END $$
"""


def register_claim_fn(metadata: MetaData) -> None:
    """Create the claim + batch-reaper resolvers after the tables exist (test path)."""

    def _apply(target: object, connection: Connection, **kw: object) -> None:
        connection.execute(text(CLAIM_FN_SQL))
        connection.execute(text(REAP_BATCH_FN_SQL))

    event.listen(metadata, "after_create", _apply)
