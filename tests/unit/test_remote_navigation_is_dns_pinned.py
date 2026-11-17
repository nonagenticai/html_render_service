"""Fleet property: a browser never navigates a REMOTE host it did not DNS-pin.

The recurring defect shape in this repo is "guarded beside unguarded": one sink
gets the fix, its sibling keeps the hole. It happened here exactly once more.

``screenshot_renderer.capture_url_screenshots`` did it right — it captures the
address ``validate_public_url`` vetted and leases a resolver-pinned browser
(``--host-resolver-rules=MAP <host> <ip>``), so Chromium cannot re-resolve.
``design_builder.helpers._fetch_page_html`` called the *same*
``validate_public_url`` and **threw the address away**, then leased the shared
pool. Both read ``url`` from the same ``redesign_jobs.url`` column, so the same
tenant input reached both — and on the unpinned one, a hostname that answers
public at validation time and private milliseconds later at ``goto`` time got
its INTERNAL html fetched, cached under ``cache_html`` for 3600s, and fed to the
redesign model. Validate-then-resolve-again is a TOCTOU; the pin closes it.

So pin the PROPERTY, not the two call sites: AST-walk all of ``src/`` and assert
every function that navigates a browser at a non-local document leases pinned.
Regex was rejected deliberately (a regex that matches zero lines is a test that
proves nothing) — this is the same shape as
``test_pdf_render_subresource_ssrf.py``'s local-document walk, which is the test
that caught a fifth ``file://`` sink spelled ``page.goto(tmp_path.as_uri())``.
"""

from __future__ import annotations

import ast
import pathlib

SRC_ROOT = pathlib.Path(__file__).resolve().parents[2] / "src"


def _is_local_document(arg_src: str, func: ast.AST) -> bool:
    """True if a ``goto`` argument names a LOCAL document, not a remote host.

    A ``file://`` navigation has no hostname for DNS to rebind, so pinning is
    meaningless there; those sinks are covered by the *subresource* property in
    ``test_pdf_render_subresource_ssrf.py`` instead. Resolve one level of local
    indirection (``file_url = f"file://{...}"``) the same way that test does.
    """
    if "file://" in arg_src or "as_uri" in arg_src:
        return True
    return any(
        isinstance(node, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == arg_src for t in node.targets)
        and ("file://" in ast.unparse(node.value) or "as_uri" in ast.unparse(node.value))
        for node in ast.walk(func)
    )


def _calls_named(func: ast.AST, name: str) -> list[ast.Call]:
    """Every ``Call`` in *func* to a bare name or attribute called *name*."""
    out: list[ast.Call] = []
    for node in ast.walk(func):
        if not isinstance(node, ast.Call):
            continue
        fn = node.func
        bare = isinstance(fn, ast.Name) and fn.id == name
        attr = isinstance(fn, ast.Attribute) and fn.attr == name
        if bare or attr:
            out.append(node)
    return out


def _remote_navigation_sinks() -> list[tuple[str, str, int, bool, bool]]:
    """Every ``page.goto`` in ``src/`` aimed at a remote host.

    Returns:
        One ``(relpath, funcname, goto_lineno, keeps_vetted_ip, leases_pinned)``
        per sink. ``keeps_vetted_ip`` means the enclosing function ASSIGNS the
        return of ``validate_public_url`` (discarding it is the exact bug);
        ``leases_pinned`` means it calls ``lease_pinned``.
    """
    sinks: list[tuple[str, str, int, bool, bool]] = []
    for path in sorted(SRC_ROOT.rglob("*.py")):
        tree = ast.parse(path.read_text(), filename=str(path))
        for func in ast.walk(tree):
            if not isinstance(func, ast.AsyncFunctionDef | ast.FunctionDef):
                continue
            gotos = [
                node
                for node in _calls_named(func, "goto")
                if node.args and not _is_local_document(ast.unparse(node.args[0]), func)
            ]
            if not gotos:
                continue
            keeps_vetted_ip = any(
                isinstance(node, ast.Assign)
                and isinstance(node.value, ast.Call)
                and isinstance(node.value.func, ast.Name)
                and node.value.func.id == "validate_public_url"
                for node in ast.walk(func)
            )
            leases_pinned = bool(_calls_named(func, "lease_pinned"))
            rel = str(path.relative_to(SRC_ROOT))
            sinks.extend((rel, func.name, g.lineno, keeps_vetted_ip, leases_pinned) for g in gotos)
    return sinks


def test_every_remote_navigation_sink_dns_pins_its_target() -> None:
    """MUTATION KILLED: reverting ``_fetch_page_html`` to ``validate_public_url(url)``
    + ``pool.lease()`` lists ``design_builder/helpers.py`` in the failure message.
    """
    sinks = _remote_navigation_sinks()
    # Vacuity floor. Deleting a sink to make this pass trips the floor instead.
    assert len(sinks) >= 2, (
        f"expected >=2 remote-navigation sinks, found {sinks} — has the walk "
        "stopped matching (renamed modules, a different navigation spelling)?"
    )
    unpinned = [(p, fn, ln) for p, fn, ln, keeps_ip, pinned in sinks if not (keeps_ip and pinned)]
    assert not unpinned, (
        f"remote-navigation sink(s) {unpinned} hand a tenant-supplied HOSTNAME to "
        "Chromium without pinning it to the address validate_public_url vetted. "
        "Chromium re-resolves the host itself, so validation alone leaves a "
        "DNS-rebind window: capture the returned IP and use pool.lease_pinned()."
    )


def test_the_walk_still_sees_both_known_sinks() -> None:
    """Second vacuity guard: the floor above cannot tell WHICH sinks it counted.

    Both historical siblings must remain visible by name, so a rename that makes
    the walk silently blind fails here rather than passing with two unrelated
    matches.
    """
    seen = {(path, func) for path, func, _, _, _ in _remote_navigation_sinks()}
    assert ("services/design_builder/helpers.py", "_fetch_page_html") in seen, seen
    assert ("services/screenshot_renderer.py", "capture_url_screenshots") in seen, seen
