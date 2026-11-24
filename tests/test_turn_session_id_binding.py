"""Per-turn HERMES_SESSION_ID binding (#<issue>): concurrent WebUI turns must not
leak each other's session id into tool subprocesses.

The WebUI runs every turn in one process and writes the per-turn identity into the
process-global ``os.environ`` mirror, which concurrent turns overwrite while an
earlier turn is still executing tools. hermes-agent's subprocess bridge
(``tools/environments/local.py::_inject_session_context_env``) uses the task-local
``gateway.session_context._SESSION_ID`` ContextVar when it is bound and otherwise
falls back to that (clobbered) mirror. These tests pin the binding that keeps a
child process on its own turn's id without calling ``set_session_vars`` (which
would zero HERMES_SESSION_PLATFORM and break notify_on_complete wakeups).
"""
import os
import subprocess
import sys
import threading

import pytest

PRINT_ID = "import os,sys;sys.stdout.write(os.environ.get('HERMES_SESSION_ID',''))"


def _agent_imports():
    """Return (session_context, inject_env) or skip when the agent is unavailable."""
    try:
        import gateway.session_context as sc  # noqa: F401
    except Exception:  # pragma: no cover - agent not installed
        pytest.skip("hermes-agent not importable")
    try:
        from tools.environments.local import _inject_session_context_env as inject
    except Exception:  # pragma: no cover - older agent
        pytest.skip("hermes-agent has no _inject_session_context_env")
    return sc, inject


def _child_session_id(env):
    out = subprocess.run([sys.executable, "-c", PRINT_ID], capture_output=True, text=True,
                         env=env, timeout=60)
    assert out.returncode == 0, out.stderr
    return out.stdout.strip()


def test_binding_sets_and_restores_the_session_id(monkeypatch):
    sc, _ = _agent_imports()
    import api.streaming as streaming

    monkeypatch.setenv("HERMES_SESSION_ID", "mirror-value")
    tokens = streaming._set_turn_session_identity("sid-bound")
    try:
        assert sc.get_session_env("HERMES_SESSION_ID") == "sid-bound"
    finally:
        streaming._reset_turn_session_identity(tokens)
    # Reset-token semantics: back to the env fallback, not a pinned "".
    assert sc.get_session_env("HERMES_SESSION_ID") == "mirror-value"


def test_child_env_uses_bound_id_even_when_process_mirror_is_clobbered(monkeypatch):
    """The regression: the mirror says another session, the bound var must win."""
    _, inject = _agent_imports()
    import api.streaming as streaming

    monkeypatch.setenv("HERMES_SESSION_ID", "other-session")
    tokens = streaming._set_turn_session_identity("this-turn")
    try:
        env = dict(os.environ)
        inject(env)
        assert env["HERMES_SESSION_ID"] == "this-turn"
        assert _child_session_id(env) == "this-turn"
    finally:
        streaming._reset_turn_session_identity(tokens)
    # Nothing was written into the process-wide mirror by the binding.
    assert os.environ["HERMES_SESSION_ID"] == "other-session"


def test_concurrent_turns_each_spawn_children_with_their_own_id(monkeypatch):
    """Real concurrency check: two turns bind different ids at the same time."""
    _, inject = _agent_imports()
    import api.streaming as streaming

    monkeypatch.setenv("HERMES_SESSION_ID", "mirror-value")
    ids = ["sess-A", "sess-B"]
    barrier = threading.Barrier(len(ids))
    results = {}
    errors = []

    def run(sid):
        try:
            with streaming._bind_turn_session_identity(sid):
                barrier.wait(timeout=30)          # force the two turns to overlap
                seen = []
                for _ in range(5):
                    env = dict(os.environ)
                    inject(env)
                    seen.append(_child_session_id(env))
                results[sid] = seen
        except Exception as exc:                   # noqa: BLE001 - surface in the assert
            errors.append((sid, exc))

    threads = [threading.Thread(target=run, args=(sid,)) for sid in ids]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=120)

    assert not errors, errors
    for sid in ids:
        assert results[sid] == [sid] * 5, f"{sid} leaked: {results[sid]}"
    assert os.environ["HERMES_SESSION_ID"] == "mirror-value"
