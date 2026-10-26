"""
Shared test setup.

- helper/visualize.py imports the ALR simulation framework, which training does not need: stubbed if missing.
- Tests that produce figures write them to test_outputs/ (or $DIGNET_TEST_OUTPUT) and list them in
  test_outputs/index.html, so the results can be checked by eye after `pytest tests`.
"""
import html
import os
import sys
import types

import pytest

try:
    import alr_sim  # noqa: F401
except ImportError:
    for name in ('alr_sim', 'alr_sim.utils', 'alr_sim.utils.sim_path', 'alr_sim.utils.geometric_transformation'):
        sys.modules.setdefault(name, types.ModuleType(name))
    sys.modules['alr_sim.utils.sim_path'].sim_framework_path = lambda *a: ''
    sys.modules['alr_sim.utils.geometric_transformation'].quat2mat = lambda *a: None

OUTPUT_DIR = os.environ.get('DIGNET_TEST_OUTPUT', os.path.join(os.path.dirname(__file__), '..', 'test_outputs'))
_figures = []


@pytest.fixture(scope='session')
def figure_dir():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    return OUTPUT_DIR


def register_figure(path, caption):
    """ adds a figure to test_outputs/index.html """
    _figures.append((os.path.relpath(path, OUTPUT_DIR), caption))


def pytest_sessionfinish(session, exitstatus):
    if not _figures:
        return
    items = '\n'.join(f'<figure><figcaption>{html.escape(c)}</figcaption><img src="{html.escape(p)}"></figure>'
                      for p, c in sorted(_figures))
    page = f"""<!doctype html><meta charset="utf-8"><title>DIGNet test figures</title>
<style>body{{font-family:sans-serif;margin:16px;background:#fff;color:#222}} figure{{margin:0 0 32px}}
img{{max-width:100%;border:1px solid #ddd}} figcaption{{font-weight:600;margin-bottom:6px}}</style>
<h1>DIGNet test figures</h1>
<p>Grasps: green = both contacts visible in the cloud, orange = one, red = none (successful grasps only).
Model plots: blue = ground truth, green = sampled.</p>
{items}"""
    open(os.path.join(OUTPUT_DIR, 'index.html'), 'w').write(page)
