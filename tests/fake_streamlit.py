"""
A recording fake of the parts of `streamlit` the app uses, plus a minimal fake of `plotly.graph_objects`
(installed only when real plotly is missing). It lets the app script run end to end in a plain Python process.

It cannot catch misuse of Streamlit's real API (that is what tests/test_app.py::test_real_streamlit_* is for, on a machine
that has Streamlit); it catches wiring errors: bad attribute names, wrong column names, wrong types passed between layers.
"""
import importlib
import sys
import types
from types import SimpleNamespace


class StopApp(Exception):
    pass


class RerunRequested(Exception):
    pass


class _State(dict):
    __getattr__ = dict.get

    def __setattr__(self, k, v):
        self[k] = v


class FakeStreamlit:
    def __init__(self, inputs=None):
        self.calls = []
        self.session_state = _State()
        self.inputs = inputs or {}

    # --- context-manager style containers
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    sidebar = property(lambda self: self)

    def _rec(self, name, *a, **k):
        self.calls.append((name, a, k))

    def chat_message(self, *a, **k): self._rec("chat_message", *a, **k); return self
    def spinner(self, *a, **k): self._rec("spinner", *a, **k); return self
    def expander(self, *a, **k): self._rec("expander", *a, **k); return self
    def container(self, *a, **k): self._rec("container", *a, **k); return self

    def columns(self, spec, **k):
        self._rec("columns", spec); return [self] * (spec if isinstance(spec, int) else len(spec))

    def tabs(self, labels): self._rec("tabs", labels); return [self] * len(labels)

    # --- caching decorators: identity
    def cache_data(self, *a, **k):
        return a[0] if a and callable(a[0]) else (lambda f: f)

    cache_resource = cache_data

    # --- widgets
    def selectbox(self, label, options, index=0, format_func=None, **k):
        options = list(options); self._rec("selectbox", label, options); return options[index]

    def slider(self, label, min_value=None, max_value=None, value=None, **k):
        self._rec("slider", label); return self.inputs.get("slider", value)

    def button(self, label, **k):
        self._rec("button", label)
        return self.inputs.get("buttons", {}).pop(label, False)

    def chat_input(self, *a, **k):
        queue = self.inputs.get("chat_input", [])
        return queue.pop(0) if queue else None

    def stop(self): raise StopApp()
    def rerun(self): raise RerunRequested()

    def plotly_chart(self, fig, **k): self._rec("plotly_chart", fig, **k)

    def __getattr__(self, name):                 # every other st.* call: just record it
        if name.startswith("__"):
            raise AttributeError(name)
        return lambda *a, **k: self._rec(name, *a, **k)

    # --- inspection helpers
    def named(self, name):
        return [(a, k) for n, a, k in self.calls if n == name]

    def texts(self):
        out = []
        for n, a, k in self.calls:
            if n in ("markdown", "write", "info", "success", "warning", "error", "caption", "subheader", "header", "title", "metric") and a:
                out.append(" ".join(str(x) for x in a))
        return out


def install_fake_plotly_if_missing():
    try:
        import plotly.graph_objects  # noqa: F401
        return False
    except ImportError:
        pass

    class Figure:
        def __init__(self):
            self.data, self.layout = [], {}

        def add_trace(self, t): self.data.append(t); return self
        def update_layout(self, **k): self.layout.update(k); return self
        update_yaxes = update_xaxes = lambda self, **k: self
        def add_hline(self, **k): return self

    go = types.ModuleType("plotly.graph_objects")
    go.Figure = Figure
    for name in ("Scatter", "Bar", "Pie", "Indicator"):
        go.__dict__[name] = (lambda n: (lambda **k: SimpleNamespace(type=n, **k)))(name.lower())
    plotly = types.ModuleType("plotly")
    plotly.graph_objects = go
    sys.modules["plotly"], sys.modules["plotly.graph_objects"] = plotly, go
    return True


def run_app(runner, inputs=None, max_reruns=4):
    """Execute app/streamlit_app.py's main() against the fake, honouring st.rerun(). Returns (fake_st, app_module)."""
    install_fake_plotly_if_missing()
    fake = FakeStreamlit(inputs)
    saved = {k: sys.modules.get(k) for k in ("streamlit", "app.streamlit_app", "app.charts")}
    sys.modules["streamlit"] = fake
    for k in ("app.streamlit_app", "app.charts"):
        sys.modules.pop(k, None)
    try:
        mod = importlib.import_module("app.streamlit_app")
        for _ in range(max_reruns):
            try:
                mod.main(runner_factory=lambda: runner)
                break
            except RerunRequested:
                continue
            except StopApp:
                break
        return fake, mod
    finally:
        for k, v in saved.items():
            if v is None:
                sys.modules.pop(k, None)
            else:
                sys.modules[k] = v
