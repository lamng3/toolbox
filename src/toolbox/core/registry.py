"""Component registries — the backbone of the plugin architecture.

A :class:`Registry` maps a string name to a component *factory* (a class or a function).
Configs refer to components by name, and :meth:`Registry.build` turns a config mapping such as
``{"name": "tiny_gpt", "d_model": 128}`` into an object (pattern: *Factory*).

Built-in components register themselves with the :meth:`Registry.register` decorator
(pattern: *Decorator*). Third-party packages extend the toolbox without touching its source by
exposing an entry point in the ``toolbox.plugins`` group (pattern: *Plugin Architecture*)::

    # pyproject.toml of an external package
    [project.entry-points."toolbox.plugins"]
    my_plugin = "my_package.toolbox_plugin"

Importing ``my_package.toolbox_plugin`` is enough: its ``@OBJECTIVES.register(...)`` calls run
at import time.
"""

from __future__ import annotations

import importlib
import importlib.metadata
import inspect
import threading
from collections.abc import Callable, Iterator, Mapping
from typing import Any, Generic, TypeVar

from toolbox.core.constants import PLUGIN_ENTRY_POINT_GROUP

T = TypeVar("T")
F = TypeVar("F", bound=Callable[..., Any])

NAME_KEY = "name"


class RegistryError(KeyError):
    """Raised when a component name is unknown or registered twice."""

    def __str__(self) -> str:  # KeyError quotes its message; keep it readable.
        return str(self.args[0]) if self.args else ""


class Registry(Generic[T]):
    """A named collection of component factories.

    Args:
        kind: Human readable category (``"model"``, ``"objective"``, ...), used in errors
            and by ``toolbox list``.
    """

    def __init__(self, kind: str) -> None:
        self.kind = kind
        self._entries: dict[str, Callable[..., T]] = {}

    # -- registration ----------------------------------------------------------------------
    def register(self, name: str | None = None, *, override: bool = False) -> Callable[[F], F]:
        """Decorator registering a class or function under ``name`` (defaults to ``__name__``)."""

        def decorator(factory: F) -> F:
            key = name or factory.__name__
            if key in self._entries and not override:
                raise RegistryError(
                    f"{self.kind} '{key}' is already registered; pass override=True to replace it."
                )
            self._entries[key] = factory
            return factory

        return decorator

    def add(self, name: str, factory: Callable[..., T], *, override: bool = False) -> None:
        """Imperative alternative to :meth:`register`."""
        self.register(name, override=override)(factory)

    # -- lookup ----------------------------------------------------------------------------
    def get(self, name: str) -> Callable[..., T]:
        load_builtins()
        try:
            return self._entries[name]
        except KeyError:
            available = ", ".join(sorted(self._entries)) or "<none>"
            raise RegistryError(f"Unknown {self.kind} '{name}'. Available: {available}.") from None

    def build(
        self,
        spec: str | Mapping[str, Any],
        /,
        inject: Mapping[str, Any] | None = None,
        **overrides: Any,
    ) -> T:
        """Instantiate a component from a name or a ``{"name": ..., **kwargs}`` mapping.

        Args:
            spec: component name, or mapping with a ``name`` key plus constructor kwargs.
            inject: *optional* dependencies (tokenizer, data-derived shapes such as
                ``vocab_size`` / ``in_dim``). Each one is passed only if the factory's
                signature accepts it and the spec did not set it explicitly — so one builder can
                serve components with different needs (pattern: *Dependency Injection*).
            **overrides: *mandatory* kwargs that win over the spec.
        """
        if isinstance(spec, str):
            name, kwargs = spec, {}
        else:
            kwargs = dict(spec)
            if NAME_KEY not in kwargs:
                raise RegistryError(f"{self.kind} spec is missing the '{NAME_KEY}' key: {spec!r}")
            name = kwargs.pop(NAME_KEY)
        factory = self.get(name)
        if inject:
            accepted = _accepted_kwargs(factory)
            for key, value in inject.items():
                if key not in kwargs and (accepted is None or key in accepted):
                    kwargs[key] = value
        kwargs.update(overrides)
        return factory(**kwargs)

    def names(self) -> list[str]:
        load_builtins()
        return sorted(self._entries)

    def __contains__(self, name: object) -> bool:
        load_builtins()
        return name in self._entries

    def __iter__(self) -> Iterator[str]:
        return iter(self.names())

    def __len__(self) -> int:
        return len(self.names())

    def __repr__(self) -> str:
        return f"Registry(kind={self.kind!r}, entries={len(self._entries)})"


# -- the global registries ------------------------------------------------------------------
MODELS: Registry[Any] = Registry("model")
TOKENIZERS: Registry[Any] = Registry("tokenizer")
DATASETS: Registry[Any] = Registry("dataset")
OBJECTIVES: Registry[Any] = Registry("objective")
REWARDS: Registry[Any] = Registry("reward")
ADVANTAGES: Registry[Any] = Registry("advantage")
ENVS: Registry[Any] = Registry("env")
OPTIMIZERS: Registry[Any] = Registry("optimizer")
SCHEDULERS: Registry[Any] = Registry("scheduler")
CALLBACKS: Registry[Any] = Registry("callback")
TRAINERS: Registry[Any] = Registry("trainer")
STRATEGIES: Registry[Any] = Registry("strategy")
ENGINES: Registry[Any] = Registry("engine")
LAUNCHERS: Registry[Any] = Registry("launcher")
MEMORIES: Registry[Any] = Registry("memory")
EVALUATORS: Registry[Any] = Registry("evaluator")

ALL_REGISTRIES: dict[str, Registry[Any]] = {
    r.kind: r
    for r in (
        MODELS,
        TOKENIZERS,
        DATASETS,
        OBJECTIVES,
        REWARDS,
        ADVANTAGES,
        ENVS,
        OPTIMIZERS,
        SCHEDULERS,
        CALLBACKS,
        TRAINERS,
        STRATEGIES,
        ENGINES,
        LAUNCHERS,
        MEMORIES,
        EVALUATORS,
    )
}

# Modules whose import registers the built-in components. Kept as strings so that importing
# ``toolbox.core`` stays cheap and free of circular imports.
BUILTIN_MODULES: tuple[str, ...] = (
    "toolbox.nn.models",
    "toolbox.data.tokenization",
    "toolbox.data.datasets",
    "toolbox.objectives",
    "toolbox.rewards",
    "toolbox.rl",
    "toolbox.decision",
    "toolbox.trainer",
    "toolbox.distributed",
    "toolbox.serving",
    "toolbox.launch",
    "toolbox.memory",
    "toolbox.tracing",
    "toolbox.eval",
)


def _accepted_kwargs(factory: Callable[..., Any]) -> set[str] | None:
    """Names ``factory`` accepts as keywords; ``None`` means anything (it takes ``**kwargs``)."""
    try:
        params = inspect.signature(factory).parameters.values()
    except (TypeError, ValueError):  # builtins without signatures
        return set()
    if any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params):
        return None
    return {
        p.name
        for p in params
        if p.kind in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)
    }


_load_lock = threading.Lock()
_loaded = False


def load_builtins() -> None:
    """Import built-in component modules and installed plugins exactly once."""
    global _loaded
    if _loaded:
        return
    with _load_lock:
        if _loaded:
            return
        _loaded = True  # set first: the imports below call back into the registries
        for module in BUILTIN_MODULES:
            importlib.import_module(module)
        load_plugins()


def load_plugins(group: str = PLUGIN_ENTRY_POINT_GROUP) -> list[str]:
    """Import every module advertised under the ``toolbox.plugins`` entry-point group."""
    loaded: list[str] = []
    for entry_point in importlib.metadata.entry_points(group=group):
        entry_point.load()
        loaded.append(entry_point.name)
    return loaded
