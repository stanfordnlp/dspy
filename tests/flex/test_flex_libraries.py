"""dspy.Flex code written with the user's own libraries: how they are named, what they reach, where the code runs,
what the optimizer is told, the environment recorded with the code, and the tool calls a metric can score."""

import importlib
import json
import logging
import pickle
import shutil
import subprocess
import sys
import textwrap

import cloudpickle
import pydantic
import pytest

import dspy
from dspy.predict.flex import flex as flex_module
from dspy.primitives import libraries, sandbox_bundle
from dspy.primitives.code_interpreter import CodeInterpreterError, _create_interpreter
from dspy.primitives.facade import SandboxSession
from dspy.teleprompt.gepa.gepa_flex_utils import flex_task_context, pin_flex_environments
from dspy.teleprompt.gepa.gepa_utils import DspyAdapter
from dspy.utils.dummies import DummyLM

# Run by CI's Deno job (pytest -m deno --deno), which has Deno. Without --deno they are skipped.
deno_required = pytest.mark.deno
INTERPRETERS = [
    pytest.param(dspy.LocalInterpreter, id="local"),
    pytest.param(dspy.PythonInterpreter, marks=deno_required, id="deno"),
]

# A project of the user's own, which no model has seen: a package on the path, not installed.
SHOP = {
    "flexshop/__init__.py": '''
        """flexshop: the shop's own pricing abstractions. Prices are integer cents."""

        from flexshop.cart import Cart, Line
        from flexshop.vat import Country, vat_rate

        __all__ = ["Cart", "Line", "Country", "vat_rate"]
    ''',
    "flexshop/cart.py": '''
        from dataclasses import dataclass, field

        from .vat import vat_rate


        @dataclass
        class Line:
            """One cart line: a SKU, a quantity and a unit price in cents."""

            sku: str
            qty: int
            unit_cents: int


        class Basket:
            """What any basket of lines can tell."""

            def is_empty(self) -> bool:
                """Whether it has no line."""
                return not self.lines

            def gross_cents(self) -> int:
                raise NotImplementedError


        @dataclass
        class Cart(Basket):
            """A cart for one country (ISO code), which decides the VAT rate."""

            country: str
            lines: list[Line] = field(default_factory=list)

            def add(self, sku: str, qty: int, unit_cents: int) -> "Cart":
                """Add a line and return the cart, so calls chain."""
                self.lines.append(Line(sku, qty, unit_cents))
                return self

            def gross_cents(self) -> int:
                """Net plus VAT for the cart's country, rounded half up."""
                net = sum(line.qty * line.unit_cents for line in self.lines)
                return int(net * (1 + vat_rate(self.country)) + 0.5)

            def _audit(self) -> str:
                return repr(self.lines)
    ''',
    "flexshop/vat.py": '''
        import enum

        _RATES = {"DE": 0.19, "FR": 0.20}


        class Country(enum.Enum):
            DE = "DE"
            FR = "FR"


        def vat_rate(country: str) -> float:
            """The standard VAT rate of a country, by ISO code.

            Unknown countries raise KeyError.
            """
            return _RATES[country]
    ''',
    # Nothing above imports these.
    "flexshop/reports.py": """
        import flexshop_not_installed_anywhere


        def monthly_report():
            return "report"
    """,
    "flexshop/receipts/__init__.py": '''
        import pydantic


        class Receipt(pydantic.BaseModel):
            """What a customer gets."""

            total_cents: int
            note: str = ""


        def optional():
            try:
                import flexshop_not_installed_anywhere  # an optional import, which nothing provides
            except ImportError:
                return None
            return flexshop_not_installed_anywhere
    ''',
}

SIGNATURE = "lines: list[tuple[str, int, int]], country: str -> total_cents: int"
TOTAL = textwrap.dedent(
    """
    from flexshop import Cart


    class TotalModule(dspy.Module):
        def forward(self, **inputs):
            cart = Cart(inputs["country"])
            for sku, qty, cents in inputs["lines"]:
                cart.add(sku, qty, cents)
            return dspy.Prediction(total_cents=cart.gross_cents())
    """
)
LINES = [("pen", 2, 500)]

# A start for code that only uses the libraries, for the optimizer to write.
STUB = textwrap.dedent(
    """
    class TotalModule(dspy.Module):
        def forward(self, **inputs):
            raise NotImplementedError("Not written yet: implement forward with the libraries.")
    """
)


def _write(root, files):
    for name, text in files.items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(textwrap.dedent(text).lstrip())


@pytest.fixture
def shop(tmp_path, monkeypatch):
    _write(tmp_path, SHOP)
    monkeypatch.syspath_prepend(str(tmp_path))
    importlib.invalidate_caches()
    yield importlib.import_module("flexshop")
    for name in [name for name in sys.modules if name.split(".")[0] == "flexshop"]:
        del sys.modules[name]
    sandbox_bundle._library_plans.clear()


@pytest.fixture(autouse=True)
def nothing_compared():
    flex_module._checked.clear()  # each test compares its recorded environments anew


@pytest.fixture
def warnings_seen(caplog, monkeypatch):
    monkeypatch.setattr(logging.getLogger("dspy"), "propagate", True)  # so that caplog sees dspy's warnings
    caplog.set_level(logging.WARNING)
    return caplog


def _total(shop, **kwargs):
    kwargs.setdefault("interpreter_factory", dspy.LocalInterpreter)
    kwargs.setdefault("module_src", TOTAL)
    return dspy.Flex(SIGNATURE, libraries=[shop], **kwargs)


# -- naming libraries ---------------------------------------------------------------------------------------------------


def test_libraries_are_modules_classes_functions_or_their_names(shop):
    flex = dspy.Flex(
        SIGNATURE,
        libraries=[shop, shop.Cart, shop.vat_rate, shop.Cart.add, "flexshop.receipts", "flexshop.cart:Line", shop],
    )
    assert flex._flex_ctx.libraries == [
        "flexshop", "flexshop.cart:Cart", "flexshop.vat:vat_rate", "flexshop.cart:Cart.add", "flexshop.receipts",
        "flexshop.cart:Line",
    ]
    assert flex._flex_ctx.library_modules() == [
        "flexshop", "flexshop.cart", "flexshop.vat", "flexshop.cart", "flexshop.receipts", "flexshop.cart",
    ]


@pytest.mark.parametrize(
    "library, error, says",
    [
        (lambda: None, ValueError, "cannot be imported from"),  # no name an interpreter could import it by
        ("not a module", ValueError, "does not name a library"),
        ("flexshop:not a name", ValueError, "does not name a library"),
        (object(), TypeError, "stays on the host"),  # an object with state is a tool's business
    ],
)
def test_what_an_interpreter_could_not_import_is_refused(library, error, says):
    with pytest.raises(error, match=says):
        dspy.Flex(SIGNATURE, libraries=[library])


# -- what libraries reach, and what an environment resolves them to ----------------------------------------------------


def test_reach_follows_imports_through_your_own_files_only(shop):
    files, packages = libraries.reach(["flexshop"])
    # Not reports.py, which nothing imports, and no package: this part of the project imports none.
    assert sorted(files) == ["flexshop/__init__.py", "flexshop/cart.py", "flexshop/vat.py"]
    assert packages == set()

    files, packages = libraries.reach(["flexshop.receipts"])
    # A package's parents are imported with it, and so is what they import. pydantic stands for itself: its files
    # are not followed. An import that nothing provides, like an optional one, is passed over.
    assert sorted(files) == ["flexshop/__init__.py", "flexshop/cart.py", "flexshop/receipts/__init__.py", "flexshop/vat.py"]
    assert packages == {"pydantic"}
    assert libraries.missing(["flexshop", "flexshop.receipts", "flexshop.nope", "flexshop_not_installed_anywhere"]) == [
        "flexshop.nope", "flexshop_not_installed_anywhere",
    ]


def test_describe_and_verify_agree_until_something_changes(shop, tmp_path):
    record = json.loads(libraries.describe(["flexshop.receipts", "flexshop.cart:Cart"]))

    assert record["python"] == f"{sys.version_info.major}.{sys.version_info.minor}"
    # The packages the libraries reach, with what those require, as installed here.
    versions = dict(requirement.split("==") for requirement in record["requires"])
    assert versions["pydantic"] == pydantic.VERSION and "pydantic-core" in versions and "dspy" not in versions
    assert sorted(record["sources"]["flexshop"]) == [
        "flexshop/__init__.py", "flexshop/cart.py", "flexshop/receipts/__init__.py", "flexshop/vat.py",
    ]
    found = json.loads(libraries.verify(record))
    assert found == {"python": record["python"], "versions": versions, "hashes": record["sources"]["flexshop"]}

    vat = tmp_path / "flexshop" / "vat.py"
    vat.write_text(vat.read_text().replace("0.19", "0.21"))  # what it does, not how it is called
    (tmp_path / "flexshop" / "cart.py").unlink()
    record["requires"].append("flexshop-not-installed-anywhere==1.0")
    found = json.loads(libraries.verify(record))
    assert found["hashes"]["flexshop/vat.py"] != record["sources"]["flexshop"]["flexshop/vat.py"]
    assert found["hashes"]["flexshop/cart.py"] is None
    assert found["hashes"]["flexshop/__init__.py"] == record["sources"]["flexshop"]["flexshop/__init__.py"]
    assert found["versions"]["flexshop-not-installed-anywhere"] is None
    assert flex_module.differences(record, found["python"], found["versions"], found["hashes"]) == [
        "flexshop-not-installed-anywhere is not installed", "flexshop/cart.py is gone", "flexshop/vat.py changed",
    ]


def test_a_hash_ignores_how_a_checkout_ends_its_lines(shop, tmp_path):
    record = json.loads(libraries.describe(["flexshop"]))
    vat = tmp_path / "flexshop" / "vat.py"
    vat.write_bytes(vat.read_bytes().replace(b"\n", b"\r\n"))
    assert json.loads(libraries.describe(["flexshop"])) == record


def test_an_installed_package_is_its_version_and_a_project_installed_from_its_folder_is_its_files(shop, tmp_path):
    info = tmp_path / "flexshop-0.3.1.dist-info"
    info.mkdir()
    (info / "METADATA").write_text("Metadata-Version: 2.1\nName: flexshop\nVersion: 0.3.1\n")
    (info / "top_level.txt").write_text("flexshop\n")
    (info / "RECORD").write_text("flexshop/__init__.py,,\n")
    importlib.invalidate_caches()

    record = json.loads(libraries.describe(["flexshop"]))
    assert record["requires"] == ["flexshop==0.3.1"] and record["sources"] == {}
    assert libraries.reference(["flexshop"]).startswith("### flexshop 0.3.1\n")

    # Installed from a local folder, like a project into its own environment: the version says nothing of the code.
    (info / "direct_url.json").write_text(json.dumps({"url": tmp_path.as_uri(), "dir_info": {"editable": True}}))
    record = json.loads(libraries.describe(["flexshop"]))
    assert record["requires"] == [] and "flexshop/vat.py" in record["sources"]["flexshop"]
    assert libraries.reference(["flexshop"]).startswith("### flexshop\n")


def test_a_library_is_documented_from_its_own_api(shop):
    reference = libraries.reference(["flexshop", "flexshop.vat:vat_rate"])

    assert reference.startswith("### flexshop\n\nimport flexshop\n# flexshop: the shop's own pricing abstractions.")
    assert "class flexshop.Cart(country: str, lines: list[flexshop.cart.Line] = <factory>)\n" in reference
    assert "    def add(sku: str, qty: int, unit_cents: int) -> flexshop.cart.Cart\n        # Add a line" in reference
    assert "class flexshop.Country(*values)\n    # members: DE, FR" in reference or "    # members: DE, FR" in reference
    assert "_audit" not in reference and "flexshop.field" not in reference  # private, and what it imported
    # A class has what its bases in the library give it, once, as the class nearest to it defines it.
    assert "    def is_empty() -> bool\n        # Whether it has no line." in reference
    assert reference.count("    def gross_cents() -> int\n") == 1 and "Net plus VAT" in reference
    # What a base from elsewhere gives it is named, not listed.
    receipt = libraries.reference(["flexshop.receipts:Receipt"])
    assert "    # inherits from pydantic.main.BaseModel" in receipt and "model_dump" not in receipt
    # One function, named on its own: how to import it, and its whole docstring.
    assert (
        "### flexshop.vat.vat_rate\n\nfrom flexshop.vat import vat_rate\ndef vat_rate(country: str) -> float\n"
        "    # The standard VAT rate of a country, by ISO code.\n    #\n    # Unknown countries raise KeyError."
    ) in reference


def test_a_long_reference_is_cut_with_a_way_to_get_the_rest(shop, monkeypatch):
    monkeypatch.setattr(libraries, "REFERENCE_CHARS", 120)
    reference = libraries.reference(["flexshop"])
    assert len(reference) < 400 and reference.endswith("to have it documented in full.")


# -- where the code runs ------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("factory", INTERPRETERS)
def test_code_imports_your_own_abstractions_at_the_top_of_the_module(shop, factory):
    flex = _total(shop, interpreter_factory=factory)
    assert flex(lines=LINES, country="DE").total_cents == 1190


@deno_required
def test_the_sandbox_gets_what_the_libraries_reach_and_no_more(shop):
    source = textwrap.dedent(
        """
        class Probe(dspy.Module):
            def forward(self, **inputs):
                import importlib.util
                import importlib.metadata
                found = {name: importlib.util.find_spec(name) is not None for name in inputs["names"]}
                return dspy.Prediction(found=found, pydantic=importlib.metadata.version("pydantic"))
        """
    )
    flex = dspy.Flex("names: list[str] -> found: dict[str, bool], pydantic: str", libraries=["flexshop.receipts"],
                     module_src=source)
    out = flex(names=["flexshop.cart", "flexshop.receipts", "flexshop.reports"])
    assert out.found == {"flexshop.cart": True, "flexshop.receipts": True, "flexshop.reports": False}
    # The record has the version the sandbox runs, which is Pyodide's build, whatever this environment has.
    assert f"pydantic=={out.pydantic}" in flex.pin()["requires"]


def test_a_library_this_environment_lacks_is_refused_for_the_sandbox(shop, monkeypatch):
    monkeypatch.setattr(sandbox_bundle, "_distributions", lambda: [])
    with pytest.raises(sandbox_bundle.SandboxBundleError, match=r"flexshop\.nope cannot be imported in this environment"):
        sandbox_bundle._library_plan({}, ("flexshop.nope",))


def test_a_compiled_library_is_refused_for_the_sandbox_and_one_it_only_imports_is_left_out(shop, monkeypatch, warnings_seen):
    monkeypatch.setattr(sandbox_bundle, "_distributions", lambda: [])
    # pydantic_core is compiled. Pyodide ships a build of it, but say it did not.
    with pytest.raises(sandbox_bundle.SandboxBundleError, match="pydantic_core cannot run in the Pyodide sandbox"):
        sandbox_bundle._library_plan({}, ("pydantic_core",))

    # flexshop.receipts imports pydantic, which needs it: the import may be optional, so the rest is still planned.
    needed, files, own = sandbox_bundle._library_plan({}, ("flexshop.receipts",))
    assert "The Pyodide sandbox has no build of" in warnings_seen.text and "pydantic-core" in warnings_seen.text
    # For the bundle: the package, from this environment, with the metadata that says which version it is.
    names = {name for _, name in files}
    assert "pydantic/__init__.py" in names and not any(name.startswith("flexshop") for name in names)
    assert any(name.endswith(".dist-info/METADATA") and name.startswith("pydantic-") for name in names)
    assert needed == []
    # For an archive of their own: your files, which change more often than any package.
    assert sorted(name for _, name in own) == [
        "flexshop/__init__.py", "flexshop/cart.py", "flexshop/receipts/__init__.py", "flexshop/vat.py",
    ]


def test_only_the_code_the_libraries_import_is_copied(shop, tmp_path, monkeypatch):
    monkeypatch.setattr(sandbox_bundle, "_distributions", lambda: [])
    # Files next to the code may hold anything, and a link may point anywhere: none of it leaves the host.
    secret = tmp_path / "credentials.json"
    secret.write_text('{"key": "local"}')
    (tmp_path / "flexshop" / "rates.json").write_text('{"CH": 0.081}')
    (tmp_path / "flexshop" / "config.json").symlink_to(secret)
    if shutil.which("git"):  # tracked by git or not
        subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
        subprocess.run(["git", "add", "flexshop"], cwd=tmp_path, check=True)
    _, files, own = sandbox_bundle._library_plan({}, ("flexshop",))
    assert files == []
    assert sorted(name for _, name in own) == ["flexshop/__init__.py", "flexshop/cart.py", "flexshop/vat.py"]


def test_a_plan_is_made_again_when_a_file_it_followed_changes(shop, tmp_path, monkeypatch):
    monkeypatch.setattr(sandbox_bundle, "_distributions", lambda: [])
    first = sandbox_bundle._library_plan({}, ("flexshop",))
    assert sandbox_bundle._library_plan({}, ("flexshop",)) is first
    vat = tmp_path / "flexshop" / "vat.py"
    vat.write_text(vat.read_text() + "\nfrom flexshop import reports\n")
    again = sandbox_bundle._library_plan({}, ("flexshop",))
    assert "flexshop/reports.py" in {name for _, name in again[2]}
    assert "flexshop/reports.py" not in {name for _, name in first[2]}

    # A file it listed that is gone is not listed again.
    (tmp_path / "flexshop" / "reports.py").unlink()
    vat.write_text(vat.read_text().replace("\nfrom flexshop import reports\n", ""))
    assert "flexshop/reports.py" not in {name for _, name in sandbox_bundle._library_plan({}, ("flexshop",))[2]}


def test_a_plan_is_made_again_when_the_files_of_a_package_it_listed_change(shop, monkeypatch):
    monkeypatch.setattr(sandbox_bundle, "_distributions", lambda: [])
    times = {"now": 1}
    monkeypatch.setattr(sandbox_bundle, "_folder_times", lambda files: [times["now"]] if files else [])
    first = sandbox_bundle._library_plan({}, ("flexshop.receipts",))  # lists pydantic's files, from this environment
    assert first[1] and sandbox_bundle._library_plan({}, ("flexshop.receipts",)) is first
    times["now"] = 2  # as after the package was upgraded: a file added to, or gone from, one of its folders
    assert sandbox_bundle._library_plan({}, ("flexshop.receipts",)) is not first


def test_a_plan_is_made_again_when_a_module_an_import_names_appears(shop, tmp_path, monkeypatch):
    monkeypatch.setattr(sandbox_bundle, "_distributions", lambda: [])
    vat = tmp_path / "flexshop" / "vat.py"
    vat.write_text(vat.read_text() + "\ntry:\n    from flexshop import extras\nexcept ImportError:\n    extras = None\n")
    first = sandbox_bundle._library_plan({}, ("flexshop",))
    assert "flexshop/extras.py" not in {name for _, name in first[2]}
    (tmp_path / "flexshop" / "extras.py").write_text("RATE = 1\n")  # no file it listed changed: its folder did
    assert "flexshop/extras.py" in {name for _, name in sandbox_bundle._library_plan({}, ("flexshop",))[2]}


@deno_required
def test_an_interpreter_takes_libraries_until_its_bundle_is_prepared(shop):
    interpreter = dspy.PythonInterpreter()
    interpreter.provide_libraries(["flexshop"])
    interpreter.provide_libraries(["flexshop"])  # nothing new: nothing to do
    interpreter.provide_dspy()
    with interpreter:
        assert interpreter.execute("import flexshop\nflexshop.vat_rate('FR')") == 0.2
        interpreter.provide_libraries(["flexshop"])
        with pytest.raises(sandbox_bundle.SandboxBundleError, match="prepared without json5"):
            interpreter.provide_libraries(["json5"])


class _Interpreter:
    """An interpreter that records what it is asked to provide."""

    def __init__(self):
        self.tools, self.provided = {}, []

    def provide_libraries(self, modules):
        self.provided.append(tuple(modules))

    def start(self):
        pass

    def execute(self, code, variables=None):
        return None

    def shutdown(self):
        pass


def test_a_factory_that_prepares_interpreters_is_told_the_libraries_and_an_interpreter_class_is_not():
    class Factory:
        def __init__(self):
            self.told = []

        def provide_libraries(self, modules):
            self.told.append(tuple(modules))

        def __call__(self):
            return _Interpreter()

    factory = Factory()
    _create_interpreter(factory, ["flexshop"])
    _create_interpreter(factory)  # nothing to tell
    assert factory.told == [("flexshop",)]
    # A class has provide_libraries for its instances: it is only called to make one.
    assert isinstance(_create_interpreter(_Interpreter, ["flexshop"]), _Interpreter)


def test_an_interpreter_made_for_sandboxed_code_gets_the_libraries_too():
    made = []

    def factory():
        made.append(_Interpreter())
        return made[-1]

    session = SandboxSession({}, interpreter_factory=factory, libraries=["flexshop"])
    handle = session._serve_interpreter("create")
    session._serve_interpreter("provide", id=handle)
    assert made[0].provided == [("flexshop",)]
    session.close()


# -- values of a library's classes -------------------------------------------------------------------------------------


@pytest.mark.parametrize("factory", INTERPRETERS)
def test_a_value_of_a_library_class_is_an_object_of_it_in_the_interpreter(shop, factory):
    from flexshop.receipts import Receipt

    class Checkout(dspy.Signature):
        cart: shop.Cart = dspy.InputField()
        country: shop.Country = dspy.InputField()
        receipt: Receipt = dspy.OutputField()
        kinds: list[str] = dspy.OutputField()

    source = textwrap.dedent(
        """
        from flexshop.receipts import Receipt


        class CheckoutModule(dspy.Module):
            def forward(self, **inputs):
                cart, country = inputs["cart"], inputs["country"]
                cart.add("ink", 1, 1000)  # its methods are there
                kinds = [type(cart).__name__, type(cart.lines[0]).__name__, type(country).__name__, country.value]
                return dspy.Prediction(receipt=Receipt(total_cents=cart.gross_cents(), note="paid"), kinds=kinds)
        """
    )
    flex = dspy.Flex(Checkout, libraries=[shop, Receipt], module_src=source, interpreter_factory=factory)

    out = flex(cart=shop.Cart("DE").add("pen", 2, 500), country=shop.Country.DE)

    assert out.kinds == ["Cart", "Line", "Country", "DE"]
    # What comes back is read as the signature's type, here: the host builds no class on the sandbox's word.
    assert out.receipt == Receipt(total_cents=2380, note="paid") and type(out.receipt) is Receipt


@pytest.mark.parametrize("factory", INTERPRETERS)
def test_a_predictor_in_the_code_reads_and_writes_a_library_class_as_the_host_has_it(shop, factory):
    class Order(dspy.Signature):
        """Read the order."""

        order: str = dspy.InputField()
        cart: shop.Cart = dspy.OutputField()

    flex = dspy.Flex(Order, libraries=[shop], interpreter_factory=factory)  # the baseline: one Predict over Order
    lm = DummyLM([{"cart": '{"country": "FR", "lines": [{"sku": "pen", "qty": 1, "unit_cents": 100}]}'}])
    with dspy.context(lm=lm):
        out = flex(order="one pen to France")

    assert out.cart == shop.Cart("FR", [shop.Line("pen", 1, 100)])
    assert "unit_cents" in str(lm.history[-1]["messages"])  # the adapter described the type, from the host's class


def test_a_type_the_libraries_do_not_reach_stays_plain_data(shop):
    from flexshop.receipts import Receipt  # in the project, but not among the libraries below

    source = textwrap.dedent(
        """
        class Probe(dspy.Module):
            def forward(self, **inputs):
                return dspy.Prediction(kind=type(inputs["receipt"]).__name__, total=inputs["receipt"]["total_cents"])
        """
    )

    class Probe(dspy.Signature):
        receipt: Receipt = dspy.InputField()
        kind: str = dspy.OutputField()
        total: int = dspy.OutputField()

    flex = dspy.Flex(Probe, libraries=["json"], module_src=source, interpreter_factory=dspy.LocalInterpreter)
    out = flex(receipt=Receipt(total_cents=5))
    assert (out.kind, out.total) == ("dict", 5)
    assert "values of Receipt are plain data" in flex._flex_ctx.render_context_blurb(sandboxed=True)
    assert "plain data" not in dspy.Flex(Probe, libraries=[Receipt])._flex_ctx.render_context_blurb(sandboxed=True)


# -- the environment recorded with the code ----------------------------------------------------------------------------


def test_nothing_is_recorded_until_the_code_is_pinned_and_a_record_is_for_one_source(shop):
    flex = _total(shop)
    assert flex.environment is None and "environment" not in flex.dump_state()

    record = flex.pin()

    assert record == flex.environment == flex.dump_state()["environment"]
    assert record["python"] == f"{sys.version_info.major}.{sys.version_info.minor}" and record["requires"] == []
    assert sorted(record["sources"]["flexshop"]) == ["flexshop/__init__.py", "flexshop/cart.py", "flexshop/vat.py"]
    record["sources"].clear()
    assert flex.environment["sources"]  # a copy
    flex._bind_code(TOTAL)
    assert flex.environment is not None  # the same code
    flex._bind_code(STUB)
    assert flex.environment is None  # recorded for other code

    plain = dspy.Flex(SIGNATURE)  # no libraries, no tools: nothing to record
    assert plain.pin() is None and plain.check_environment() == [] and set(plain.dump_state()) == {"module_src", "lm"}


@pytest.mark.parametrize(
    "save, factory",
    [
        ("json", dspy.LocalInterpreter),
        ("pickle", dspy.LocalInterpreter),
        pytest.param("json", dspy.PythonInterpreter, marks=deno_required, id="json-deno"),
    ],
)
def test_a_saved_program_warns_once_when_your_code_changed_under_it(shop, tmp_path, warnings_seen, save, factory):
    flex = _total(shop, interpreter_factory=factory)
    flex.pin()
    if save == "json":
        flex.save(tmp_path / "total.json")
        loaded = _total(shop, module_src=None, interpreter_factory=factory)
        loaded.load(tmp_path / "total.json")
    else:
        loaded = pickle.loads(cloudpickle.dumps(flex))
    assert loaded.module_src == TOTAL and loaded.environment == flex.environment

    assert loaded(lines=LINES, country="DE").total_cents == 1190
    assert loaded.check_environment() == [] and not warnings_seen.records  # the same environment

    vat = tmp_path / "flexshop" / "vat.py"
    vat.write_text(vat.read_text().replace("0.19", "0.21"))  # what it does, not how it is called
    flex_module._checked.clear()  # as in another process

    assert loaded(lines=LINES, country="DE").total_cents == 1210
    assert "validated against another environment (flexshop/vat.py changed)" in warnings_seen.text
    assert loaded.check_environment() == ["flexshop/vat.py changed"]
    warnings_seen.clear()
    loaded(lines=LINES, country="DE")
    loaded.deepcopy()(lines=LINES, country="DE")
    assert not warnings_seen.records  # once

    loaded.pin()  # the evaluation still passes: this is now what it was validated against
    flex_module._checked.clear()
    loaded(lines=LINES, country="DE")
    assert loaded.check_environment() == [] and not warnings_seen.records


def test_the_record_covers_the_tools_and_what_a_module_no_longer_has(shop, warnings_seen):
    def lookup(order_id: int) -> dict:
        """Look up an order."""
        return {}

    def other_lookup(order_id: int) -> dict:
        """Look up an order, with its items."""
        return {}

    state = _total(shop, tools=[lookup]).dump_state()
    assert "environment" not in state
    flex = _total(shop, tools=[lookup])
    state = {**flex.dump_state(), "environment": flex.pin()}
    assert set(state["environment"]["tools"]) == {"lookup"}

    changed = _total(shop, tools=[dspy.Tool(other_lookup, name="lookup")])
    changed.load_state(state)
    assert changed.check_environment() == ["tool lookup changed"]

    bare = dspy.Flex(SIGNATURE, interpreter_factory=dspy.LocalInterpreter)
    bare.load_state(state)
    assert bare.check_environment() == ["it was validated with libraries, and now has none", "tool lookup is missing"]

    only_tools = dspy.Flex(SIGNATURE, tools=[lookup], interpreter_factory=dspy.LocalInterpreter)
    assert only_tools.pin() == {"tools": state["environment"]["tools"]}  # no interpreter is asked without libraries


def test_a_comparison_made_for_one_program_does_not_stand_for_another(shop, warnings_seen):
    def lookup(order_id: int) -> dict:
        """Look up an order."""
        return {}

    def other_lookup(order_id: int) -> dict:
        """Look up an order, with its items."""
        return {}

    flex = _total(shop, tools=[lookup])
    state = {**flex.dump_state(), "environment": flex.pin()}
    flex(lines=LINES, country="DE")
    assert not warnings_seen.records  # compared, in this process: nothing differs

    # The same code and record in a program whose tool changed: its own first call says so.
    changed = _total(shop, tools=[dspy.Tool(other_lookup, name="lookup")])
    changed.load_state(state)
    changed(lines=LINES, country="DE")
    assert "validated against another environment (tool lookup changed)" in warnings_seen.text

    # And in one that runs it elsewhere: compared there too, where nothing differs.
    warnings_seen.clear()
    elsewhere = _total(shop, tools=[lookup], interpreter_factory=lambda: dspy.LocalInterpreter())
    elsewhere.load_state(state)
    assert elsewhere._comparison() != flex._comparison()
    elsewhere(lines=LINES, country="DE")
    assert not warnings_seen.records


def test_a_forward_is_not_failed_by_an_environment_that_cannot_be_compared(shop, warnings_seen):
    flex = _total(shop)
    flex.load_state({**flex.dump_state(), "environment": {"python": "3.13", "sources": "not a record"}})
    assert flex(lines=LINES, country="DE").total_cents == 1190
    assert not warnings_seen.records


def test_a_module_pickled_before_environments_were_recorded_still_loads(shop):
    flex = _total(shop)
    state = flex.__getstate__()
    del state["_environment"]
    del state["_flex_ctx"].__dict__["libraries"]
    restored = dspy.Flex.__new__(dspy.Flex)
    restored.__setstate__(state)
    assert restored.environment is None and restored._flex_ctx.libraries == []


def test_a_library_the_interpreter_cannot_import_fails_when_it_is_asked(shop):
    flex = dspy.Flex(SIGNATURE, libraries=["flexshop_not_installed_anywhere"], interpreter_factory=dspy.LocalInterpreter)
    with pytest.raises(CodeInterpreterError, match="flexshop_not_installed_anywhere cannot be imported in this interpreter"):
        flex.pin()
    with pytest.raises(CodeInterpreterError, match="flexshop_not_installed_anywhere"):
        flex_task_context(flex)  # before an optimizer spends anything


# -- the optimizer --------------------------------------------------------------------------------------------------------


def test_the_optimizer_is_told_about_the_libraries_as_the_interpreter_has_them(shop):
    flex = _total(shop, module_src=None)
    assert "dspy.Predict" in flex.module_src  # the usual baseline
    assert _total(shop, module_src=STUB).module_src == STUB  # or code of your own, like a library-only start

    listed = flex._flex_ctx.render_context_blurb(sandboxed=True)
    assert "  - import flexshop" in listed and "DSPy's modules for what needs a language model" in listed
    _, blurbs = flex_task_context(flex)
    assert libraries.reference(["flexshop"]) in blurbs["self"] and "at the top of the module" in blurbs["self"]


def test_the_code_proposer_reads_the_library_reference(shop):
    reflection = DummyLM([{"revised_source": TOTAL}])
    student = _total(shop, module_src=STUB)
    adapter = DspyAdapter(student_module=student, metric_fn=lambda *a, **k: 0.0, feedback_map={}, reflection_lm=reflection)
    reflective = {"self": [{"Inputs": {"lines": "..."}, "Generated Outputs": "NotImplementedError", "Feedback": "x"}]}

    out = adapter.propose_new_texts({"self": student.module_src}, reflective, ["self"])

    assert out["self"].strip() == TOTAL.strip()
    prompt = "\n".join(message["content"] for message in reflection.history[-1]["messages"])
    assert "def flexshop.vat_rate(country: str) -> float" in prompt  # the reference
    assert "Common patterns" in prompt  # and DSPy's own catalog


def test_gepa_writes_the_module_from_an_unimplemented_start_and_records_what_it_ran_against(shop):
    def metric(gold, pred, trace=None, pred_name=None, pred_trace=None):
        got = getattr(pred, "total_cents", None)
        return dspy.Prediction(score=float(got == gold.total_cents), feedback=f"expected {gold.total_cents}, got {got}")

    examples = [
        dspy.Example(lines=[("pen", n, 100)], country="DE", total_cents=n * 119).with_inputs("lines", "country")
        for n in range(1, 4)
    ]
    student = _total(shop, module_src=STUB)
    reflection = DummyLM([{"revised_source": TOTAL}] * 10)

    optimized = dspy.GEPA(metric=metric, reflection_lm=reflection, max_metric_calls=12).compile(
        student, trainset=examples, valset=examples
    )

    assert "NotImplementedError" in str(reflection.history[0]["messages"])  # the stub's failure, as feedback
    assert optimized(lines=[("ink", 1, 1000)], country="FR").total_cents == 1200
    assert optimized.module_src.strip() == TOTAL.strip()
    assert "flexshop/vat.py" in optimized.environment["sources"]["flexshop"]  # recorded with the code it returned
    assert student.environment is None  # not on the program it was given


def test_libraries_that_change_during_an_optimization_are_reported(shop, tmp_path, warnings_seen):
    flex = _total(shop)
    started = {"self": flex._resolve_environment()}
    vat = tmp_path / "flexshop" / "vat.py"
    vat.write_text(vat.read_text().replace("0.19", "0.21"))

    pin_flex_environments(flex, started)

    assert "changed while it was being optimized (flexshop/vat.py changed)" in warnings_seen.text
    assert flex.check_environment() == []  # what it runs against now is what is recorded

    warnings_seen.clear()
    broken = dspy.Flex(SIGNATURE, libraries=["flexshop_not_installed_anywhere"], interpreter_factory=dspy.LocalInterpreter)
    pin_flex_environments(broken, {})  # the optimized program is not lost over it
    assert "Could not record the environment" in warnings_seen.text


# -- the tool calls of a forward ---------------------------------------------------------------------------------------


def test_the_tool_calls_of_a_forward_are_on_its_prediction():
    def lookup(order_id: int) -> dict:
        """Look up an order."""
        return {"id": order_id, "paid": 1250}

    def refund(order_id: int, cents: int) -> str:
        """Refund an order."""
        raise RuntimeError("refunds are closed")

    source = textwrap.dedent(
        """
        class Support(dspy.Module):
            def __init__(self):
                super().__init__()

            def forward(self, **inputs):
                order = lookup(order_id=inputs["order_id"])
                try:
                    refund(order_id=order["id"], cents=order["paid"])
                except Exception:
                    pass
                return dspy.Prediction(reply="done")
        """
    )
    flex = dspy.Flex("order_id: int -> reply", tools=[lookup, refund], module_src=source,
                     interpreter_factory=dspy.LocalInterpreter)

    prediction = flex(order_id=7)

    # Side effects a metric can score, and explain in its feedback. They are not outputs.
    assert prediction.tool_calls == [
        {"name": "lookup", "args": {"order_id": 7}, "result": {"id": 7, "paid": 1250}},
        {"name": "refund", "args": {"order_id": 7, "cents": 1250}, "error": "RuntimeError: refunds are closed"},
    ]
    assert "tool_calls" not in prediction.keys()


def test_the_calls_of_an_async_tool_are_recorded_when_they_are_over():
    async def lookup(order_id: int) -> dict:
        """Look up an order."""
        return {"id": order_id, "paid": 1250}

    async def refund(order_id: int) -> str:
        """Refund an order."""
        raise RuntimeError("refunds are closed")

    source = textwrap.dedent(
        """
        class Support(dspy.Module):
            def forward(self, **inputs):
                order = lookup(order_id=inputs["order_id"])
                try:
                    refund(order_id=order["id"])
                except Exception:
                    pass
                return dspy.Prediction(reply=str(order["paid"]))
        """
    )
    flex = dspy.Flex("order_id: int -> reply", tools=[lookup, refund], module_src=source,
                     interpreter_factory=dspy.LocalInterpreter)

    prediction = flex(order_id=7)

    assert prediction.reply == "1250"
    assert prediction.tool_calls == [  # what each call came to, not the coroutine it started as
        {"name": "lookup", "args": {"order_id": 7}, "result": {"id": 7, "paid": 1250}},
        {"name": "refund", "args": {"order_id": 7}, "error": "RuntimeError: refunds are closed"},
    ]
