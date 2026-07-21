"""Full coverage for the ide_plugin package (pure, offline logic)."""
import asyncio

from ide_plugin import (
    IDEType,
    CompletionType,
    DiagnosticSeverity,
    CompletionItem,
    Diagnostic,
    CodeAction,
    HoverInfo,
    Definition,
    Reference,
    VSCodePlugin,
    JetBrainsPlugin,
    VimPlugin,
    AICodeAssistant,
    create_vscode_plugin,
    create_assistant,
    quick_review,
    quick_explain,
    quick_tests,
)


def test_enums():
    assert IDEType.VSCODE.value == "vscode"
    assert CompletionType.METHOD.value == "method"
    assert DiagnosticSeverity.WARNING.value == "warning"


def test_dataclasses_defaults():
    ci = CompletionItem(label="x", detail="d", completion_type=CompletionType.KEYWORD)
    assert ci.confidence == 0.8 and ci.deprecated is False
    diag = Diagnostic(message="m", severity=DiagnosticSeverity.ERROR, line=1, column=2)
    assert diag.source == "ai-assistant" and diag.related_information == []
    action = CodeAction(title="t", description="d")
    assert action.is_preferred is False and action.diagnostics == []
    hover = HoverInfo(contents="c")
    assert hover.language == "plaintext"
    d = Definition(uri="u", line=1, column=0, name="n")
    r = Reference(uri="u", line=1, column=0, name="n")
    assert d.name == "n" and r.context == ""


def test_plugin_construction():
    vs = VSCodePlugin(project_path="/tmp/proj")
    assert vs.ide_type == IDEType.VSCODE
    assert vs.project_path == "/tmp/proj"
    jb = JetBrainsPlugin(project_path="/tmp/proj")
    assert jb.ide_type == IDEType.JETBRAINS
    vim = VimPlugin(project_path="/tmp/proj")
    assert vim.ide_type == IDEType.VIM


def test_vscode_detect_workspace_and_completions():
    vs = VSCodePlugin(project_path="/tmp/myproj")
    asyncio.run(vs.connect())
    assert vs.workspace_info["root"] == "/tmp/myproj"
    assert vs.workspace_info["name"] == "myproj"
    assert asyncio.run(vs.get_completions("f.py", 0, 0)) == []
    assert asyncio.run(vs.get_diagnostics("f.py")) == []
    assert asyncio.run(vs.get_hover("f.py", 0, 0)) is None


def test_assistant_plugin_registration():
    a = AICodeAssistant(project_path="/tmp")
    vs = VSCodePlugin("/tmp")
    a.register_plugin(vs)
    assert a.get_plugin(IDEType.VSCODE) is vs
    assert a.get_plugin(IDEType.VIM) is None


def test_ai_completions_keyword(tmp_path):
    f = tmp_path / "code.py"
    f.write_text("def \n")
    a = AICodeAssistant(project_path=str(tmp_path))
    comps = asyncio.run(a.get_completions(str(f), 0, len("def ")))
    assert comps
    assert all(isinstance(c, CompletionItem) for c in comps)
    assert any("def" in c.detail for c in comps)


def test_ai_completions_method(tmp_path):
    f = tmp_path / "code.py"
    f.write_text("mylist.\n")
    a = AICodeAssistant(project_path=str(tmp_path))
    comps = asyncio.run(a.get_completions(str(f), 0, len("mylist.")))
    labels = [c.label for c in comps]
    assert "append" in labels and "sort" in labels


def test_ai_completions_missing_file():
    a = AICodeAssistant()
    comps = asyncio.run(a.get_completions("/no/such/file.py", 0, 0))
    assert comps == []


def test_explain_code():
    code = "def foo():\n    return 1\nclass Bar:\n    pass\nimport os"
    text = asyncio.run(AICodeAssistant().explain_code(code))
    assert "foo" in text
    assert "Bar" in text
    assert "os" in text


def test_explain_code_empty():
    text = asyncio.run(AICodeAssistant().explain_code("   "))
    assert text == "Code snippet"


def test_suggest_refactoring():
    code = "\n".join([f"    x{i} = compute_something_long({i})" for i in range(25)])
    code += "\nvalue = 123456\n# TODO fix this\n"
    actions = asyncio.run(AICodeAssistant().suggest_refactoring(code))
    titles = [a.title for a in actions]
    assert "Extract method" in titles
    assert "Extract constants" in titles
    assert "Address TODOs" in titles


def test_detect_bugs():
    code = "try:\n    pass\nexcept:\n    pass\nfrom os import *\n"
    diags = asyncio.run(AICodeAssistant().detect_bugs(code))
    codes = [d.code for d in diags]
    assert "W0702" in codes
    assert "W0401" in codes
    assert all(isinstance(d, Diagnostic) for d in diags)


def test_generate_tests():
    code = "def add(a, b):\n    return a + b\ndef _private():\n    pass\n"
    tests = asyncio.run(AICodeAssistant().generate_tests(code))
    assert "class TestAdd" in tests
    assert "_private" not in tests
    assert "import pytest" in tests


def test_generate_docs():
    code = "def foo(a, b):\n    return a\nclass Bar:\n    pass\n"
    docs = asyncio.run(AICodeAssistant().generate_docs(code))
    assert "def foo(a, b):" in docs
    assert "class Bar:" in docs
    assert '"""' in docs


def test_review_code():
    code = "try:\n    pass\nexcept:\n    pass\n"
    review = asyncio.run(AICodeAssistant().review_code(code))
    assert 0 <= review["score"] <= 100
    assert isinstance(review["issues"], list) and review["issues"]
    assert review["recommendation"] in ("Good", "Needs improvement", "Poor")


def test_detect_language():
    a = AICodeAssistant()
    assert a._detect_language("x.py") == "python"
    assert a._detect_language("x.ts") == "typescript"
    assert a._detect_language("x.unknown") == "unknown"


def test_index_and_stats(tmp_path):
    (tmp_path / "a.py").write_text("print(1)\nprint(2)\n")
    (tmp_path / "b.js").write_text("console.log(1)\n")
    (tmp_path / "notes.txt").write_text("ignore me\n")
    a = AICodeAssistant(project_path=str(tmp_path))
    a.index_project()
    stats = a.get_project_stats()
    assert stats["total_files"] == 2
    assert stats["languages"].get("python") == 1
    assert stats["languages"].get("javascript") == 1
    assert stats["total_lines"] >= 3


def test_convenience_factories():
    assert isinstance(create_vscode_plugin("/tmp"), VSCodePlugin)
    assert isinstance(create_assistant("/tmp"), AICodeAssistant)


def test_quick_helpers():
    code = "def foo():\n    return 1\n"
    assert isinstance(asyncio.run(quick_explain(code)), str)
    assert "class TestFoo" in asyncio.run(quick_tests(code))
    review = asyncio.run(quick_review(code))
    assert "score" in review
