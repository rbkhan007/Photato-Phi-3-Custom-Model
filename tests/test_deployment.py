"""Full coverage for the deployment package (docker + model registry)."""
import json
import os
from pathlib import Path

from deployment.docker_setup import DockerSetup
from deployment.registry import ModelRegistry, ModelInfo, ModelVersion


def test_docker_generate_dockerfile(tmp_path):
    ds = DockerSetup(output_dir=str(tmp_path))
    dockerfile = ds.generate_dockerfile(model_path="./phi3-q4.gguf", port=8080, gpu=True)
    content = Path(dockerfile).read_text()
    assert "FROM" in content.upper()


def test_docker_generate_compose(tmp_path):
    ds = DockerSetup(output_dir=str(tmp_path))
    compose = ds.generate_docker_compose(model_path="./phi3-q4.gguf", port=8080, gpu=True)
    content = Path(compose).read_text()
    assert "services" in content


def test_docker_generate_prometheus(tmp_path):
    ds = DockerSetup(output_dir=str(tmp_path))
    cfg = ds.generate_prometheus_config()
    assert isinstance(str(cfg), str)
    content = Path(cfg).read_text()
    assert "global" in content.lower() or "scrape" in content.lower()


def test_docker_generate_all(tmp_path):
    ds = DockerSetup(output_dir=str(tmp_path))
    result = ds.generate_all(model_path="./m.gguf", port=8080, gpu=True)
    assert isinstance(result, dict)
    assert os.path.exists(os.path.join(str(tmp_path), "Dockerfile"))


def test_registry_register_and_get(tmp_path):
    reg = ModelRegistry(registry_dir=str(tmp_path))
    reg.register_model(
        "phi3-custom",
        "./phi3-q4.gguf",
        version="1.0.0",
        description="test",
        tags=["phi3"],
    )
    info = reg.get_model("phi3-custom")
    assert isinstance(info, ModelInfo)
    assert info.name == "phi3-custom"


def test_registry_latest_version(tmp_path):
    reg = ModelRegistry(registry_dir=str(tmp_path))
    reg.register_model("m", "./m.gguf", version="1.0.0")
    reg.register_model("m", "./m.gguf", version="1.1.0")
    latest = reg.get_latest_version("m")
    assert latest is not None and latest.version == "1.1.0"


def test_registry_list_and_search(tmp_path):
    reg = ModelRegistry(registry_dir=str(tmp_path))
    reg.register_model("alpha", "./a.gguf", version="1.0.0", tags=["x"])
    models = reg.list_models()
    assert any(d["name"] == "alpha" for d in models)
    found = reg.search_models("alpha")
    assert any(d["name"] == "alpha" for d in found)


def test_registry_compare_versions(tmp_path):
    reg = ModelRegistry(registry_dir=str(tmp_path))
    reg.register_model("m", "./m.gguf", version="1.0.0", description="old")
    reg.register_model("m", "./m.gguf", version="1.1.0", description="new")
    comparison = reg.compare_versions("m", "1.0.0", "1.1.0")
    assert isinstance(comparison, dict)


def test_registry_export(tmp_path):
    reg = ModelRegistry(registry_dir=str(tmp_path))
    reg.register_model("m", "./m.gguf", version="1.0.0")
    data = reg.export_registry()
    assert isinstance(data, str)
