"""Coverage for model_recommendations pure-Python logic."""
import pytest

from model_recommendations import (
    HardwareTier,
    QuantLevel,
    SystemInfo,
    ModelRecommendation,
    HardwareProfile,
    HardwareDetector,
    ModelDatabase,
    ModelRecommender,
    PendriveDeployer,
    generate_report,
)


def make_system(ram=8.0, vram=0.0, tier=None, cuda=False, gpu="None"):
    info = SystemInfo(
        os="Linux",
        arch="x86_64",
        python_version="3.11.0",
        cpu_count=8,
        cpu_name="TestCPU",
        ram_gb=ram,
        ram_available_gb=ram / 2,
        gpu_name=gpu,
        gpu_vram_gb=vram,
        cuda_available=cuda,
        storage_available_gb=100.0,
    )
    info.hardware_tier = tier or HardwareDetector._classify_tier(info)
    return info


class TestClassifyTier:
    def test_ultra_low(self):
        info = make_system(ram=2.0, vram=0.0)
        assert HardwareDetector._classify_tier(info) == HardwareTier.ULTRA_LOW

    def test_low(self):
        info = make_system(ram=6.0, vram=0.0)
        assert HardwareDetector._classify_tier(info) == HardwareTier.LOW

    def test_medium(self):
        info = make_system(ram=12.0, vram=0.0)
        assert HardwareDetector._classify_tier(info) == HardwareTier.MEDIUM

    def test_high(self):
        info = make_system(ram=24.0, vram=8.0)
        assert HardwareDetector._classify_tier(info) == HardwareTier.HIGH

    def test_ultra_high(self):
        info = make_system(ram=64.0, vram=24.0)
        assert HardwareDetector._classify_tier(info) == HardwareTier.ULTRA_HIGH


class TestHardwareDetectorDetect:
    def test_detect_returns_system_info(self):
        info = HardwareDetector.detect()
        assert isinstance(info, SystemInfo)
        assert info.os
        assert info.cpu_count >= 1
        assert isinstance(info.hardware_tier, HardwareTier)
        assert info.ram_gb >= 0


class TestModelDatabase:
    def test_get_model_known(self):
        m = ModelDatabase.get_model("phi-3-mini-3.8b")
        assert m["name"] == "Phi-3 Mini 3.8B"
        assert m["parameters"] == "3.8B"
        assert m["base_size_gb"] == 7.6

    def test_get_model_unknown(self):
        assert ModelDatabase.get_model("nonexistent") == {}

    def test_list_models(self):
        models = ModelDatabase.list_models()
        assert len(models) == len(ModelDatabase.MODELS)
        assert all("key" in m and "name" in m for m in models)


class TestQuantHelpers:
    def test_quant_ratio_known(self):
        assert ModelRecommender._quant_ratio(QuantLevel.F16) == 1.0
        assert ModelRecommender._quant_ratio(QuantLevel.Q4_K_M) == 0.33

    def test_build_recommendation(self):
        info = make_system(ram=8.0)
        model_info = ModelDatabase.get_model("phi-3-mini-3.8b")
        rec = ModelRecommender._build_recommendation(model_info, QuantLevel.Q4_K_M, info)
        assert isinstance(rec, ModelRecommendation)
        assert rec.model_name == "Phi-3 Mini 3.8B"
        assert rec.quant_level == QuantLevel.Q4_K_M
        # 7.6 * 0.33 = 2.508 -> rounded 2.51
        assert rec.quant_size_gb == pytest.approx(2.51, abs=0.01)
        # ram = quant_size * 1.4
        assert rec.ram_required_gb == pytest.approx(round(rec.quant_size_gb * 1.4, 2), abs=0.01)
        assert rec.vram_required_gb == ModelRecommender.VRAM_REQUIREMENTS[QuantLevel.Q4_K_M]
        assert 0 <= rec.quality_score <= 100
        assert rec.download_url.startswith("https://huggingface.co/")

    def test_build_recommendation_quality_scales_with_quant(self):
        info = make_system(ram=8.0)
        model_info = ModelDatabase.get_model("phi-3-mini-3.8b")
        low = ModelRecommender._build_recommendation(model_info, QuantLevel.Q2_K, info)
        high = ModelRecommender._build_recommendation(model_info, QuantLevel.Q8_0, info)
        assert high.quality_score > low.quality_score
        assert high.quant_size_gb > low.quant_size_gb


class TestRecommend:
    def test_recommend_ultra_low_has_warning(self):
        info = make_system(ram=2.0, vram=0.0)
        profile = ModelRecommender.recommend(info)
        assert isinstance(profile, HardwareProfile)
        assert profile.warnings
        assert profile.recommendations
        assert len(profile.recommendations) <= 6

    def test_recommend_low_tier(self):
        info = make_system(ram=6.0, vram=0.0)
        profile = ModelRecommender.recommend(info)
        assert profile.recommendations
        names = [r.model_name for r in profile.recommendations]
        assert any("Phi-3" in n for n in names)

    def test_recommend_pendrive(self):
        info = make_system(ram=12.0, vram=0.0)
        profile = ModelRecommender.recommend(info)
        assert profile.pendrive_recommendations
        for rec in profile.pendrive_recommendations:
            assert "pendrive" in rec.notes.lower()

    def test_recommend_tips_present(self):
        info = make_system(ram=8.0)
        profile = ModelRecommender.recommend(info)
        assert any("Pendrive budget" in t for t in profile.tips)
        assert any("Ollama" in t for t in profile.tips)

    def test_recommend_cuda_tip(self):
        info = make_system(ram=24.0, vram=8.0, cuda=True, gpu="NVIDIA RTX 4090")
        profile = ModelRecommender.recommend(info)
        assert any("NVIDIA" in t for t in profile.tips)

    def test_recommend_max_size_filter(self):
        info = make_system(ram=64.0, vram=24.0)
        limit = 2.6
        profile = ModelRecommender.recommend(info, max_model_size_gb=limit)
        for rec in profile.recommendations:
            assert rec.quant_size_gb <= limit + 1e-6

    def test_estimate_size(self):
        size = ModelRecommender._estimate_size("phi-3-mini-3.8b", QuantLevel.Q4_K_M)
        assert isinstance(size, float) and size > 0
        assert ModelRecommender._estimate_size("nonexistent-model", QuantLevel.Q4_K_M) == float("inf")

    def test_recommend_code_use_case(self):
        info = make_system(ram=12.0, vram=0.0)
        profile = ModelRecommender.recommend(info, use_case="code")
        assert profile.recommendations


class TestPendriveDeployer:
    def test_calculate_model_budget(self):
        budget = PendriveDeployer.calculate_model_budget()
        assert budget["total_gb"] == 28.0
        assert budget["system_reserve_gb"] == 5.0
        assert budget["available_for_models_gb"] == 23.0
        assert budget["recommendations"]["q4_k_m_3.8b"]["fits"] is True
        assert budget["recommendations"]["f16_3.8b"]["count"] == int(23.0 / 7.6)

    def test_create_deployment_plan(self):
        info = make_system(ram=12.0, vram=0.0)
        profile = ModelRecommender.recommend(info)
        plan = PendriveDeployer.create_deployment_plan(profile, "/mnt/pendrive")
        assert plan["pendrive_path"] == "/mnt/pendrive"
        assert plan["total_size_gb"] == 28.0
        assert "/models/" in plan["layout"]
        assert plan["setup_commands"]
        assert any("/mnt/pendrive" in cmd for cmd in plan["setup_commands"])
        if profile.pendrive_recommendations:
            assert plan["models"]


class TestGenerateReport:
    def test_generate_report(self):
        info = make_system(ram=8.0, vram=0.0)
        profile = ModelRecommender.recommend(info)
        report = generate_report(profile)
        assert isinstance(report, str)
        assert "MODEL RECOMMENDATIONS" in report
        assert "SYSTEM INFORMATION" in report
        assert "TOP RECOMMENDATIONS" in report
        assert "QUICK START" in report

    def test_generate_report_with_warnings(self):
        info = make_system(ram=2.0, vram=0.0)
        profile = ModelRecommender.recommend(info)
        report = generate_report(profile)
        assert "WARNINGS" in report
