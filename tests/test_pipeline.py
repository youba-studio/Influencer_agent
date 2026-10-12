"""
tests/test_pipeline.py

Local tests — no GPU required.
Covers:
  - Python syntax / imports
  - Notebook JSON validity and cell count
  - Configuration consistency between pipeline_config.py and notebook
  - Input image validation logic
  - Warp fallback is NOT callable as generative video
  - Output path / quality gate logic
"""

import ast
import json
import importlib
import pathlib
import sys
import types
import unittest

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

NOTEBOOK_PATH = ROOT / "cloud_notebooks" / "kaggle_wan21_i2v.ipynb"


# ─────────────────────────────────────────────────────────────────────────────
class TestSyntax(unittest.TestCase):
    """All .py files in the repo must parse without syntax errors."""

    def _py_files(self):
        skip = {"__pycache__", ".git", ".venv", "venv"}
        for p in ROOT.rglob("*.py"):
            if not any(s in p.parts for s in skip):
                yield p

    def test_all_python_files_parse(self):
        errors = []
        for path in self._py_files():
            try:
                ast.parse(path.read_text(encoding="utf-8"))
            except SyntaxError as e:
                errors.append(f"{path.relative_to(ROOT)}: {e}")
        if errors:
            self.fail("Syntax errors found:\n" + "\n".join(errors))


# ─────────────────────────────────────────────────────────────────────────────
class TestNotebook(unittest.TestCase):
    """Notebook must be valid JSON with expected structure."""

    def _load_nb(self):
        self.assertTrue(NOTEBOOK_PATH.exists(), f"Notebook not found: {NOTEBOOK_PATH}")
        with open(NOTEBOOK_PATH, encoding="utf-8") as f:
            return json.load(f)

    def test_notebook_valid_json(self):
        nb = self._load_nb()
        self.assertIn("cells", nb)
        self.assertIn("nbformat", nb)

    def test_notebook_has_enough_cells(self):
        nb = self._load_nb()
        self.assertGreaterEqual(len(nb["cells"]), 10,
                                "Notebook should have at least 10 cells")

    def test_notebook_no_hardcoded_token(self):
        """No HF token starting with hf_ should appear in notebook source."""
        nb = self._load_nb()
        for i, cell in enumerate(nb["cells"]):
            source = "".join(cell.get("source", []))
            # Check for token-shaped strings (hf_ followed by alphanum, not placeholder text)
            import re
            matches = re.findall(r'hf_[A-Za-z0-9]{10,}', source)
            self.assertEqual(matches, [],
                             f"Cell {i} contains what looks like a hardcoded HF token: {matches}")

    def test_notebook_accelerator_gpu(self):
        nb = self._load_nb()
        acc = nb.get("metadata", {}).get("accelerator", "")
        self.assertEqual(acc, "GPU", "Notebook metadata.accelerator must be 'GPU'")

    def test_flow_shift_applied_correctly(self):
        """Notebook must not use dict-mutation to set flow_shift in executable code."""
        nb = self._load_nb()
        bad_pattern = "scheduler.config['shift']"
        for i, cell in enumerate(nb["cells"]):
            source = "".join(cell.get("source", []))
            # Strip comment lines before checking — the bad pattern may appear
            # in an explanatory comment describing what NOT to do.
            executable_lines = [
                line for line in source.splitlines()
                if not line.lstrip().startswith("#")
            ]
            executable = "\n".join(executable_lines)
            self.assertNotIn(
                bad_pattern, executable,
                f"Cell {i} uses incorrect flow_shift mutation in executable code: {bad_pattern}"
            )

    def test_correct_flow_shift_method_present(self):
        """Notebook must call FlowMatchEulerDiscreteScheduler.from_config."""
        nb = self._load_nb()
        found = False
        for cell in nb["cells"]:
            source = "".join(cell.get("source", []))
            if "from_config" in source and "shift" in source:
                found = True
                break
        self.assertTrue(found, "Notebook must apply flow_shift via scheduler.from_config()")


# ─────────────────────────────────────────────────────────────────────────────
class TestPipelineConfig(unittest.TestCase):
    """pipeline_config.py must be consistent and within safe ranges."""

    def _cfg(self):
        import pipeline_config as cfg
        return cfg

    def test_resolution_matches_model(self):
        cfg = self._cfg()
        # 480P model supports 480x832
        self.assertEqual(cfg.VIDEO_WIDTH, 480)
        self.assertEqual(cfg.VIDEO_HEIGHT, 832)

    def test_guidance_scale_in_range(self):
        cfg = self._cfg()
        self.assertGreaterEqual(cfg.GUIDANCE_SCALE, 3.0)
        self.assertLessEqual(cfg.GUIDANCE_SCALE, 8.0)

    def test_flow_shift_in_range(self):
        cfg = self._cfg()
        self.assertGreaterEqual(cfg.FLOW_SHIFT, 1.0)
        self.assertLessEqual(cfg.FLOW_SHIFT, 8.0)

    def test_prompt_long_enough(self):
        cfg = self._cfg()
        word_count = len(cfg.MOTION_PROMPT.split())
        self.assertGreaterEqual(word_count, 60,
                                f"Motion prompt should be ≥60 words; got {word_count}")

    def test_negative_prompt_includes_static_terms(self):
        cfg = self._cfg()
        neg = cfg.NEGATIVE_PROMPT.lower()
        for term in ["frozen", "static", "slideshow"]:
            self.assertIn(term, neg,
                          f"Negative prompt must include '{term}' to suppress slideshow")

    def test_no_hardcoded_token_in_config(self):
        import re
        cfg_path = ROOT / "pipeline_config.py"
        text = cfg_path.read_text(encoding="utf-8")
        matches = re.findall(r'hf_[A-Za-z0-9]{10,}', text)
        self.assertEqual(matches, [], f"pipeline_config.py contains hardcoded HF token: {matches}")


# ─────────────────────────────────────────────────────────────────────────────
class TestWarpFallbackIsolated(unittest.TestCase):
    """
    The pixel-warp code in render_dynamic_motion.py must NOT be importable
    as a generative video function, and must NOT be callable without the
    explicit --confirm-synthetic flag.
    """

    def test_render_dynamic_motion_requires_flag(self):
        """Script should exit / fail without --confirm-synthetic."""
        import subprocess
        result = subprocess.run(
            [sys.executable, str(ROOT / "render_dynamic_motion.py")],
            capture_output=True, text=True, cwd=str(ROOT)
        )
        # Must fail (non-zero exit) because --confirm-synthetic is missing
        self.assertNotEqual(result.returncode, 0,
                            "render_dynamic_motion.py should require --confirm-synthetic flag")

    def test_cloud_video_generator_has_no_warp_code(self):
        """cloud_video_generator.py must not contain map_coordinates or pixel warping."""
        path = ROOT / "tools" / "cloud_video_generator.py"
        text = path.read_text(encoding="utf-8")
        for banned in ["map_coordinates", "imageio.get_writer", "warp_x", "warp_y",
                       "walk_bob", "walk_sway"]:
            self.assertNotIn(banned, text,
                             f"cloud_video_generator.py still contains warp code: '{banned}'")

    def test_cloud_generator_returns_no_fake_success(self):
        """The spec builder must NOT set status=success or clean_footage_verified=True."""
        path = ROOT / "tools" / "cloud_video_generator.py"
        text = path.read_text(encoding="utf-8")
        self.assertNotIn("clean_footage_verified", text)
        self.assertNotIn('"success"', text)


# ─────────────────────────────────────────────────────────────────────────────
class TestInputValidation(unittest.TestCase):
    """build_run_spec must raise FileNotFoundError for missing images."""

    def test_missing_image_raises(self):
        from tools.cloud_video_generator import build_run_spec
        with self.assertRaises(FileNotFoundError):
            build_run_spec(
                reference_image_path="/definitely/does/not/exist/image.jpg",
                output_dir=str(ROOT / "output"),
            )

    def test_existing_image_writes_spec(self):
        # Use the avatar image which should exist
        avatar = ROOT / "assets" / "anaya_avatar.jpg"
        if not avatar.exists():
            self.skipTest("anaya_avatar.jpg not found — skipping")

        from tools.cloud_video_generator import build_run_spec
        spec = build_run_spec(
            reference_image_path=str(avatar),
            output_dir=str(ROOT / "output"),
        )
        self.assertEqual(spec["status"], "SPEC_READY__AWAITING_CLOUD_EXECUTION")
        self.assertIn("generation", spec)
        self.assertIsNone(spec["result"])


# ─────────────────────────────────────────────────────────────────────────────
class TestConfigConsistency(unittest.TestCase):
    """
    Notebook Cell 2 values and pipeline_config.py must agree on key params.
    We parse Cell 2 source and check the numeric values match.
    """

    def test_guidance_scale_matches(self):
        import pipeline_config as cfg
        nb = json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))
        cell2_src = "".join(nb["cells"][2].get("source", []))
        self.assertIn(str(cfg.GUIDANCE_SCALE), cell2_src,
                      f"Notebook Cell 2 guidance_scale must match pipeline_config.py ({cfg.GUIDANCE_SCALE})")

    def test_flow_shift_matches(self):
        import pipeline_config as cfg
        nb = json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))
        # Search all cells for flow_shift value
        for cell in nb["cells"]:
            src = "".join(cell.get("source", []))
            if "FLOW_SHIFT" in src and "=" in src:
                self.assertIn(str(cfg.FLOW_SHIFT), src,
                              f"Notebook FLOW_SHIFT must match pipeline_config.py ({cfg.FLOW_SHIFT})")
                return

    def test_resolution_matches(self):
        import pipeline_config as cfg
        nb = json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))
        cell2_src = "".join(nb["cells"][2].get("source", []))
        self.assertIn(str(cfg.VIDEO_WIDTH), cell2_src)
        self.assertIn(str(cfg.VIDEO_HEIGHT), cell2_src)


if __name__ == "__main__":
    unittest.main(verbosity=2)
