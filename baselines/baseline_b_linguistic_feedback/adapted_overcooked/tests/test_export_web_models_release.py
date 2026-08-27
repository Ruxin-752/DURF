from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = ROOT / "scripts" / "export_web_models.py"
SPEC = importlib.util.spec_from_file_location("export_web_models_release_test", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
export_web_models = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(export_web_models)


def args(release: str) -> argparse.Namespace:
    return argparse.Namespace(
        classifier_release=release,
        classifier=None,
        classifier_report=None,
        classifier_output_name=None,
        route2_manifest=export_web_models.DEFAULT_ROUTE2,
        output_dir=Path("virtual-browser-model-output"),
    )


class BrowserModelReleaseExportTest(unittest.TestCase):
    def test_shadow_release_only_verifies_existing_route2(self) -> None:
        with (
            patch.object(export_web_models, "parse_args", return_value=args("shadow-preview")),
            patch.object(
                export_web_models,
                "export_classifier",
                return_value=export_web_models.SHADOW_BROWSER_SHA256,
            ),
            patch.object(export_web_models, "export_route2") as export_route2,
            patch.object(
                export_web_models,
                "sha256",
                return_value=export_web_models.ROUTE2_BROWSER_SHA256,
            ) as sha256,
            patch.object(export_web_models, "write_json", return_value="manifest-hash") as write_json,
        ):
            export_web_models.main()

        export_route2.assert_not_called()
        sha256.assert_called_once_with(
            Path("virtual-browser-model-output") / "route2-v5.json"
        )
        manifest_path, manifest = write_json.call_args.args
        self.assertEqual(
            manifest_path,
            Path("virtual-browser-model-output")
            / export_web_models.SHADOW_MANIFEST_FILENAME,
        )
        self.assertEqual(
            manifest["models"]["route2"]["sha256"],
            export_web_models.ROUTE2_BROWSER_SHA256,
        )

    def test_production_release_keeps_route2_export(self) -> None:
        with (
            patch.object(export_web_models, "parse_args", return_value=args("production")),
            patch.object(
                export_web_models,
                "export_classifier",
                return_value="a" * 64,
            ),
            patch.object(
                export_web_models,
                "export_route2",
                return_value=export_web_models.ROUTE2_BROWSER_SHA256,
            ) as export_route2,
            patch.object(export_web_models, "write_json", return_value="manifest-hash"),
        ):
            export_web_models.main()

        export_route2.assert_called_once_with(
            export_web_models.DEFAULT_ROUTE2.resolve(),
            Path("virtual-browser-model-output") / "route2-v5.json",
        )


if __name__ == "__main__":
    unittest.main()
