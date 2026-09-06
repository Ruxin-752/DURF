"""Release-boundary tests use isolated fake repositories and synthetic metadata."""
from __future__ import annotations

import argparse
import copy
import importlib.util
import json
from pathlib import Path
import posixpath
import shutil
import subprocess
import tarfile
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]


def module(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / 'scripts' / (name + '.py'))
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


promotion = module('promote_synthetic_web')
packaging = module('package_synthetic_site')


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value) if isinstance(value, (dict, list)) else value, encoding='utf-8')


def git(source, *args):
    return subprocess.check_output(['git', '-c', 'user.name=Release Test', '-c', 'user.email=release-test@example.invalid', *args], cwd=source, text=True, stderr=subprocess.STDOUT)


class FakeWorkspace(unittest.TestCase):
    def setUp(self):
        temporary_root = ROOT / 'tmp'
        temporary_root.mkdir(exist_ok=True)
        self.temporary = tempfile.TemporaryDirectory(prefix='release-script-test-', dir=temporary_root)
        self.root = Path(self.temporary.name).resolve()
        assert self.root.is_relative_to(temporary_root.resolve())

    def tearDown(self):
        # The recursive cleanup target is the checked, generated workspace above.
        assert self.root.is_relative_to((ROOT / 'tmp').resolve())
        self.temporary.cleanup()


class PromotionBoundaryTests(FakeWorkspace):
    def fixture(self):
        args = argparse.Namespace(models=self.root / 'candidate', report=self.root / 'report.json',
                                  dataset=self.root / 'dataset.json', predictions=self.root / 'predictions.json',
                                  stateful_report=self.root / 'stateful.json', release_id='synthetic-test')
        dataset = {'mixed': [{'id': f'M{i:03}'} for i in range(20)], 'uncertain': [{'id': f'U{i:03}'} for i in range(20)]}
        write(args.dataset, dataset)
        for head in promotion.HEADS:
            write(args.models / (head + '.json'), {'training_scope': 'synthetic_only', 'target_semantics': head, 'source': {'model_sha256': '1' * 64}})
        for filename in set(promotion.INFERENCE_SOURCES + promotion.STATEFUL_SOURCES):
            write(self.root / filename, 'synthetic source fixture')
        required = [args.dataset, *(args.models / (head + '.json') for head in promotion.HEADS), *(self.root / name for name in promotion.INFERENCE_SOURCES)]
        provenance = {'candidate_frozen_before_inference': True, 'candidate_training_text_read_by_acceptance_agent': False,
                      'dataset_sha256': promotion.digest(args.dataset), 'hashes': {str(path): promotion.digest(path) for path in required},
                      'speech_model_source_sha256': '1' * 64, 'grounding_model_source_sha256': '1' * 64}
        write(args.predictions, {'provenance': provenance})
        report = {'aggregate_predeclared_pass': True, 'predeclared_gates': dict.fromkeys(['speech_act', 'grounding', 'mixed_components'], True),
                  'dataset_sha256': promotion.digest(args.dataset), 'prediction_sha256': promotion.digest(args.predictions),
                  'mixed': [{'id': row['id'], 'component_set_correct': True} for row in dataset['mixed']], 'mixed_component_accuracy': 1}
        for head, labels in promotion.HEADS.items():
            report[head] = {'n': 150, 'correct': 150, 'accuracy': 1, 'macro_f1': 1,
                            'per_class': {label: {'support': 50, 'recall': 1} for label in labels}}
        write(args.report, report)
        stateful = {'dataset_sha256': promotion.digest(args.dataset), 'predictions_sha256': promotion.digest(args.predictions),
                    'source_hashes': {str(self.root / name): promotion.digest(self.root / name) for name in promotion.STATEFUL_SOURCES},
                    'records': [{'id': row['id'], 'context': context, 'learning_applied': False, 'update_status': 'rejected'}
                                for row in dataset['uncertain'] for context in sorted(promotion.CONTEXTS)]}
        write(args.stateful_report, stateful)
        write(self.root / 'web/public/models/route2-v5.json', {'claim_scope': {'trained_on_synthetic_overcooked_feedback': True}})
        write(self.root / 'web/public/models/manifest-synthetic-v1.json', 'previous release')
        return args, report, stateful

    def assert_refused_without_release_write(self, args):
        with self.assertRaises((ValueError, KeyError, FileNotFoundError)):
            promotion.promote(args, self.root)
        public = self.root / 'web/public/models'
        self.assertEqual((public / 'manifest-synthetic-v1.json').read_text(), 'previous release')
        self.assertEqual(list(public.glob('synthetic-*.json')), [])

    def test_valid_complete_hash_bound_acceptance_promotes(self):
        args, _, _ = self.fixture()
        promotion.promote(args, self.root)
        manifest = promotion.read(self.root / 'web/public/models/manifest-synthetic-v1.json')
        self.assertEqual(manifest['evidence']['stateful_report_sha256'], promotion.digest(args.stateful_report))
        self.assertEqual(len(list((self.root / 'web/public/models').glob('synthetic-*.json'))), 2)

    def test_rejects_failed_missing_truthy_and_nonfinite_gates(self):
        args, original, _ = self.fixture()
        mutations = [lambda r: r.update(aggregate_predeclared_pass=False), lambda r: r.update(predeclared_gates={}),
                     lambda r: r['predeclared_gates'].update(grounding='true'),
                     lambda r: r['speech_act'].update(macro_f1=float('nan')),
                     lambda r: r['grounding'].update(per_class={}),
                     lambda r: next(iter(r['grounding']['per_class'].values())).update(recall=float('nan')),
                     lambda r: r.update(dataset_sha256='0' * 64), lambda r: r.update(mixed=[])]
        for mutate in mutations:
            with self.subTest(mutation=mutate):
                report = copy.deepcopy(original)
                mutate(report)
                write(args.report, report)
                self.assert_refused_without_release_write(args)

    def test_rejects_stale_or_missing_inference_sources(self):
        args, report, _ = self.fixture()
        source = self.root / promotion.INFERENCE_SOURCES[0]
        write(source, 'changed after evaluation')
        self.assert_refused_without_release_write(args)
        write(source, 'synthetic source fixture')
        predictions = promotion.read(args.predictions)
        del predictions['provenance']['hashes'][str(source)]
        write(args.predictions, predictions)
        report['prediction_sha256'] = promotion.digest(args.predictions)
        write(args.report, report)
        self.assert_refused_without_release_write(args)

    def test_rejects_unsafe_empty_duplicate_contexts_or_unbound_stateful_report(self):
        args, _, original = self.fixture()
        mutations = [lambda r: r['records'][0].update(learning_applied=True, update_status='updated'),
                     lambda r: r.update(records=[]),
                     lambda r: r['records'][0].update(context=r['records'][1]['context']),
                     lambda r: r.update(source_hashes={}), lambda r: r.update(predictions_sha256='0' * 64)]
        for mutate in mutations:
            with self.subTest(mutation=mutate):
                stateful = copy.deepcopy(original)
                mutate(stateful)
                write(args.stateful_report, stateful)
                self.assert_refused_without_release_write(args)


class PackagingBoundaryTests(FakeWorkspace):
    def fixture(self):
        source = self.root / 'source'
        source.mkdir()
        write(source / '.gitignore', 'dist/\nnode_modules/\n.env*\n!.env.example\n.dev.vars*\n')
        write(source / '.openai/hosting.json', {'project_id': 'synthetic-test', 'd1': 'DB', 'r2': None})
        write(source / 'main.txt', 'current synthetic source')
        write(source / 'package.json', {'type': 'module', 'scripts': {'build': 'node build.mjs'}})
        write(source / 'build.mjs', "import fs from 'node:fs'; for (const d of ['dist/.openai','dist/client','dist/server']) fs.mkdirSync(d,{recursive:true}); fs.copyFileSync('.openai/hosting.json','dist/.openai/hosting.json'); fs.writeFileSync('dist/server/index.js',fs.readFileSync('main.txt')); fs.writeFileSync('dist/client/index.html','synthetic client'); fs.writeFileSync('dist/server/wrangler.json',JSON.stringify({main:'index.js',assets:{directory:'../client'}}));")
        git(source, 'init', '-q')
        git(source, 'add', '.')
        git(source, 'commit', '-qm', 'synthetic fixture')
        return source

    def test_build_receipt_binds_source_commit_and_dist(self):
        source = self.fixture()
        with patch.object(packaging, 'known_secrets', return_value=[]):
            result = packaging.package(source, self.root / 'valid.tar.gz', rebuild=True)
        self.assertEqual(result['source_commit'], git(source, 'rev-parse', 'HEAD').strip())
        self.assertEqual(result['files'], 4)
        write(source / 'dist/server/index.js', 'stale or replaced build')
        with self.assertRaisesRegex(ValueError, 'receipt'):
            packaging.package(source, self.root / 'changed-dist.tar.gz')

    def test_sites_entrypoint_and_wrangler_relative_assets_survive_archive_mapping(self):
        source = self.fixture()
        packaging.verified_build(source)
        receipt_before = (source / 'dist' / packaging.RECEIPT).read_bytes()
        archive = self.root / 'sites-layout.tar.gz'
        with patch.object(packaging, 'known_secrets', return_value=[]):
            result = packaging.package(source, archive)
        self.assertEqual((source / 'dist' / packaging.RECEIPT).read_bytes(), receipt_before)
        with tarfile.open(archive, 'r:gz') as tar:
            names = set(tar.getnames())
            supported = {'dist/server/index.js', '.output/server/index.js', 'dist/index.js', 'index.js'}
            self.assertEqual(names & supported, {'dist/server/index.js'})
            self.assertIn('.openai/hosting.json', names)
            self.assertNotIn('server/index.js', names)
            self.assertNotIn('dist/.openai/hosting.json', names)
            config_path = 'dist/server/wrangler.json'
            config = json.load(tar.extractfile(config_path))
            main_path = posixpath.normpath(posixpath.join(posixpath.dirname(config_path), config['main']))
            assets_path = posixpath.normpath(posixpath.join(posixpath.dirname(config_path), config['assets']['directory']))
            self.assertEqual(main_path, result['worker_entrypoint'])
            self.assertIn(main_path, names)
            self.assertEqual(assets_path, 'dist/client')
            self.assertIn(assets_path + '/index.html', names)
            for row in result['inventory']:
                self.assertEqual(tar.extractfile(row['path']).read(), (source / 'dist' / row['build_path']).read_bytes())
        self.assertEqual(result['hosting_metadata'], '.openai/hosting.json')

    def test_local_wrangler_state_cannot_be_silently_skipped(self):
        source = self.fixture()
        packaging.verified_build(source)
        write(source / 'dist/server/.wrangler/state/diagnostic.sqlite', 'synthetic local state')
        archive = self.root / 'contaminated.tar.gz'
        with self.assertRaisesRegex(ValueError, 'Unsafe build'):
            packaging.package(source, archive)
        self.assertFalse(archive.exists())

    def test_new_clean_commit_cannot_relabel_old_build(self):
        source = self.fixture()
        packaging.verified_build(source)
        write(source / 'main.txt', 'new source after build')
        git(source, 'add', '.')
        git(source, 'commit', '-qm', 'new clean source')
        with self.assertRaisesRegex(ValueError, 'receipt'):
            packaging.package(source, self.root / 'stale.tar.gz')

    def test_ignored_env_and_dev_vars_are_rejected_before_build(self):
        source = self.fixture()
        for name in ['.env', '.env.production', '.env.production.local', '.dev.vars']:
            path = source / name
            write(path, 'SYNTHETIC_SECRET=synthetic-value')
            with self.assertRaisesRegex(ValueError, 'Unsafe source'):
                packaging.verified_build(source)
            path.unlink()

    def test_known_secret_or_database_cannot_enter_archive(self):
        source = self.fixture()
        packaging.verified_build(source)
        with patch.object(packaging, 'known_secrets', return_value=[b'current synthetic source']):
            with self.assertRaisesRegex(ValueError, 'secret'):
                packaging.package(source, self.root / 'secret.tar.gz')
        write(source / 'dist/client/events.sqlite-wal', 'synthetic fixture')
        with self.assertRaisesRegex(ValueError, 'Unsafe build'):
            packaging.package(source, self.root / 'db.tar.gz')
        self.assertFalse((self.root / 'secret.tar.gz').exists())

    def test_directory_junction_cannot_smuggle_files_into_build(self):
        source = self.fixture()
        packaging.verified_build(source)
        outside = self.root / 'not-build-input'
        write(outside / 'synthetic-secret.txt', 'synthetic fixture')
        junction = source / 'dist/client/linked'
        shell = shutil.which('pwsh') or shutil.which('powershell')
        command = f"New-Item -ItemType Junction -Path '{str(junction).replace(chr(39), chr(39) * 2)}' -Target '{str(outside).replace(chr(39), chr(39) * 2)}' | Out-Null"
        subprocess.run([shell, '-NoProfile', '-Command', command], check=True, capture_output=True)
        with self.assertRaisesRegex(ValueError, 'Linked directory'):
            packaging.build_inventory(source / 'dist')
        self.assertTrue((outside / 'synthetic-secret.txt').exists())

    def test_prepare_removes_stale_sources_and_refuses_ignored_env(self):
        workspace = self.root / 'workspace'
        web = workspace / 'web'
        release = workspace / 'outputs/sites-synthetic-release/source'
        for name in ['app', 'components', 'lib', 'migrations', 'public', 'tests', 'node_modules']:
            (web / name).mkdir(parents=True)
        names = ['.env.example', '.gitignore', '.openai/hosting.json', 'THIRD_PARTY_NOTICES.md', 'package.json', 'package-lock.json',
                 'next.config.ts', 'tsconfig.json', 'vite.config.ts', 'vitest.config.ts', 'worker-configuration.d.ts', 'app/new.ts']
        for name in names:
            write(web / name, 'synthetic fixture')
        (release / '.git').mkdir(parents=True)
        write(release / 'app/removed-route.ts', 'stale')
        write(release / '.env.production', 'SYNTHETIC_SECRET=synthetic-value')
        shell = shutil.which('pwsh') or shutil.which('powershell')
        command = [shell, '-NoProfile', '-ExecutionPolicy', 'Bypass', '-File', str(ROOT / 'scripts/prepare_sites_synthetic_source.ps1'), '-WorkspaceRoot', str(workspace)]
        failed = subprocess.run(command, capture_output=True, text=True)
        self.assertNotEqual(failed.returncode, 0)
        self.assertTrue((release / 'app/removed-route.ts').exists())
        (release / '.env.production').unlink()
        passed = subprocess.run(command, capture_output=True, text=True)
        self.assertEqual(passed.returncode, 0, passed.stdout + passed.stderr)
        self.assertFalse((release / 'app/removed-route.ts').exists())
        self.assertEqual((release / 'app/new.ts').read_bytes(), (web / 'app/new.ts').read_bytes())
        self.assertTrue((release / '.git').is_dir())


if __name__ == '__main__':
    unittest.main()
