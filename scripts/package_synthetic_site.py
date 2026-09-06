"""Build with an immutable source receipt, then package only verified Worker assets."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import stat
import subprocess
import tarfile

RECEIPT = '.synthetic-build-receipt.json'
BUILD_ROOTS = ('.openai', 'client', 'server')
GENERATED_ROOTS = {'.git', 'node_modules', 'dist', '.wrangler', '.next', '.vinext', 'coverage', 'out'}
GENERATED_FILES = {'next-env.d.ts', 'tsconfig.tsbuildinfo'}


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical_digest(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def is_link(path: Path) -> bool:
    # Python <3.12 has no Path.is_junction; Windows reparse attributes still
    # identify junctions before os.walk or cleanup can enter them.
    attributes = getattr(path.lstat(), 'st_file_attributes', 0)
    return path.is_symlink() or bool(attributes & getattr(stat, 'FILE_ATTRIBUTE_REPARSE_POINT', 0))


def unsafe_name(path: Path) -> bool:
    return any((part.startswith('.env') and part != '.env.example') or
               part.startswith('.dev.vars') or part in {'node_modules', '.wrangler', 'private'}
               for part in path.parts) or bool(re.search(r'\.(?:sqlite|db|sqlite3)(?:-|$)', path.name, re.I))


def files_under(root: Path):
    if is_link(root):
        raise ValueError(f'Linked directory is not a release input: {root}')
    for folder, directories, files in os.walk(root, followlinks=False):
        for name in directories:
            if is_link(Path(folder) / name):
                raise ValueError(f'Linked directory is not a release input: {Path(folder) / name}')
        for name in files:
            path = Path(folder) / name
            if is_link(path):
                raise ValueError(f'Linked file is not a release input: {path}')
            yield path


def source_inventory(source: Path):
    inventory = []
    for child in sorted(source.iterdir()):
        if child.name in GENERATED_ROOTS or child.name in GENERATED_FILES:
            continue
        if is_link(child):
            raise ValueError('Linked source input: ' + str(child))
        paths = files_under(child) if child.is_dir() else [child]
        for path in paths:
            relative = path.relative_to(source)
            if unsafe_name(relative):
                raise ValueError('Unsafe source input: ' + relative.as_posix())
            inventory.append({'path': relative.as_posix(), 'sha256': digest(path)})
    return sorted(inventory, key=lambda row: row['path'])


def clean_commit(source: Path) -> str:
    status = subprocess.check_output(['git', 'status', '--porcelain', '--untracked-files=all'], cwd=source, text=True)
    if status.strip():
        raise ValueError('Commit the exact source before building or packaging')
    return subprocess.check_output(['git', 'rev-parse', '--verify', 'HEAD'], cwd=source, text=True).strip()


def build_inventory(build: Path):
    inventory = []
    for base in BUILD_ROOTS:
        root = build / base
        if not root.is_dir():
            raise ValueError('Missing build directory: ' + base)
        for path in files_under(root):
            relative = path.relative_to(build)
            if unsafe_name(relative) or any(part.startswith('.env') or part.startswith('.dev.vars') for part in relative.parts):
                raise ValueError('Unsafe build member: ' + relative.as_posix())
            inventory.append({'path': relative.as_posix(), 'sha256': digest(path), 'bytes': path.stat().st_size})
    if not (build / 'server/index.js').is_file():
        raise ValueError('Missing completed Worker build')
    return sorted(inventory, key=lambda row: row['path'])


def verified_build(source: Path):
    """Only this before/after build operation creates a source-binding receipt."""
    commit = clean_commit(source)
    inputs = source_inventory(source)
    build = source / 'dist'
    # One explicitly bounded target, never a junction or computed outside source.
    if build.exists():
        if is_link(build) or build.resolve() != source.resolve() / 'dist':
            raise ValueError('Unsafe build cleanup target')
        shutil.rmtree(build)
    npm = shutil.which('npm.cmd' if os.name == 'nt' else 'npm')
    if not npm:
        raise ValueError('npm is unavailable')
    subprocess.run([npm, 'run', 'build'], cwd=source, check=True)
    if clean_commit(source) != commit or source_inventory(source) != inputs:
        raise ValueError('Source changed during build; no receipt was issued')
    outputs = build_inventory(build)
    receipt = {'schema_version': 'durf-synthetic-build-receipt-v1', 'source_commit': commit,
               'source_tree_sha256': canonical_digest(inputs), 'source_files': inputs,
               'dist_sha256': canonical_digest(outputs), 'dist_files': outputs}
    (build / RECEIPT).write_text(json.dumps(receipt, sort_keys=True, indent=2) + '\n', encoding='utf-8')
    return receipt


def known_secrets():
    values = []
    local_env = Path(__file__).resolve().parents[1] / 'web/.env.local'
    if local_env.is_file():
        for line in local_env.read_text(encoding='utf-8-sig').splitlines():
            match = re.match(r'^\s*(?:export\s+)?([A-Za-z_][\w]*)\s*=\s*(.*)$', line)
            if match and re.search(r'TOKEN|SECRET|PASSWORD|API_KEY|PRIVATE_KEY', match[1], re.I):
                value = match[2].strip().strip('"\'')
                if len(value) >= 8:
                    values.append(value.encode())
    for name, value in os.environ.items():
        if re.search(r'TOKEN|SECRET|PASSWORD|API_KEY|PRIVATE_KEY', name, re.I) and len(value) >= 8:
            values.append(value.encode())
    return values


def archive_member_path(build_path: str) -> str:
    """Sites reads hosting metadata at root and discovers dist/server/index.js."""
    if build_path.startswith('.openai/'):
        return build_path
    if build_path.startswith(('client/', 'server/')):
        return 'dist/' + build_path
    raise ValueError('Unsupported archive input: ' + build_path)


def package(source: Path, archive: Path, rebuild: bool = False):
    source, archive = source.resolve(), archive.resolve()
    build = source / 'dist'
    if archive.exists() or archive.with_suffix('.inventory.json').exists():
        raise ValueError('Use a new archive path; do not overwrite saved release artifacts')
    if archive.is_relative_to(source):
        raise ValueError('Write the archive outside the source checkout')
    if rebuild:
        verified_build(source)
    receipt = json.loads((build / RECEIPT).read_text(encoding='utf-8'))
    commit, inputs, inventory = clean_commit(source), source_inventory(source), build_inventory(build)
    if (receipt.get('schema_version') != 'durf-synthetic-build-receipt-v1' or receipt.get('source_commit') != commit or
            receipt.get('source_files') != inputs or receipt.get('source_tree_sha256') != canonical_digest(inputs) or
            receipt.get('dist_files') != inventory or receipt.get('dist_sha256') != canonical_digest(inventory)):
        raise ValueError('Build receipt does not match current source and assets; rebuild with --build')
    hosting = json.loads((build / '.openai/hosting.json').read_text(encoding='utf-8'))
    if hosting != json.loads((source / '.openai/hosting.json').read_text(encoding='utf-8')):
        raise ValueError('Hosting metadata mismatch')
    secrets = known_secrets()
    for row in inventory:
        data = (build / row['path']).read_bytes()
        if any(value in data for value in secrets):
            raise ValueError('Local secret found in build: ' + row['path'])
    # Keep the build receipt's relative paths unchanged. Only the archive layout
    # is mapped: metadata at root; sibling client/server trees beneath dist/.
    archive_inventory = [dict(row, path=archive_member_path(row['path']), build_path=row['path']) for row in inventory]
    with tarfile.open(archive, 'x:gz') as tar:
        for row in archive_inventory:
            tar.add(build / row['build_path'], arcname=row['path'], recursive=False)
    with tarfile.open(archive, 'r:gz') as tar:
        if len(tar.getmembers()) != len(archive_inventory) or set(tar.getnames()) != {row['path'] for row in archive_inventory}:
            raise ValueError('Archive inventory mismatch')
        for row in archive_inventory:
            member = tar.getmember(row['path'])
            if not member.isfile() or member.size != row['bytes'] or hashlib.sha256(tar.extractfile(member).read()).hexdigest() != row['sha256']:
                raise ValueError('Archive content mismatch')
    result = {'source_commit': commit, 'source_tree_sha256': receipt['source_tree_sha256'],
              'build_receipt_sha256': digest(build / RECEIPT), 'project_id': hosting['project_id'],
              'archive': str(archive), 'archive_sha256': digest(archive), 'files': len(inventory),
              'bytes': archive.stat().st_size, 'secret_matches': 0,
              'worker_entrypoint': 'dist/server/index.js', 'hosting_metadata': '.openai/hosting.json',
              'inventory': archive_inventory}
    archive.with_suffix('.inventory.json').write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--archive', type=Path, required=True)
    parser.add_argument('--build', action='store_true', help='Build the clean committed source and issue a source/dist receipt before packaging')
    args = parser.parse_args()
    result = package(args.source, args.archive, args.build)
    print(json.dumps({key: value for key, value in result.items() if key != 'inventory'}))


if __name__ == '__main__':
    main()

