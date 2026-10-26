#!/usr/bin/env python3
"""Install the known-good Humble tf2 release locally, without changing apt packages.

The ROS snapshot signature and every downloaded package hash are verified.
"""

import gzip
import hashlib
import shutil
import subprocess
import tempfile
import urllib.request
from pathlib import Path


SNAPSHOT = 'http://snapshots.ros.org/humble/2026-07-02/ubuntu'
KEY_URL = ('https://keyserver.ubuntu.com/pks/lookup?op=get&search='
           '0x4B63CF8FDE49746E98FA01DDAD19BAB3CBF125EA')
KEY_FINGERPRINT = '4B63CF8FDE49746E98FA01DDAD19BAB3CBF125EA'
PACKAGES = (
    'ros-humble-tf2', 'ros-humble-tf2-bullet', 'ros-humble-tf2-eigen',
    'ros-humble-tf2-eigen-kdl', 'ros-humble-tf2-geometry-msgs',
    'ros-humble-tf2-kdl', 'ros-humble-tf2-msgs', 'ros-humble-tf2-py',
    'ros-humble-tf2-ros', 'ros-humble-tf2-ros-py',
    'ros-humble-tf2-sensor-msgs', 'ros-humble-tf2-tools',
)
DESTINATION = (Path.home() / '.local/share/smart_vision_amr/tf2-0.25.20')


def download(url, destination):
    with urllib.request.urlopen(url, timeout=30) as response:
        destination.write_bytes(response.read())


def verify_digest(path, expected):
    actual = hashlib.sha256(path.read_bytes()).hexdigest()
    if actual != expected:
        raise RuntimeError(f'SHA256 mismatch for {path.name}')


def package_index_digest(inrelease):
    lines = inrelease.splitlines()
    start = lines.index('SHA256:') + 1
    for line in lines[start:]:
        if not line.startswith(' '):
            break
        digest, _, name = line.split()
        if name == 'main/binary-amd64/Packages.gz':
            return digest
    raise RuntimeError('Snapshot has no amd64 package index')


def read_packages(index):
    entries = {}
    for block in gzip.decompress(index).decode().split('\n\n'):
        fields = {}
        for line in block.splitlines():
            if line and not line.startswith(' ') and ': ' in line:
                key, value = line.split(': ', 1)
                fields[key] = value
        name = fields.get('Package')
        if name in PACKAGES and fields.get('Version', '').startswith('0.25.20-'):
            entries[name] = fields
    if set(entries) != set(PACKAGES):
        raise RuntimeError(f'Snapshot is missing: {sorted(set(PACKAGES) - set(entries))}')
    return entries


def main():
    if (DESTINATION / 'root/opt/ros/humble/lib/libtf2.so').exists() and (
            DESTINATION / 'root/opt/ros/humble/lib/libtf2_ros.so').exists():
        print(f'tf2 0.25.20 overlay already installed: {DESTINATION}')
        return
    if DESTINATION.exists():
        raise RuntimeError(f'Remove incomplete overlay before retrying: {DESTINATION}')
    DESTINATION.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=DESTINATION.parent) as temp_name:
        temp = Path(temp_name)
        key = temp / 'snapshot.asc'
        keyring = temp / 'snapshot.gpg'
        inrelease = temp / 'InRelease'
        index = temp / 'Packages.gz'
        overlay = temp / 'overlay'
        root = overlay / 'root'
        root.mkdir(parents=True)

        download(KEY_URL, key)
        fingerprints = subprocess.check_output(
            ['gpg', '--show-keys', '--with-colons', str(key)], text=True)
        if f'fpr:::::::::{KEY_FINGERPRINT}:' not in fingerprints:
            raise RuntimeError('Unexpected ROS snapshot signing key')
        subprocess.run(
            ['gpg', '--no-default-keyring', '--keyring', str(keyring), '--import', str(key)],
            check=True, stdout=subprocess.DEVNULL)
        download(f'{SNAPSHOT}/dists/jammy/InRelease', inrelease)
        subprocess.run(['gpgv', '--keyring', str(keyring), str(inrelease)], check=True)

        download(f'{SNAPSHOT}/dists/jammy/main/binary-amd64/Packages.gz', index)
        verify_digest(index, package_index_digest(inrelease.read_text()))
        entries = read_packages(index.read_bytes())
        for name in PACKAGES:
            entry = entries[name]
            deb = temp / Path(entry['Filename']).name
            download(f"{SNAPSHOT}/{entry['Filename']}", deb)
            verify_digest(deb, entry['SHA256'])
            subprocess.run(['dpkg-deb', '-x', str(deb), str(root)], check=True)
            print(f"Verified {name} {entry['Version']}")
        shutil.move(str(overlay), str(DESTINATION))
    print(f'Installed tf2 0.25.20 overlay: {DESTINATION}')


if __name__ == '__main__':
    main()
