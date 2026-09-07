"""Reference capture must launch and identify the selected player."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import reference


class ReferencePlayerTests(unittest.TestCase):
    def player(self, root, native):
        binary = root / "BoplBattle.x86_64"
        runtime = (root / "GameAssembly.so" if native else
                   root / "BoplBattle_Data/Managed/Assembly-CSharp.dll")
        files = [binary, root / "UnityPlayer.so", runtime]
        if native:
            files.append(root / "BoplBattle_Data/il2cpp_data/Metadata/global-metadata.dat")
        for path in files:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(str(path).encode())
        return binary

    def test_capture_selects_explicit_player_before_environment(self):
        for explicit in (False, True):
            with self.subTest(explicit=explicit), tempfile.TemporaryDirectory() as temp:
                root = Path(temp)
                native = self.player(root / "native", True)
                mono = self.player(root / "mono", False)
                output = root / "capture.jsonl"
                replay = root / "episode.rep"
                replay.write_bytes(b"replay")
                selected = mono if explicit else native
                args = argparse.Namespace(output=output, replay=replay, ticks=1, batch=1,
                                          timeout=30, game=mono if explicit else None)

                def capture(command, **kwargs):
                    self.assertEqual(command[0], str(selected))
                    self.assertEqual(kwargs["env"]["BOPL_TRACE"], str(output))
                    rows = [dict(kind="header", schema="bopl-managed-state-v3"),
                            dict(kind="input", index=0), dict(kind="state", index=0),
                            dict(kind="end", ticks=1)]
                    output.write_text("".join(json.dumps(row) + "\n" for row in rows))

                with patch.dict(os.environ, BOPL_RL_GAME=str(native)), \
                        patch.object(reference, "run_reference", side_effect=capture), \
                        patch("builtins.print"):
                    self.assertEqual(reference.capture(args), 0)
                record = json.loads(output.with_suffix(".provenance.json").read_text())
                for key, path in reference.player_sources(selected).items():
                    self.assertEqual(record[key], hashlib.sha256(path.read_bytes()).hexdigest())
                self.assertEqual("il2cpp_metadata_sha256" in record, not explicit)

    def test_compare_rejects_changed_native_metadata_before_reading_traces(self):
        hashes = dict(replay_sha256="replay", game_assembly_sha256="code",
                      unity_player_sha256="unity", player_binary_sha256="binary")
        with patch.object(reference, "provenance", side_effect=[
                dict(hashes, il2cpp_metadata_sha256="first"),
                dict(hashes, il2cpp_metadata_sha256="second")]):
            with self.assertRaisesRegex(ValueError, "il2cpp_metadata_sha256"):
                reference.compare("left", "right")


if __name__ == "__main__":
    unittest.main()
