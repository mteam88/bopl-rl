"""Canonical map-set parsing and maps-round2 read-time translation."""
from pathlib import Path
import tomllib
import unittest

import map_catalogue
import map_set


ROOT = Path(__file__).resolve().parents[3]
CONFIGS = ROOT / 'native/rl/sweep/configs'


class MapCatalogueTests(unittest.TestCase):
    def test_named_sets_ranges_and_explicit_additions(self):
        sets = map_catalogue.named_sets()
        self.assertEqual(map_catalogue.parse_map_ids('legacy31'), sets['legacy31'])
        self.assertEqual(map_catalogue.parse_map_ids('legacy31-heldout'), sets['legacy31-heldout'])
        self.assertEqual(map_catalogue.parse_map_ids('legacy31,123'), sets['legacy31'] + [123])
        self.assertEqual(map_catalogue.parse_map_ids('120-123'), [120, 121, 122, 123])
        self.assertEqual(map_catalogue.parse_map_ids('all-124'), list(range(124)))

    def test_legacy_hash_and_explicit_catalogue_translate(self):
        legacy = map_catalogue.legacy_metadata()
        record = {'runtime': {'player_binary_sha256': legacy['sim_sha256'][0]}}
        self.assertEqual(map_catalogue.catalogue_from_records(record)[7], 8)
        self.assertEqual(map_catalogue.translate_ids([0, 7, 53], record), [0, 8, 92])
        self.assertEqual(map_catalogue.catalogue_from_records(
            {'map_catalogue': map_catalogue.IDENTITY, **record})[7], 7)

    def test_record_normalization_translates_every_shared_shape(self):
        record = {'map_catalogue': map_catalogue.LEGACY}
        payload = {
            'training_maps': [7, 53], 'map_matches': {'7': 2, '53': 1},
            'groups': [{'map': 7}], 'games': [{'map': 53}], 'tasks': [{'map': 31}],
            'config': {'heldout_maps': [8]},
        }
        value = map_catalogue.normalize_record(payload, record)
        self.assertEqual(value['training_maps'], [8, 92])
        self.assertEqual(value['map_matches'], {'8': 2, '92': 1})
        self.assertEqual([value[key][0]['map'] for key in ('groups', 'games', 'tasks')], [8, 92, 7])
        self.assertEqual(value['config']['heldout_maps'], [10])
        self.assertEqual(value['map_catalogue'], map_catalogue.IDENTITY)

    def test_historical_tomls_translate_to_named_sets_without_edits(self):
        sets = map_catalogue.named_sets()
        legacy = {'map_catalogue': map_catalogue.LEGACY}
        cases = (
            ('v4-aim-maps.toml', 'legacy31', 'legacy31-heldout'),
            ('v4-aim-maps-r2.toml', 'legacy54', 'legacy54-heldout'),
        )
        for filename, training, heldout in cases:
            with self.subTest(filename=filename):
                with (CONFIGS / filename).open('rb') as handle:
                    fixed = tomllib.load(handle)['fixed']
                self.assertEqual(
                    map_catalogue.translate_ids(fixed['training_maps'], legacy), sets[training])
                self.assertEqual(
                    map_catalogue.translate_ids(fixed['heldout_maps'], legacy), sets[heldout])

    def test_map_set_and_mac_translation_are_identity_for_all_124(self):
        table = map_set.load()
        self.assertEqual(map_set.levels(table), dict(enumerate(range(124))))
        self.assertEqual(map_set.parse_maps('all', table), list(range(124)))
        self.assertEqual(map_set.parse_maps('legacy31', table), map_catalogue.named_sets()['legacy31'])


if __name__ == '__main__':
    unittest.main()
