"""Canonical map IDs plus read-time translation of maps-round2 run artifacts."""
import copy
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
IDENTITY = 'identity-124'
LEGACY = 'legacy-curated-54'
LEGACY_PATH = Path(__file__).with_name('legacy_map_index.json')
SETS_PATH = ROOT / 'native/data/map-sets.json'
MAP_COUNT = 124


def _json(path, default=None):
    try:
        return json.loads(Path(path).read_text())
    except (FileNotFoundError, OSError, ValueError):
        return default


def legacy_metadata():
    return _json(LEGACY_PATH, {})


def named_sets():
    data = _json(SETS_PATH, {})
    return {name: list(values) for name, values in data.items() if not name.startswith('_')}


def parse_map_ids(value):
    """IDs, inclusive ranges, ``all``, and names from native/data/map-sets.json."""
    if isinstance(value, (list, tuple)):
        result = [int(item) for item in value]
    else:
        sets = named_sets()
        result = []
        try:
            for raw in str(value).split(','):
                item = raw.strip()
                if not item:
                    raise ValueError
                if item == 'all':
                    result.extend(range(MAP_COUNT))
                elif item in sets:
                    result.extend(sets[item])
                elif '-' in item:
                    lo_text, hi_text = item.split('-', 1)
                    lo, hi = int(lo_text), int(hi_text)
                    if lo > hi:
                        raise ValueError
                    result.extend(range(lo, hi + 1))
                else:
                    result.append(int(item))
        except ValueError as error:
            raise ValueError('Expected map IDs, inclusive ranges, all, or a named map set') from error
    if not result or len(set(result)) != len(result) or any(map_id < 0 or map_id >= MAP_COUNT for map_id in result):
        raise ValueError(f'Distinct map IDs between 0 and {MAP_COUNT - 1} required')
    return result


def _records(record):
    if not isinstance(record, dict):
        return
    yield record
    for key in ('config', 'status', 'result'):
        child = record.get(key)
        if isinstance(child, dict):
            yield from _records(child)


def catalogue_from_records(*records):
    """Return the stored-index -> identity-ID table proved by artifact metadata."""
    flattened = [item for record in records for item in _records(record)]
    for record in flattened:
        name = record.get('map_catalogue')
        if name == IDENTITY:
            return tuple(range(MAP_COUNT))
        if name == LEGACY:
            return tuple(legacy_metadata()['rl_to_unity_level'])
    legacy_hashes = set(legacy_metadata().get('sim_sha256', ()))
    hashes = set()
    for record in flattened:
        runtime = record.get('runtime') or record.get('game_runtime') or {}
        if isinstance(runtime, dict):
            hashes.update(value for key, value in runtime.items() if key.endswith('sha256') and isinstance(value, str))
        for key in ('game_sha256', 'sim_sha256', 'player_binary_sha256'):
            if isinstance(record.get(key), str):
                hashes.add(record[key])
        game = record.get('game')
        if isinstance(game, dict) and isinstance(game.get('sha256'), str):
            hashes.add(game['sha256'])
    if hashes & legacy_hashes:
        return tuple(legacy_metadata()['rl_to_unity_level'])
    return tuple(range(MAP_COUNT))


def run_records(run_dir):
    run_dir = Path(run_dir)
    return tuple(filter(None, (_json(run_dir / name) for name in ('config.json', 'status.json', 'result.json'))))


def run_catalogue(run_dir):
    return catalogue_from_records(*run_records(run_dir))


def to_identity(run_dir, index):
    return translate_index(run_catalogue(run_dir), index)


def translate_index(table, index):
    index = int(index)
    if index < 0 or index >= len(table):
        raise ValueError(f'Map index {index} is outside catalogue width {len(table)}')
    return int(table[index])


def translate_ids(values, *records):
    table = catalogue_from_records(*records)
    return [translate_index(table, value) for value in values]


def translate_map_matches(values, *records):
    table = catalogue_from_records(*records)
    translated = {}
    for key, value in (values or {}).items():
        identity = str(translate_index(table, key))
        translated[identity] = translated.get(identity, 0) + value
    return translated


def normalize_record(payload, *catalogue_records):
    """Deep-copy one run/report record with every stored map ID made canonical."""
    if not isinstance(payload, dict):
        return payload
    result = copy.deepcopy(payload)
    records = (*catalogue_records, payload)
    table = catalogue_from_records(*records)
    for key in ('training_maps', 'evaluation_maps', 'heldout_maps', 'maps'):
        if isinstance(result.get(key), list):
            result[key] = [translate_index(table, value) for value in result[key]]
    if isinstance(result.get('map_matches'), dict):
        result['map_matches'] = translate_map_matches(result['map_matches'], *records)
    for collection in ('groups', 'games', 'tasks'):
        rows = result.get(collection)
        if not isinstance(rows, list):
            continue
        for row in rows:
            if isinstance(row, dict) and 'map' in row:
                row['map'] = translate_index(table, row['map'])
    if isinstance(result.get('config'), dict):
        result['config'] = normalize_record(result['config'], *records)
    result['map_catalogue'] = IDENTITY
    return result


def translate_keyed_maps(values, *records):
    """Translate a JSON object keyed by map ID, merging only if aliases collide."""
    table = catalogue_from_records(*records)
    translated = {}
    for key, value in (values or {}).items():
        identity = str(translate_index(table, key))
        if identity in translated:
            raise ValueError(f'Multiple stored map IDs translate to identity map {identity}')
        translated[identity] = value
    return translated
