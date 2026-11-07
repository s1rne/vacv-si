"""Offline, conservative record-level SI pairs; never aggregate endpoint values."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import re

import numpy as np
import pandas as pd

from vacv import classify
from vacv.curate import standardize
from vacv.dataset import load
from vacv.sources.chembl import is_biochemical
from vacv.units import to_micromolar


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def interval(ic, cc):
    """Return bounds only when ratio direction is identified by the two records."""
    if ic['value_um'] <= 0 or cc['value_um'] <= 0:
        return None
    ratio = cc['value_um'] / ic['value_um']
    a, b = ic['relation'], cc['relation']
    if a == b == '=':
        return ratio, ratio, 'exact'
    if a == '=' and b in ('>', '>=') or a in ('<', '<=') and b == '=':
        return ratio, np.inf, 'lower_bound'
    if a in ('>', '>=') and b == '=' or a == '=' and b in ('<', '<='):
        return 0., ratio, 'upper_bound'
    return None


def singleton(records):
    """Exact duplicate activity records may collapse; different assays may not."""
    groups = defaultdict(list)
    for row in records:
        key = (row['assay_id'], row['endpoint'], row['value_um'], row['relation'])
        groups[key].append(row)
    if len(groups) != 1:
        return None
    rows = next(iter(groups.values()))
    result = dict(rows[0])
    result['activity_ids'] = sorted({str(r['activity_id']) for r in rows})
    return result


def times(description):
    found = re.findall(r'(?<![\w.])(\d+(?:\.\d+)?)\s*(hours?|hrs?|h|days?|d)\b', description.lower())
    return sorted({float(n) * (24 if unit.startswith('d') else 1) for n, unit in found})


def cache_activity(activities, record, cache_file):
    """Conflicting snapshots stay quarantined even if the original repeats later."""
    aid = str(record['activity_id'])
    if aid in activities:
        if activities[aid]['record'] != record:
            activities[aid]['conflict'] = True
    else:
        activities[aid] = {'record': record, 'cache_file': cache_file, 'conflict': False}


def build(source, data_dir, out):
    out.mkdir(parents=True, exist_ok=False)
    source = Path(source)
    source_tables = [source/'data/processed/external_v2'/n for n in ('measured_clean.csv', 'censored.csv')]
    wanted = set()
    for path in source_tables:
        for value in pd.read_csv(path, usecols=['source_id']).source_id.dropna():
            wanted.update(re.findall(r'CHEMBL\d+', str(value)))
    # Read cache only. There is deliberately no HTTP client or network fallback.
    assays, cells, molecules, activities = {}, {}, {}, {}
    hashes = {}
    cache = source/'data/raw/chembl'
    for path in sorted(cache.glob('*.json')):
        try:
            payload = json.loads(path.read_text())
        except (ValueError, UnicodeError):
            continue
        if not isinstance(payload, dict):
            continue
        used = False
        for record in payload.get('molecules', []):
            smi = (record.get('molecule_structures') or {}).get('canonical_smiles')
            if smi:
                molecules[record['molecule_chembl_id']] = smi
                used = True
        for record in payload.get('cell_lines', []):
            cells[record['cell_chembl_id']] = record
            used = True
        for record in payload.get('assays', []):
            if record.get('document_chembl_id') in wanted:
                assays[record['assay_chembl_id']] = record
                used = True
        for record in payload.get('activities', []):
            if record.get('document_chembl_id') in wanted and record.get('activity_id'):
                cache_activity(activities, record, path.name)
                used = True
        if used:
            hashes[path.name] = digest(path)
    data = load(data_dir, require_v2=True)
    competition_smiles = set(pd.concat([data.train.SMILES, data.test.SMILES]).astype(str))
    blocked, invalid_competition = set(), []
    for smi in competition_smiles:
        std = standardize(smi)
        if std.inchikey:
            blocked.add(std.inchikey[:14])
        else:
            invalid_competition.append(smi)
    standardized, records, rejected = {}, [], Counter()
    for aid, cached in activities.items():
        a = cached['record']
        endpoint = str(a.get('standard_type') or '').upper()
        if endpoint not in ('CC50', 'IC50', 'EC50'):
            rejected['other_endpoint'] += 1
            continue
        if cached['conflict']:
            rejected['conflicting_cache_activity'] += 1
            continue
        assay = assays.get(a.get('assay_chembl_id'), {})
        description = a.get('assay_description') or assay.get('description') or ''
        organism = a.get('target_organism') or assay.get('assay_organism') or ''
        if is_biochemical(a.get('bao_label') or assay.get('bao_format') or ''):
            rejected['biochemical'] += 1
            continue
        if endpoint != 'CC50':
            if not classify.mentions_vaccinia(organism, description):
                rejected['non_vaccinia'] += 1
                continue
            if re.search(r'cowpox|ectromelia|monkeypox|variola|CPXV|ECTV|MPXV', description, re.I):
                rejected['multiple_virus_description'] += 1
                continue
        mid = a.get('molecule_chembl_id')
        smi = molecules.get(mid)
        if not smi:
            rejected['missing_smiles'] += 1
            continue
        if smi not in standardized:
            standardized[smi] = standardize(smi)
        std = standardized[smi]
        if std.reject or not std.inchikey:
            rejected['invalid_external_structure'] += 1
            continue
        if smi in competition_smiles or std.inchikey[:14] in blocked:
            rejected['competition_structure'] += 1
            continue
        cell_record = cells.get(assay.get('cell_chembl_id'), {})
        cell = classify.extract_cell_line(cell_record.get('cell_name'), cell_record.get('cell_description'),
                                          assay.get('assay_cell_type'), a.get('target_pref_name'), description)
        if not cell:
            rejected['unknown_cell'] += 1
            continue
        relation = a.get('standard_relation')
        if relation not in ('=', '>', '>=', '<', '<='):
            rejected['unsupported_relation'] += 1
            continue
        value, reason = to_micromolar(a.get('standard_value'), a.get('standard_units'), mol_weight=std.mol_weight)
        if value is None or not np.isfinite(value) or value <= 0:
            rejected['invalid_value_or_units'] += 1
            continue
        records.append({'activity_id': aid, 'assay_id': a.get('assay_chembl_id'),
            'document_id': a.get('document_chembl_id'), 'molecule_id': mid,
            'SMILES': std.smiles, 'inchikey14': std.inchikey[:14], 'cell': cell,
            'endpoint': endpoint, 'side': 'cc' if endpoint == 'CC50' else 'ic',
            'raw_value': a.get('value'), 'raw_units': a.get('units'), 'raw_relation': a.get('relation'),
            'standard_value': a.get('standard_value'), 'standard_units': a.get('standard_units'),
            'relation': relation, 'value_um': float(value), 'organism': organism,
            'strain': classify.extract_strain(assay.get('assay_strain'), organism, description),
            'time_hours': times(description), 'description': description, 'cache_file': cached['cache_file']})
    groups = defaultdict(lambda: defaultdict(list))
    for r in records:
        groups[(r['document_id'], r['inchikey14'], r['cell'])][r['side']].append(r)
    candidates, quarantine = [], []
    for key, sides in groups.items():
        ic, cc = singleton(sides['ic']), singleton(sides['cc'])
        reason = None
        if not ic or not cc:
            reason = 'missing_or_ambiguous_endpoint_pair'
        elif ic['time_hours'] and cc['time_hours'] and ic['time_hours'] != cc['time_hours']:
            reason = 'different_exposure_times'
        elif interval(ic, cc) is None:
            reason = 'unidentified_ratio_interval'
        if reason:
            quarantine.append({'document_id': key[0], 'inchikey14': key[1], 'cell': key[2], 'reason': reason,
                               'ic_records': len(sides['ic']), 'cc_records': len(sides['cc'])})
            continue
        lower, upper, kind = interval(ic, cc)
        candidates.append({'SMILES': ic['SMILES'], 'inchikey14': key[1], 'lower_si': lower,
            'upper_si': upper, 'kind': kind, 'source_id': key[0]+':'+key[1], 'cell': key[2],
            'strain': ic['strain'], 'ic_activity_ids': '|'.join(ic['activity_ids']),
            'cc_activity_ids': '|'.join(cc['activity_ids']), 'ic_assay_id': ic['assay_id'],
            'cc_assay_id': cc['assay_id'], 'ic_endpoint': ic['endpoint'], 'cc_endpoint': cc['endpoint'],
            'ic_relation': ic['relation'], 'cc_relation': cc['relation'],
            'ic_value_um': ic['value_um'], 'cc_value_um': cc['value_um'],
            'ic_time_hours': json.dumps(ic['time_hours']), 'cc_time_hours': json.dumps(cc['time_hours']),
            'exposure_verified': bool(ic['time_hours'] and cc['time_hours']),
            'provenance_limit': 'same paper and known cell; full assay protocol equivalence not independently verified'})
    key_counts = Counter(r['inchikey14'] for r in candidates)
    eligible = []
    for row in candidates:
        if key_counts[row['inchikey14']] != 1:
            quarantine.append({'source_id': row['source_id'], 'inchikey14': row['inchikey14'], 'reason': 'multiple_pair_contexts'})
        else:
            eligible.append(row)
    cols = ['SMILES','lower_si','upper_si','kind','source_id']
    frame = pd.DataFrame(eligible) if eligible else pd.DataFrame(columns=cols)
    frame.to_csv(out/'eligible.csv', index=False)
    pd.DataFrame(records).to_json(out/'source-records.json', orient='records', indent=2)
    pd.DataFrame(quarantine).to_csv(out/'quarantine.csv', index=False)
    manifest = {'source': str(source), 'network_used': False, 'wanted_document_ids': sorted(wanted),
        'cache_files_sha256': hashes, 'source_tables_sha256': {str(p): digest(p) for p in source_tables},
        'script_sha256': digest(Path(__file__)), 'competition_keys': len(blocked),
        'invalid_competition_smiles': invalid_competition, 'source_activities': len(activities),
        'retained_records': len(records), 'eligible_rows': len(eligible),
        'kind_counts': dict(Counter(r['kind'] for r in eligible)), 'record_rejections': dict(rejected),
        'quarantined_groups': len(quarantine), 'overlap': len({r['inchikey14'] for r in eligible} & blocked),
        'eligible_sha256': digest(out/'eligible.csv'),
        'limitation': 'record-unambiguous pairs, not independent full assay-protocol verification; missing times remain flagged'}
    (out/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    print({k:manifest[k] for k in ('source_activities','retained_records','eligible_rows','kind_counts','overlap')})


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--source', type=Path, default=Path('/Users/mike/Projects/vacv-si'))
    p.add_argument('--data-dir', type=Path, default=Path('data/raw_v2'))
    p.add_argument('--out', type=Path, default=Path('experiments/results/v3-source-pairs'))
    args = p.parse_args()
    build(args.source, args.data_dir, args.out)
