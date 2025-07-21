from joblib import Parallel, delayed # to avoid bad structure delay
from multiprocessing import cpu_count  # multi-processing
from datetime import datetime
import time
import pandas as pd
import psutil
import sys, os, json, gc
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from mp_api.client import MPRester
from pymatgen.core import Composition # to extract atomic fraction
from pymatgen.core.structure import Structure
from pymatgen.analysis.graphs import StructureGraph
from pymatgen.analysis.local_env import MinimumDistanceNN  # faster way to get bond structure w.r.t VoronoiNN, CrystalNN

# Costum modules
from thermo_stability import utils, config, processing, creds

scripter = utils.Scripter() # decorator
filepath      = config.FILE_DIR
logpath       = config.LOG_DIR
bondstatspath = config.BONDSTATS_DIR

logger = utils.setup_logging(name='features', log_path=logpath + "/feature.txt")

@scripter
def data_acquisition():
    fields = [
        "material_id",      # to match bond_struct, atomic_fraction, other features 
        "formation_energy_per_atom",
        "band_gap",
        "energy_per_atom",
        "total_magnetization",
        "volume",
        "density",
        "energy_above_hull",
        "is_stable",
        "nelements",
        "nsites",
        "vbm",
        "cbm",
        "composition",      # needed for atomic fractions
        "structure",        # material structure
        "formula_pretty",   # filter seen elements
    ]
    logger.info(f"start getting material properties:  {datetime.now().time().strftime('%H:%M:%S')}")
    # Basic material properties
    logger.info("Querying material summaries...")
    api_key = creds.api_key
    mpr = MPRester(api_key)
    process = psutil.Process(os.getpid())
    start_cpu_times = process.cpu_times()
    docs = mpr.materials.summary.search(
        fields=fields,
        formation_energy=(-20, 5),
    )
    docs_list, all_elements, seen_formula, structures_dict = [], set(), set(), {}
    for doc in docs:
        if (doc.structure is None or pd.isna(doc.cbm) or pd.isna(doc.vbm) or doc.nsites > 100 or doc.volume > 2000): 
            continue # skip entries with bad/missing data
        formula = getattr(doc, 'formula_pretty', None) # remove duplicate formulas + save structure_dict
        if formula in seen_formula: continue
        seen_formula.add(formula)
        structures_dict[doc.material_id] = doc.structure.as_dict()

        doc_data = {field: getattr(doc, field, None) for field in fields}
        docs_list.append(doc_data)    
        # Collect all elements composition
        comp = Composition(doc.composition)
        all_elements.update(comp.get_el_amt_dict().keys())
        
    all_elements = sorted(list(all_elements))
    logger.info(f"done getting material properties:  {datetime.now().time().strftime('%H:%M:%S')}")

    # JSON serialization of Structure objects
    with open(filepath + '/structures_dict.json', 'w') as f:
        json.dump(structures_dict, f)

    logger.info('DONE WITH STRUCTURE_DICT!!!')
    df = pd.DataFrame(docs_list) # to pandas df
    logger.info(f'DF columns: {df.columns}')
    df.drop_duplicates('formula_pretty', keep='first', inplace=True) # filter duplicate polymorph per composition
    logger.info(f"structure dict done:  {datetime.now().time().strftime('%H:%M:%S')}")

    # Additional features
    df['vpa'] = df['volume'] / df['nsites']
    df['magmom_pa'] = df['total_magnetization'] / df['nsites']
    df['all_elements'] = pd.DataFrame(all_elements)

    # save df
    df.to_csv(filepath + '/df.csv',index=False)
    logger.info(f"finished adding more features at:  {datetime.now().time().strftime('%H:%M:%S')}")
    end_cpu_times = process.cpu_times()

    # Calculate CPU time (user + system)
    cpu_time_used = (end_cpu_times.user - start_cpu_times.user) + (end_cpu_times.system - start_cpu_times.system)
    logger.info(f"\nCPU time used (main process): {cpu_time_used:.2f} seconds")

    
@scripter
def atomic_fraction():
    df = pd.read_csv(filepath + '/df.csv', low_memory=False)
    elements_list = df['all_elements']  # Use df['all_elements'].iloc[0] if it’s a list inside a column
    records = df.to_dict("records")

    fraction_records = []
    process = psutil.Process(os.getpid())
    start_cpu_times = process.cpu_times()

    for record in records:
        fraction = processing.atomic_fraction_row(record, elements_list)
        fraction_records.append(fraction)

    logger.info(f"Atomic fractions computed at: {datetime.now().strftime('%H:%M:%S')}")

    df_fractions = pd.DataFrame(fraction_records)
    df_fractions.to_csv(filepath + '/df_fractions.csv', index=False)

    end_cpu_times = process.cpu_times()
    cpu_time_used = (end_cpu_times.user - start_cpu_times.user) + (end_cpu_times.system - start_cpu_times.system)
    logger.info(f"CPU time used (main process): {cpu_time_used:.2f} seconds")


@scripter
def bond_structure():
    start = int(sys.argv[1])
    end = int(sys.argv[2])

    logger.info(f'Processing batch from {start} to {end}')
    with open(filepath + '/structures_dict.json', 'r') as f:
        structures_dict_raw = json.load(f)

    material_ids = list(structures_dict_raw.keys())[start:end]
    args = [
        (mid, Structure.from_dict(structures_dict_raw[mid]))
        for mid in material_ids
    ]

    process = psutil.Process(os.getpid())
    start_cpu_times = process.cpu_times()

    structures_dict = {
        mid: Structure.from_dict(data)
        for mid, data in structures_dict_raw.items()
    }

    logger.info("Calculating bond statistics with multiprocessing and joblib...")
    n_jobs = min(4, cpu_count())
    logger.info(f'Number of jobs: {n_jobs}')  

    results = Parallel(n_jobs=n_jobs, verbose=10, batch_size="auto")(
        delayed(processing.safe_process_material)(arg) for arg in args
    )
    results = [r for r in results if r is not None]
    df_bond_stats = pd.DataFrame(results)
    os.makedirs(filepath + '/bond_stats', exist_ok=True)
    df_bond_stats.to_csv(filepath + '/bond_stats' + f'/df_bond_stats_from{start}to{end}.csv',index=False)
    end_cpu_times = process.cpu_times()
    # Calculate CPU time (user + system)
    cpu_time_used = (end_cpu_times.user - start_cpu_times.user) + (end_cpu_times.system - start_cpu_times.system)
    logger.info(f"\nCPU time used (main process): {cpu_time_used:.2f} seconds  with {n_jobs:.1f} cpu_counts")

    # clear memory
    del structures_dict_raw, args, results, df_bond_stats
    gc.collect()
    time.sleep(2)


@scripter
def merge_df_file():
    df_bond_struct = pd.read_csv(filepath + '/df_bond_stats.csv')
    df_atomic_frac = pd.read_csv(filepath + '/df_fractions.csv')
    df_features    = pd.read_csv(filepath + '/df.csv')
    df_combined = pd.merge(df_features, df_atomic_frac, on='material_id', how='inner')
    df_combined = pd.merge(df_combined, df_bond_struct, on='material_id', how='inner')
    df_combined.to_csv(filepath + '/df_combined.csv', index=False)


@scripter
def merge_df_structure():    
    full_paths = [os.path.join(bondstatspath, f) for f in os.listdir(bondstatspath)]
    # remove 1st index column
    df_list = []
    for path in full_paths:
        df = pd.read_csv(path)
        df.drop(columns='Unnamed: 0', inplace=True, errors='ignore')  # Drop early
        df_list.append(df)
    df_merged_struct = pd.concat(df_list, ignore_index=True)
    df_merged_struct.to_csv(filepath + '/df_bond_stats.csv', index=False)


if __name__ == '__main__':
    scripter.run()
