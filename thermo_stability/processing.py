from mp_api.client import MPRester
from pymatgen.core.structure import Structure # to load data structure from json file
from pymatgen.analysis.graphs import StructureGraph
from pymatgen.core import Composition # to extract atomic fraction
from pymatgen.analysis.local_env import MinimumDistanceNN  # faster way to get bond structure w.r.t VoronoiNN, CrystalNN

import numpy as np
import os,sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from thermo_stability import utils, config

logpath = config.LOG_DIR
print(logpath)
logger = utils.setup_logging(log_path=logpath + "/processin.txt")

logger.info("Process material")
def process_material(args):
    oxidation = 0
    mid, structure = args
    if structure is None or structure.num_sites > 100:
        print(f"Warning: No structure for {mid}, skipping...")
        return None
    try:
        if not structure.is_ordered:
            structure = structure.get_ordered_structure()
        # Cache oxidation states if possible
        if not structure.site_properties.get("oxidation_states"):
            logger.info('oxidation not available')
            oxidation += 1
            structure.add_oxidation_state_by_guess()
        sg = StructureGraph.from_local_env_strategy(structure, MinimumDistanceNN(cutoff=5))  # cutoff x d_min
        bond_lengths = []
        bond_types = set()
        
        for i, j, attr in sg.graph.edges(data=True):
            site_i = structure[i]
            site_j = structure[j]
            bond_types.add("-".join(sorted([site_i.specie.symbol, site_j.specie.symbol])))
            bond_lengths.append(attr.get("weight", site_i.distance(site_j)))  # Use precomputed weight if available

        if not bond_lengths:
            return None
        
        mean_bond_length = np.mean(bond_lengths)
        std_bond_length = np.std(bond_lengths)
        return {
            "material_id": mid,
            "num_bonds": len(bond_lengths),
            "mean_bond_length": mean_bond_length,
            "std_bond_length": std_bond_length,
            "unique_bond_types": len(bond_types)
        }
    
    except Exception as e:
        logger.info(f"Error processing {mid}: {e}")
        return None
    logger.info('structures without oxidation data = ', oxidation)

def get_atomic_fractions(comp, elements_list):
    logger.info("Get atomic fractions")
    fractions = dict.fromkeys(elements_list, 0.0)
    total = sum(comp.get_el_amt_dict().values())
    for el, amt in comp.get_el_amt_dict().items():
        if el in fractions:
            fractions[el] = amt / total
    return fractions

def safe_process_material(args):
    try:
        return process_material(args)
    except Exception as e:
        logger.info(f"Failed to process {args[0]}: {e}")
        return None

def get_atomic_fractions(comp, elements_list):
    fractions = dict.fromkeys(elements_list, 0.0)
    total = sum(comp.get_el_amt_dict().values())
    for el, amt in comp.get_el_amt_dict().items():
        if el in fractions:
            fractions[el] = amt / total
    return fractions
    
def atomic_fraction_row(row):
    comp = Composition(row['composition'])
    fractions = get_atomic_fractions(comp, ALL_ELEMENTS)
    fractions['material_id'] = row['material_id']
    return fractions

def classify_stability(energy_above_hull, threshold=0.05):
    """
    Classify materials into 0 (stable) and 1 (unstable).
    
    Args:
        energy_above_hull (array-like): Energy above hull values (eV/atom).
        threshold (float): Stability threshold (eV/atom). Default is 0.05.
        
    Returns:
        np.ndarray: Array of 0 (stable) and 1 (unstable).
    """
    energy_above_hull = np.array(energy_above_hull)
    labels = np.where(energy_above_hull <= threshold, 0, 1)
    return labels

