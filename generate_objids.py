"""Transcribe a merger tree into a flow of 'ObjectIDs', as in Hydrangea.

The idea is to identify individual 'persistent' galaxies that are represented
by unique ObjectIDs (in Hydrangea these are 'GalaxyIDs', but that word is
already used differently by the original merger trees). A second code will
then build a merge list to connect object IDs (galaxies) that merge.

Started 6 Dec 2022.

"""

import numpy as np
import h5py as h5
import os

from pdb import set_trace


# Main parameters, will eventually go into argparse structure
tree_dir = '/cosma8/data/dp004/jch/FLAMINGO/MergerTrees/ScienceRuns/L2800N5040/HYDRO_FIDUCIAL/trees_f0.1_min10_max100'

tree_file = f'{tree_dir}/vr_trees.hdf5'
num_snaps = 79

output_dir = '/cosma8/data/dp004/dc-bahe1/ProtoclusterFate/L2800N5040/HYDRO_FIDUCIAL'
if not os.path.isdir(output_dir):
    os.makedirs(output_dir)

output_file = f'{output_dir}/TracingTables.hdf5'
if os.path.isfile(output_file):
    os.rename(output_file, output_file + '.old')
aux_file = f'{output_dir}/Aux.hdf5'
if os.path.isfile(aux_file):
    os.rename(aux_file, aux_file + '.old')

# Initialise highest currently assigned ID
max_current_id = -1

# Read full merger tree info
with h5.File(tree_file, 'r') as f:
    print("Loading full merger tree info...")
    mt = {}
    print("   ... VR IDs...")
    mt['vrid'] = f['Subhalo/ID'][...]

    print("   ... Snapshots...") 
    mt['isnap'] = f['Subhalo/SnapNum'][...]

    print("   ... DescendantIDs...")
    mt['descid'] = f['MergerTree/DescendantID'][...]

    print("   ... TopLeafIDs...")
    mt['topid'] = f['MergerTree/TopLeafID'][...]
        
def main():

    # Initial set-up
    objIDs = setup_first_snap()
    
    for isnap in range(num_snaps):
        objIDs = process_snapshot(isnap, objIDs)


def setup_first_snap():

    # Set up the objIDs for each snapshot. Don't initialize all of them to arrays,
    # because that may mean memory trouble.
    objIDs = [None] * num_snaps
    
    return objIDs

def process_snapshot(isnap, objIDs):
    """Find descendants of haloes in current snapshot."""

    global max_current_id
    print(f"\n --- Processing snapshot {isnap} ---\n")
    
    with h5.File(tree_file, 'r') as f:
        mt_inds = f[f'SOAP/Snapshot{isnap:04d}'][...]
        num_haloes = mt_inds.shape[0]

    # Set up this snapshot's ObjID list if we don't have it (only really in first snap)
    if objIDs[isnap] is None:
        objIDs[isnap] = np.zeros(num_haloes, dtype=np.int32) - 1
    
    # Check for any new galaxies and assign ObjIDs to them
    ind_new = np.nonzero(objIDs[isnap] < 0)[0]
    first_new_id = max_current_id + 1                 # First ID to assign new
    max_current_id = max_current_id + len(ind_new)    # Last ID to assign new
    objIDs[isnap][ind_new] = np.arange(first_new_id, max_current_id+1, dtype=np.int32)

    # Verify that we have the correct entries (all >= 0) in the merger tree list
    if num_haloes > 0 and np.min(objIDs[isnap]) < 0:
        print(f"We have invalid ObjIDs in snapshot {isnap}...")
        set_trace()
    
    # This snapshot's ObjIDs are now complete. Write them.
    dname = f'ObjIDs/ObjIDs_{isnap:04d}'
    print(f"   <MaxID: {max_current_id}, FirstNewID: {first_new_id}>")
    with h5.File(output_file, 'a') as f:
        f[dname] = objIDs[isnap]
        f[dname].attrs['MaxObjID'] = max_current_id
        f[dname].attrs['FirstNewObjID'] = first_new_id
        f[dname].attrs['Description'] = (
            f"ObjID list. It gives, for each VR halo in snapshot {isnap}, the "
            "persistent identifier (ObjID) of this galaxy across snapshots. These "
            "identifiers are unique in each snapshot, but in general not dense (in "
            "other words, the ObjIDs typically reach higher numbers than there are "
            f"VR haloes), since some ObjIDs do not exist in snapshot {isnap}."
        )
            
    if num_haloes == 0:
        return objIDs
        
    # ------------- Find descendants ----------

    # First consistency checks, to make sure that we are picking up the right MT entries
    if np.min(mt['isnap'][mt_inds]) != isnap or np.max(mt['isnap'][mt_inds]) != isnap:
        print("Unexpected snapshot indices for current haloes in MT...")
        set_trace()
    if not np.array_equal(mt['vrid'][mt_inds], np.arange(num_haloes) + 1):
        print("Unexpected VR IDs for current haloes in MT...")

    # Retrieve the index of the descendants in the merger tree (index = ID - 1!) 
    desc_mtinds = mt['descid'][mt_inds] - 1

    # Some haloes will not have any descendants (they "fade away" completely). These
    # will have negative DescendantIDs, and so need to be filtered out manually.
    ind_no_desc = np.nonzero(desc_mtinds < 0)[0]
    desc_mtinds[ind_no_desc] = -1
    
    # ... and then convert that to the VR index and snapshot number for each
    #     (again, we want the VR *index*, which is VR ID minus 1)
    desc_vrinds = mt['vrid'][desc_mtinds] - 1
    desc_vrinds[ind_no_desc] = -1
    desc_snap = mt['isnap'][desc_mtinds]
    desc_snap[ind_no_desc] = -1
    
    # The final piece of information is which of our galaxies are main progenitors
    # of their descendants, i.e. which ones carry forward their ObjIDs. The way to do
    # that is to look for galaxies that are on the main branch of their descendant,
    # i.e. whose index is <= that of the descendant's top leaf.
    ind_is_main_prog = np.nonzero(
        (mt_inds <= mt['topid'][desc_mtinds] - 1) &
        (desc_mtinds >= 0)
    )[0]
    ind_is_merging = np.nonzero(
        (mt_inds > mt['topid'][desc_mtinds] - 1) &
        (desc_mtinds >= 0)
    )[0]
    
    # Print some summary info at this stage
    unique_desc_snaps = np.unique(desc_snap)
    print("Descendants sit in snapshots", unique_desc_snaps)
    print(f"{len(ind_is_main_prog)} out of {num_haloes} are main progenitors.")

    # ----------- Update ObjIDs ---------------------

    # Do this separately for each descendant snapshot...
    for idsnap in unique_desc_snaps:

        # Skip the placeholder '-1' snapshot for faded haloes
        if idsnap < 0:
            continue
        
        print(f"Marking main descendants (snap {isnap} --> {idsnap}...)")
        
        # Do we need to start a new ObjID list for this snapshot?
        if objIDs[idsnap] is None:
            with h5.File(tree_file, 'r') as f:
                num_haloes_dsnap = f[f'SOAP/Snapshot{idsnap:04d}'].shape[0]
            objIDs[idsnap] = np.zeros(num_haloes_dsnap, dtype=np.int32) - 1

        # Find the VR haloes in the current (progenitor) snapshot that link to
        # this (descendant) snapshot.
        subind_to_dsnap = np.nonzero(desc_snap[ind_is_main_prog] == idsnap)[0]
        ind_to_dsnap = ind_is_main_prog[subind_to_dsnap]
        if len(np.unique(desc_vrinds[ind_to_dsnap])) != len(ind_to_dsnap):
            print("Detected duplicate target VR haloes of main progenitors!")
            set_trace()

        # If we get here, things look ok. Carry forward ObjIDs.
        objIDs[idsnap][desc_vrinds[ind_to_dsnap]] = objIDs[isnap][ind_to_dsnap]
        print(f"   done [{len(ind_to_dsnap)} haloes]")

    # We now don't need the ObjIDs from isnap any more -- free up the memory
    objIDs[isnap] = None

    # Write out auxiliary information for mergelist generation
    with h5.File(aux_file, 'a') as f:
        f[f'Snap_{isnap:04d}_fading'] = ind_no_desc
        f[f'Snap_{isnap:04d}_surviving'] = ind_is_main_prog
       
        f[f'Snap_{isnap:04d}_merging'] = ind_is_merging
        f[f'Snap_{isnap:04d}_targetvr'] = desc_vrinds[ind_is_merging]
        f[f'Snap_{isnap:04d}_targetsnap'] = desc_snap[ind_is_merging]

        f[f'Snap_{isnap:04d}_numskip'] = desc_snap - isnap
    
    return objIDs



if __name__ == "__main__":
    main()
    
