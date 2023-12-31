"""Transcribe merging galaxy flows into a "merge list", like in Hydrangea.

This builds on the `ObjIDs.hdf5` files generated by `generate_objids.py`.
The idea here is to build a series of lists (arrays, really) that point for
each galaxy (ObjID) to its "carrier" galaxy in a particular snapshot.

Started 12 Dec 2022.

"""

import numpy as np
import h5py as h5
import os

from pdb import set_trace

# Main parameters, will eventually go into argparse structure

num_snaps = 79

output_dir = '/cosma8/data/dp004/dc-bahe1/ProtoclusterFate/L2800N5040/HYDRO_FIDUCIAL'
if not os.path.isdir(output_dir):
    os.makedirs(output_dir)

output_file = f'{output_dir}/TracingTables.hdf5'

aux_file = f'{output_dir}/Aux.hdf5'
obj_file = output_file

num_objs_total = None

def main():

    # Initial set-up
    print("Setting up carrier and ObjID lists...")
    carrierIDs, objIDs = setup_first_snap()
    
    for isnap in range(num_snaps):
        carrierIDs, objIDs = process_snapshot(isnap, carrierIDs, objIDs)


def setup_first_snap():

    # Set up the carrierIDs for each snapshot. Don't initialize all of them to arrays,
    # because that may mean memory trouble.
    carrierIDs = [None] * num_snaps
    objIDs = [None] * num_snaps

    global num_objs_total
    with h5.File(obj_file, 'r') as f:
        num_objs_total = f[f'ObjIDs/ObjIDs_{num_snaps-1:04d}'].attrs['MaxObjID'] + 1
    print(f"There are {num_objs_total} ObjIDs in total.")
        
    return carrierIDs, objIDs


def process_snapshot(isnap, carrierIDs, objIDs):
    """Append merger list..."""

    print(f"\n --- Processing snapshot {isnap} ---\n")

    # Make sure we have this snapshot's objIDs loaded
    objIDs = load_objids(isnap, objIDs)
    num_haloes = len(objIDs[isnap])
    if num_haloes == 0:
        return carrierIDs, objIDs

    aux = load_aux_data(isnap)
    
    # Set up carrierIDs for all required snaps (including this, if not yet done)
    max_n_skip = np.max(aux['vr_numskip'])
    max_target_snap = isnap + max_n_skip
    for itsnap in range(isnap, max_target_snap + 1):
        carrierIDs = setup_carrier_ids(itsnap, carrierIDs)
        objIDs = load_objids(itsnap, objIDs)

    print(f"   [{num_haloes} VRs, {len(carrierIDs[isnap])} Objs]")
        
    # Safety check: there should be no -1's in this snapshot's list anymore
    if np.count_nonzero(carrierIDs[isnap] == -1) > 0:
        print("Unexpected entries in carrier ID list before propagation!")
        set_trace()

    # Write this snapshot's carrierIDs
    write_carrier_ids(isnap, carrierIDs)

    if isnap == num_snaps - 1:
        return carrierIDs, objIDs
        
    # ---------- Fill in the carrierIDs... --------------

    # Already faded ones --> simple, keep faded

    ind_faded = np.nonzero(carrierIDs[isnap] == -10)[0]
    carrierIDs[isnap + 1][ind_faded] = -10

    obj_status = np.zeros_like(carrierIDs[isnap], dtype=np.int32) - 1
    
    # Haloes that are fading now --> set all associated CIDs to -10
    objID_fading = objIDs[isnap][aux['vr_fading']]
    obj_status[objID_fading] = -10
    cid_fading = np.nonzero(obj_status[carrierIDs[isnap]] == -10)[0]
    print(f"   {len(objID_fading)} fading objIDs, carrying {len(cid_fading)}.")
    carrierIDs[isnap + 1][cid_fading] = -10

    # Haloes that are simply surviving to the next snap
    subind_simple = np.nonzero(aux['vr_numskip'][aux['vr_surviving']] == 1)[0]
    ind_simple = aux['vr_surviving'][subind_simple]
    obj_status[objIDs[isnap][ind_simple]] = 1
    cid_simple = np.nonzero(
        (obj_status[carrierIDs[isnap]] == 1) &
        (carrierIDs[isnap] >= 0)
    )[0]
    carrierIDs[isnap + 1][cid_simple] = carrierIDs[isnap][cid_simple]

    # Haloes that skip one or more snapshots (incl. ones that then merge)
    ind_skippers = np.nonzero(aux['vr_numskip'] > 1)[0]
    print(f"   {len(ind_skippers)} skipping objs...")
    
    for iskip in range(2, max_n_skip + 1):
        ind_iskip = np.nonzero(aux['vr_numskip'] == iskip)[0]
        print(f"      [{len(ind_iskip)} skip {iskip}]")
        if len(ind_iskip) == 0:
            continue

        if np.count_nonzero(obj_status[objIDs[isnap][ind_iskip]] == iskip) > 0:
            set_trace()
        obj_status[objIDs[isnap][ind_iskip]] = iskip
        ind_carried = np.nonzero(
            (obj_status[carrierIDs[isnap]] == iskip) & (carrierIDs[isnap] >= 0))[0]
        for itsnap in range(isnap+1, isnap+iskip+1):
            carrierIDs[itsnap][ind_carried] = carrierIDs[isnap][ind_carried]        
            
    #for ivr in ind_skippers:
    #    nskip = aux['vr_numskip'][ivr]
    #    if nskip < 2:
    #        set_trace()
    #    obj = objIDs[isnap][ivr]
    #    ind_carried = np.nonzero(carrierIDs[isnap] == obj)[0]
    #    if len(ind_carried) == 0:
    #        set_trace()
    #    for itsnap in range(isnap+1, isnap+nskip+1):
    #        carrierIDs[itsnap][ind_carried] = carrierIDs[isnap][ind_carried]

    # Haloes that are merging
    print(f"   {len(aux['vr_merging'])} merging objs...")

    #for iivr, ivr in enumerate(aux['vr_merging']):
    #    obj = objIDs[isnap][ivr]
    #    ind_carried = np.nonzero(carrierIDs[isnap] == obj)[0]
    #    if len(ind_carried) == 0:
    #        set_trace()
    #    if len(ind_carried) > 1:
    #        print(f"Obj {obj} carries other obj!")
    #    itsnap = aux['merge_targets_snap'][iivr]
    #    targ_obj = objIDs[itsnap][aux['merge_targets_vr'][iivr]]
    #    
    #    carrierIDs[itsnap][ind_carried] = targ_obj

    if len(aux['merge_targets_snap']) > 0:
        for itsnap in range(isnap+1, np.max(aux['merge_targets_snap']) + 1):
            subind_tsnap = np.nonzero(aux['merge_targets_snap'] == itsnap)[0]
            print(f"      [{len(subind_tsnap)} merge to snap {itsnap}]")
            if len(subind_tsnap) == 0:
                continue
            vr_tsnap = aux['vr_merging'][subind_tsnap]
            obj_status[:] = -1
            obj_status[objIDs[isnap][vr_tsnap]] = (
                objIDs[itsnap][aux['merge_targets_vr'][subind_tsnap]])
            ind_carried = np.nonzero(
                (obj_status[carrierIDs[isnap]] >= 0) & (carrierIDs[isnap] >= 0))[0]
            carrierIDs[itsnap][ind_carried] = obj_status[carrierIDs[isnap][ind_carried]]

    # For convenience, also create and write a reverse objectID list
    rev_list = np.zeros(num_objs_total, dtype=np.int32) - 1
    rev_list[objIDs[isnap]] = np.arange(num_haloes)
    rdname = f'VRIndices/VRIndices_{isnap:04d}'
    with h5.File(output_file, 'a') as f:
        if rdname in f.keys():
            del f[rdname]

        dset = f.create_dataset(
            rdname, (num_objs_total,), compression='lzf', chunks=(100000),
            dtype=rev_list.dtype
        )
        dset.write_direct(rev_list)
        dset.attrs['NumHaloes'] = num_haloes
        dset.attrs['MaxObjID'] = np.max(objIDs[isnap])
        dset.attrs['NumObjIDsTotal'] = num_objs_total
        dset.attrs['Description'] = (
                'Reverse ObjID list. It gives, for each ObjID that has been assigned '
                f'up to snapshot {isnap}, the VR halo index corresponding to it in '
                f'snapshot {isnap}. Galaxies that are not identified in snapshot '
                f'{isnap} are assigned a negative number as a placeholder.'
        )
            
    # Free arrays for this snapshot to save memory
    objIDs[isnap] = None
    carrierIDs[isnap] = None
        
    return carrierIDs, objIDs

    
def load_objids(isnap, objIDs):
    """Load the ObjIDs for a given snapshot into the full list."""
    if objIDs[isnap] is None:
        with h5.File(obj_file, 'r') as f:
            objIDs[isnap] = f[f'ObjIDs/ObjIDs_{isnap:04d}'][...]
    return objIDs


def load_aux_data(isnap):
    """Load the auxiliary data about merging and fading galaxies."""
    aux = {}
    with h5.File(aux_file, 'r') as f:
        aux['vr_merging'] = f[f'Snap_{isnap:04d}_merging'][...]
        aux['vr_surviving'] = f[f'Snap_{isnap:04d}_surviving'][...]
        aux['vr_fading'] = f[f'Snap_{isnap:04d}_fading'][...]
        aux['merge_targets_vr'] = f[f'Snap_{isnap:04d}_targetvr'][...]
        aux['merge_targets_snap'] = f[f'Snap_{isnap:04d}_targetsnap'][...]
        aux['vr_numskip'] = f[f'Snap_{isnap:04d}_numskip'][...]
    return aux
        

def setup_carrier_ids(isnap, carrierIDs):
    """Set up the carrier ID list for a specified snapshot."""

    # If the list already exists, do nothing
    if carrierIDs[isnap] is not None:
        return carrierIDs

    with h5.File(obj_file, 'r') as f:
        max_objid = f[f'ObjIDs/ObjIDs_{isnap:04d}'].attrs['MaxObjID']
        first_new_objid = f[f'ObjIDs/ObjIDs_{isnap:04d}'].attrs['FirstNewObjID']
        
    carrierIDs[isnap] = np.zeros(max_objid+1, dtype=np.int32) - 1

    # Immediately initialize new galaxies to their own objIDs
    # (they cannot have done anything yet, so carrierID must be equal to objID)
    carrierIDs[isnap][first_new_objid:] = np.arange(first_new_objid, max_objid+1)
    
    return carrierIDs


def write_carrier_ids(isnap, carrierIDs):
    """Write the carrierIDs list for a specified snapshot to file."""
    print(f"   [writing carriers for snapshot {isnap}...]")
    carrier_ids_write = np.zeros(num_objs_total, dtype=np.int32) - 1
    carrier_ids_write[:len(carrierIDs[isnap])] = carrierIDs[isnap]
    with h5.File(output_file, 'a') as f:
        dname = f'MergeLists/CarrierIDs_{isnap:04d}'
        if dname in f.keys():
            del f[dname]

        dset = f.create_dataset(
            dname, (num_objs_total,), compression='lzf', chunks=(100000),
            dtype=carrier_ids_write.dtype
        )
        dset.write_direct(carrier_ids_write)
        dset.attrs['MaxObjID'] = len(carrierIDs[isnap]) - 1
        dset.attrs['Description'] = (
            f"Descendants of each galaxy in snapshot {isnap}. This list gives the "
            "object ID of the 'carrier' galaxy for each galaxy that has existed up to "
            "this snapshot, i.e. the one that it is part of in snapshot {isnap}. For "
            "galaxies that still exist independently (i.e. ones that have not been "
            "swallowed by another in a merger), the carrier ID is identical to their "
            "own object ID. To find the VR halo index of the carrier galaxy, use "
            f"table VRIndices/VRIndices_{isnap:04d} (but beware that this may not "
            "always point to a valid index, in case the carrier is skipped in "
            f"snapshot {isnap})."
        )

if __name__ == "__main__":
    main()
