import os

from pymatgen.io.vasp.inputs import Poscar
import yaml
from pymatgen.core.structure import Structure
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

from materials import PhononDBMaterial, find_fmin
import vaspapi as api
import param
from utils import mkdir

if __name__ == "__main__":
    PDB_path = param.PDB_path
    imin = param.imin
    imax = param.imax
    wkdir = os.getcwd()
    mp_id_list = []
    material_list = os.listdir(PDB_path)
    for i in range(imin, imax + 1):
        fi = "mp-" + str(i)
        if fi in material_list:
            os.chdir(PDB_path + "/" + fi)
            structure = Structure.from_file("POSCAR-unitcell")
            analyzer = SpacegroupAnalyzer(structure)
            space_group_num = analyzer.get_space_group_number()
            fmin = find_fmin(directory="./")
            if (space_group_num >= 195) and (fmin < -0.2):
                print(fi, fmin)
                mp_id_list.append(fi)
            os.chdir("..")
    os.chdir(wkdir)
    for mp_id in mp_id_list:
        mkdir(mp_id)

    if param.method == "lcopt":
        for mp_id in mp_id_list:
            matid = PhononDBMaterial(mp_id, PDB_path)
            matid.get_POSCAR_unit()
            os.chdir(mp_id)
            mkdir("alat_opt")
            os.chdir("alat_opt")
            api.lc_calculation(rmin=0.95, rmax=1.05, num=15, material=matid)
            mkdir("phonon_calculation")
            os.chdir("phonon_calculation")
            api.phonon_calculation(supercell=[2, 2, 2], material=matid)
            os.chdir(wkdir)
    else:
        print("This mode has not been developed.")

