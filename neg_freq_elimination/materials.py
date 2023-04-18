import numpy as np
import os

from fractions import Fraction

from poscar import POSCAR

from phonopy import Phonopy
from phonopy.interface.vasp import read_vasp
from phonopy.file_IO import (parse_FORCE_SETS, parse_BORN)
from phonopy.phonon.tetrahedron_mesh import TetrahedronMesh
import phonopy.units as unit

import ase.io


class PhononDBMaterial:
    def __init__(self, mp_id, PDB_path):
        self.mp_id = mp_id
        self.PDB_path = PDB_path

        directory = self.PDB_path + "/" + str(self.mp_id)
        poscar_path = f"{directory}/POSCAR-unitcell"

        self.poscar = POSCAR(poscar_path)

    # operate material object via poscar instance
    def get_POSCAR_unit(self):
        print("Reading the structure of " + self.mp_id)
        self.poscar.loadPOSCAR()

    def write_POSCAR(self, path):
        self.poscar.writePOSCAR(path)

    def scaling(self, scale):
        self.poscar.poscar['scale_coeff'] = scale

    def extract_poscar_info(self, item):
        return self.poscar.poscar[item]


def read_conf(filename):
    ofs = open(filename, 'r')
    lines = ofs.readlines()
    cells = np.zeros((2, 9))
    for il in range(len(lines)):
        line = lines[il]
        data = line.split()
        for j in range(9):
            if "/" in data[2 + j]:
                cells[il, j] = Fraction(data[2 + j])
            else:
                cells[il, j] = float(data[2 + j])

    if len(lines) == 1:
        cells[1] = cells[0]

    return cells[0].reshape(3, 3), cells[1].reshape(3, 3)


def get_bands(kpoints, label):
    band = [kpoints[0, 0], kpoints[0, 1]]
    band_label = label[0]
    for i in range(1, len(kpoints)):
        if (kpoints[i, 0] == band[-1]).all():
            band.append(kpoints[i, 1])
            band_label.append(label[i][1])
        else:
            band += [",", kpoints[i, 0], kpoints[i, 1]]
            band_label += label[i]
    return band, band_label


def get_phonon(fn_poscar, fn_force, primat, nc_scell, fn_born='BORN'):
    """
    make phononpy object from unit cell and force set
    :param fn_poscar: POSCAR file name of unit cell
    :param fn_force: FORCE_SETS
    :param primat: cell matrix of primitive cell
    :param nc_scell: size of supercell
    :param fn_born: BORN
    :return:
    """
    unitcell = read_vasp(fn_poscar)

    try:
        phonon = Phonopy(unitcell, nc_scell, primitive_matrix=primat)
    except:
        return None

    force_sets = parse_FORCE_SETS(filename=fn_force)
    phonon.set_displacement_dataset(force_sets)
    phonon.produce_force_constants()
    phonon.symmetrize_force_constants()

    if fn_born is not None:
        primitive = phonon.get_primitive()
        nac_params = parse_BORN(primitive, filename=fn_born)
        nac_params['factor'] = 14.400
        phonon.set_nac_params(nac_params)

    return phonon


def get_symmetry(atoms):
    """
    get symmetry info of the structure
    :param atoms: atoms object of ase
    :return: POSCAR
    """
    import spglib
    symmetry = spglib.get_symmetry(atoms, symprec=1e-5)
    dataset = spglib.get_symmetry_dataset(atoms, symprec=1e-5,
                                          angle_tolerance=-1.0, hall_number=0)
    return dataset


def get_qpoints(fposcar, format="seekpath", deltak=0.02):
    atoms = ase.io.read(fposcar, format='vasp')
    kpoints, labels_tmp = get_kpath(atoms, format=format)

    labels = []
    nk = len(labels_tmp)
    labels.append(labels_tmp[0][0])
    for ik in range(nk):
        if ik == nk - 1:
            labels.append(labels_tmp[ik][1])
        else:
            if labels_tmp[ik][1] == labels_tmp[ik + 1][0]:
                labels.append(labels_tmp[ik][1])
            else:
                ll = "%s|%s" % (
                    labels_tmp[ik][1], labels_tmp[ik + 1][0])
                labels.append(ll)

    kband = []
    nsec = len(kpoints)
    for isec in range(nsec):
        kband.append([])
        kvec = kpoints[isec][1] - kpoints[isec][0]
        lk = np.linalg.norm(kvec)
        nk = int(lk / deltak) + 1
        if nk < 2:
            nk = 2
        for ik in range(nk):
            kcurr = kpoints[isec][0] + kvec * float(ik) / float(nk - 1)
            kband[isec].append(kcurr)

    return kband, kpoints, labels


def output_band_structure(outfile, ksymmetry, labels, qpoints, frequencies):
    nk_sym = len(labels)
    ofs = open(outfile, "w")

    ofs.write("#")
    for lab in labels:
        ofs.write(" %s" % lab)
    ofs.write("\n")

    ofs.write("#")
    kabs = 0.
    for isym in range(nk_sym):
        if isym != 0:
            kabs += np.linalg.norm(
                ksymmetry[isym - 1, 1] -
                ksymmetry[isym - 1, 0])
        ofs.write(" %.7f" % kabs)
    ofs.write("\n")

    ofs.write("# k-axis, Eigenvalues [cm^-1]\n")
    kabs = 0.
    for isym in range(nk_sym - 1):
        nkeach = len(qpoints[isym])
        for ik in range(nkeach):
            if ik != 0:
                kabs += np.linalg.norm(
                    qpoints[isym][ik] -
                    qpoints[isym][ik - 1]
                )
            ofs.write("%.7f" % kabs)
            for ff in frequencies[isym][ik]:
                ofs.write(" %15.7e" % (ff * unit.THzToCm))
            ofs.write("\n")


def get_kpath(atoms, format=None):
    if format == "seekpath":
        return get_kpath_seekpath(atoms)
    elif format == "pymatgen":
        return get_kpath_pymatgen(atoms)
    else:
        print("Error %s is not assigned." % format)
        exit()


def get_kpath_seekpath(atoms):
    import seekpath
    structure = [atoms.cell, atoms.get_scaled_positions(), atoms.numbers]
    kpath = seekpath.get_path(structure, with_time_reversal=True, recipe='hpkot')
    kpoints, labels = _extract_kpath_seekpath(kpath)
    return kpoints, labels


def get_kpath_pymatgen(atoms):
    from pymatgen.io.ase import AseAtomsAdaptor as pyase
    structure = pyase.get_structure(atoms)
    kpoints, labels = get_ksymmetry_pymatgen(structure)
    return kpoints, labels


def _extract_kpath_seekpath(kpaths):
    nk = len(kpaths['path'])
    kpoints = np.zeros((nk, 2, 3))
    labels = []
    for ik, points in enumerate(kpaths['path']):
        labels.append([])
        for i in range(2):
            for j in range(3):
                kpoints[ik, i, j] = kpaths['point_coords'][points[i]][j]
            labels[ik].append(points[i])
    return kpoints, labels


def get_ksymmetry_pymatgen(structure):
    from pymatgen.symmetry.bandstructure import HighSymmKpath
    kpath, labels_dump = HighSymmKpath(structure).get_kpoints(line_density=1)

    kall = []
    laball = []
    nk = len(kpath)
    for ik in range(nk):
        if labels_dump[ik] != "":
            kall.append(kpath[ik])
            laball.append(labels_dump[ik])
    nkall = len(kall)

    nsec = int(nkall / 2)
    kpoints = np.zeros((nsec, 2, 3))
    labels = []
    for isec in range(nsec):
        labels.append([])
        for i in range(2):
            num = isec * 2 + i
            for j in range(3):
                kpoints[isec, i, j] = kall[num][j]
            labels[isec].append(laball[num])
    return kpoints, labels


def find_fmin(directory, lkpath="seekpath", deltak=0.02):
    """
    find the min frequency in the phonon structure
    :param directory: directory of phonondb (in order to find phononpy.conf and other files)
    :param lkpath: method to find kpath
    :param deltak: k-mesh grid
    :return: the min freq
    """
    fn = directory + '/' + 'phonopy.conf'
    nc_scell, primat = read_conf(fn)

    fn_pos = directory + '/' + 'POSCAR-unitcell'
    fn_force = directory + '/' + 'FORCE_SETS'
    fn_born = directory + '/' + 'BORN'

    if not os.path.exists(fn_born):
        fn_born = None
    phonon = get_phonon(fn_pos, fn_force, primat, nc_scell, fn_born=fn_born)

    prim_phonopy = phonon.get_primitive()
    prim = ase.Atoms(cell=prim_phonopy.cell, pbc=True)
    for ia in range(len(prim_phonopy)):
        prim.append(ase.Atom(symbol=prim_phonopy.get_chemical_symbols()[ia],
                             position=prim_phonopy.get_positions()[ia]))

    symmetry = get_symmetry(prim)
    sym_name = symmetry['international']
    if phonon is None:
        print('%s Nan' % (prim.get_chemical_formula()))
        exit()

    kband, ksymmetry, labels = get_qpoints(fn_pos, format=lkpath, deltak=deltak)
    phonon.set_band_structure(kband)
    qpoints, distances, frequencies, eigenvectors = phonon.get_band_structure()

    output_band_structure('phonopy.bands', ksymmetry, labels,
                          qpoints, frequencies)

    fmin = 1000.
    fmax = -1000.
    for freq in frequencies:
        fmin = min(fmin, np.amin(freq))
        fmax = max(fmax, np.amax(freq))

    phonon.set_mesh([10, 10, 10])
    phonon.set_total_DOS(freq_min=fmin, freq_max=fmax * 1.05, freq_pitch=0.05,
                         tetrahedron_method=True)

    return fmin
