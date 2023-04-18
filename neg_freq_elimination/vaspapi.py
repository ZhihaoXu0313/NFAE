import os
import time
import re
import numpy as np
import matplotlib.pyplot as plt

import subprocess
import ase.io

from utils import mkdir, grep, plot_band
from materials import get_kpath, get_bands


def submit_vasp_job(n_cores, index):
    vasp_exec = "vasp_std"
    vasp_output = "vasp.out"
    os.system("module load mpich/3.3/intel/19.0")
    os.system("mpirun -n " + str(n_cores) + " " + vasp_exec + " > " + vasp_output)
    while True:
        with open('OUTCAR', 'r') as f:
            last_line = f.readlines()[-1]
            if 'Voluntary context switches' in last_line or 'Elapsed time' in last_line:
                break
        time.sleep(20)
    print("The job ", index, " finished!")


def lc_calculation(rmin, rmax, num, material):
    """
    optimize lattice constant by manually adjust the alat and calculate the static energy;
    find the global minimum of function E(alat);
    should run in the directory of 'alat_opt';
    :param rmin: alat_min / alat_original
    :param rmax: alat_max / alat_original
    :param num: number of sampling points
    :param material: the instance of material (poscar)
    :return: a fitting result of E(alat) and the scale of poscar
    """
    mkdir("run")
    os.chdir("run")
    scale = np.linspace(rmin, rmax, num)
    for s in range(len(scale)):
        name = "run-" + str(s)
        mkdir(name)
        os.chdir(name)
        material.scaling(scale[s])
        material.write_POSCAR("POSCAR")

        generate_vasp_input("../../../../INCAR_files/INCAR-scf")

        submit_vasp_job(n_cores=48, index=str(s))
        os.chdir("..")

    E0 = []
    for s in range(len(scale)):
        name = "run-" + str(s)
        os.chdir(name)
        E0.append(float(grep("E0", "OSZICAR")[0].split()[4]))
        os.chdir("..")

    lc = scale * material.extract_poscar_info("box_coord")[0][0]
    z1 = np.polyfit(lc, E0, 2)
    lcmin = -z1[1] / (2 * z1[0])
    scale_opt = lcmin / material.extract_poscar_info("box_coord")[0][0]
    material.scaling(scale_opt)

    p1 = np.poly1d(z1)
    fE = p1(lc)
    Emin = p1(lcmin)

    plt.scatter(lc, E0)
    plt.plot(lc, fE, 'r', label='Fitting result')
    plt.plot(lcmin, Emin, 'g*', label='Minimum of Energy')
    plt.ylabel("Energy (eV)")
    plt.xlabel("Lattice Constant (Angstrom)")
    plt.title(material.mp_id)
    plt.legend()
    plt.savefig(material.mp_id + ".png")

    os.chdir("..")

    print("lattice constant optimization finished!")


def phonon_calculation(supercell, material):
    """
    calculate phonon structure of the material (Hessian, 2FC);
    should run in the directory of 'phonon_calculation';
    'supercell', 'Hessian' and '2FC' directories will be created automatically;
    :param supercell: size of supercell
    :param material: an instance of class 'Material'
    :return: after this procedure, there would be a FORCE_CONSTANTS in the directory '2FC'
    """
    mkdir("supercell")
    os.chdir("supercell")
    material.write_POSCAR("POSCAR")

    kpoints, label = get_kpath(ase.io.read("POSCAR", format='vasp'), format="seekpath")

    os.system("phonopy -d --dim='" + str(supercell[0]) + " " + str(supercell[1]) + " " + str(supercell[2]) + "'")
    os.chdir("..")

    mkdir("Hessian")
    os.chdir("Hessian")
    os.system("cp ../supercell/SPOSCAR ./POSCAR")

    generate_vasp_input("../../../../INCAR_files/INCAR-hessian")

    submit_vasp_job(n_cores=48, index="Hessian")
    os.chdir("..")

    mkdir("2FC")
    os.chdir("2FC")
    os.system("cp ../Hessian/vasprun.xml ./")
    subprocess.run(["phonopy", "--fc", "vasprun.xml"])

    band, band_label = get_bands(kpoints, label)
    atom_name = material.extract_poscar_info('atom')
    generate_band_conf(atom_name, supercell, band, band_label)

    os.system("cp ../supercell/POSCAR ./POSCAR-unitcell")
    os.system('phonopy --dim="' + str(supercell[0]) + ' ' + str(supercell[1]) + ' ' + str(supercell[2])
              + '" -c POSCAR-unitcell band.conf')

    plot_band(band, band_label)

    print("phonon calculation finished!")


def generate_vasp_input(INCAR_path):
    """
    generate INCAR, KPOINTS and POTCAR files for calculation;
    replace the ENCUT in the INCAR according to POTCAR
    :return: should have INCAR, KPOINTS and POTCAR after running
    """
    os.system('echo -e "102\n2\n0.04\n"| vaspkit')
    os.system('cp ' + INCAR_path + ' ./INCAR')

    max_enmax = grep("ENMAX", "POTCAR")[0].split()[2].replace(";", "")
    with open('INCAR', 'r') as f:
        incar_content = f.read()
    incar_content = re.sub(r'ENCUT *= *\d+', f'ENCUT = {max_enmax}', incar_content)
    with open('INCAR', 'w') as f:
        f.write(incar_content)
    print(f"ENCUT in INCAR replaced with {max_enmax}")


def generate_band_conf(atom_name, supercell, band, band_label):
    with open("band.conf", 'w') as f:
        f.writelines("ATOM_NAME = " + ' '.join(str(x) for x in atom_name) + "\n")
        f.writelines("DIM = " + str(supercell[0]) + " " + str(supercell[1]) + " " + str(supercell[2]) + "\n")
        f.writelines("PRIMITIVE_AXES = auto\n")
        f.write("BAND = ")
        for b in band:
            f.write(' '.join(str(x) for x in b))
            f.write("  ")
        f.write("\n")
        f.writelines("BAND_LABELS = " + ' '.join(str(x) for x in band_label) + "\n")
        f.writelines("FORCE_CONSTANTS = READ\n")


