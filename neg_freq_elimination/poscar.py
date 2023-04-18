import numpy as np


class POSCAR:
    def __init__(self, file_path):
        self.poscar = {}
        self.file_path = file_path

    def loadPOSCAR(self):
        """
        collect info from POSCAR file
        :return: a dict of material info
        """
        file = open(self.file_path, 'r')
        f = file.readlines()
        self.poscar['comment'] = f[0].strip()
        self.poscar['scale_coeff'] = float(f[1])
        self.poscar['box_coord'] = []
        for i in range(2, 5):
            self.poscar['box_coord'].append([float(x) * self.poscar['scale_coeff'] for x in f[i].split()])
        self.poscar['atom'] = f[5].strip().split()
        self.poscar['atom_num'] = [int(x) for x in f[6].strip().split()]
        self.poscar['all_atom'] = sum(self.poscar['atom_num'])
        self.poscar['coord_type'] = 'Cartesian' if f[7].strip().startswith('C') or f[7].strip().startswith(
            'c') else 'Direct'
        self.poscar['coord_Cartesian'] = self.poscar['coord_Direct'] = []
        if self.poscar['coord_type'] == 'Cartesian':
            for i in range(8, 8 + self.poscar['all_atom']):
                self.poscar['coord_Cartesian'].append([float(x) for x in f[i].split()[:3]])
                self.poscar['coord_Direct'] = np.dot(np.linalg.inv(np.array(self.poscar['box_coord']).T),
                                                     np.array(self.poscar['coord_Cartesian']).T).T.tolist()
        else:
            for i in range(8, 8 + self.poscar['all_atom']):
                self.poscar['coord_Direct'].append([float(x) for x in f[i].split()[:3]])
                self.poscar['coord_Cartesian'] = np.dot(np.array(self.poscar['box_coord']).T,
                                                        np.array(self.poscar['coord_Direct']).T).T.tolist()
        file.close()

    def writePOSCAR(self, path):
        """
        recover the POSCAR file according to the dict
        :param path: path to the dict
        :return: write a POSCAR file
        """
        with open(path, 'w') as f:
            f.writelines(" " + self.poscar['comment'] + "\n")
            f.writelines("   " + str(self.poscar['scale_coeff']) + "\n")
            for i in range(3):
                f.writelines("     " + ' '.join(str('%.16f' % x) for x in self.poscar['box_coord'][i]) + "\n")
            f.writelines(" " + ' '.join(str(x) for x in self.poscar['atom']) + "\n")
            f.writelines("  " + ' '.join(str(x) for x in self.poscar['atom_num']) + "\n")
            f.writelines(self.poscar['coord_type'] + "\n")
            for i in range(self.poscar['all_atom']):
                if self.poscar['coord_type'] == 'Direct':
                    f.writelines("   " + ' '.join(str('%.16f' % x) for x in self.poscar['coord_Direct'][i]) + "\n")
        print("The POSCAR has been written to " + path)
