import os
import pathlib
import numpy as np

import argparse

symbol_to_proton = {
    "H"  : 1 ,
    "He" : 2 ,
    "Li" : 3 ,
    "Be" : 4 ,
    "B"  : 5 ,
    "C"  : 6 ,
    "N"  : 7 ,
    "O"  : 8 ,
    "F"  : 9 ,
    "Ne" : 10 ,
    "Na" : 11 ,
    "Mg" : 12 ,
    "Al" : 13 ,
    "Si" : 14 ,
    "P"  : 15 ,
    "S"  : 16 ,
    "Cl" : 17 ,
    "Ar" : 18 ,
    "K"  : 19 ,
    "Ca" : 20 ,
    "Sc" : 21 ,
    "Ti" : 22 ,
    "V"  : 23 ,
    "Cr" : 24 ,
    "Mn" : 25 ,
    "Fe" : 26 ,
    "Co" : 27 ,
    "Ni" : 28 ,
    "Cu" : 29 ,
    "Zn" : 30 ,
    "Ga" : 31 ,
    "Ge" : 32 ,
    "As" : 33 ,
    "Se" : 34 ,
    "Br" : 35 ,
    "Kr" : 36 ,
    "Rb" : 37 ,
    "Sr" : 38 ,
    "Y"  : 39 ,
    "Zr" : 40 ,
    "Nb" : 41 ,
    "Mo" : 42 ,
    "Tc" : 43 ,
    "Ru" : 44 ,
    "Rh" : 45 ,
    "Pd" : 46 ,
    "Ag" : 47 ,
    "Cd" : 48 ,
    "In" : 49 ,
    "Sn" : 50 ,
    "Sb" : 51 ,
    "Te" : 52 ,
    "I"  : 53 ,
    "Xe" : 54 ,
    "Cs" : 55 ,
    "Ba" : 56 ,
    "La" : 57 ,
    "Ce" : 58 ,
    "Pr" : 59 ,
    "Nd" : 60 ,
    "Pm" : 61 ,
    "Sm" : 62 ,
    "Eu" : 63 ,
    "Gd" : 64 ,
    "Tb" : 65 ,
    "Dy" : 66 ,
    "Ho" : 67 ,
    "Er" : 68 ,
    "Tm" : 69 ,
    "Yb" : 70 ,
    "Lu" : 71 ,
    "Hf" : 72 ,
    "Ta" : 73 ,
    "W"  : 74 ,
    "Re" : 75 ,
    "Os" : 76 ,
    "Ir" : 77 ,
    "Pt" : 78 ,
    "Au" : 79 ,
    "Hg" : 80 ,
    "Tl" : 81 ,
    "Pb" : 82 ,
    "Bi" : 83 ,
    "Po" : 84 ,
    "At" : 85 ,
    "Rn" : 86 ,
    "Fr" : 87 ,
    "Ra" : 88 ,
    "Ac" : 89 ,
    "Th" : 90 ,
    "Pa" : 91 ,
    "U"  : 92 ,
    "Np" : 93 ,
    "Pu" : 94 ,
    "Am" : 95 ,
    "Cm" : 96 ,
    "Bk" : 97 ,
    "Cf" : 98 ,
    "Es" : 99 ,
    "Fm" : 100 ,
    "Md" : 101 ,
    "No" : 102 ,
    "Lr" : 103 ,
    "Rf" : 104 ,
    "Db" : 105 ,
    "Sg" : 106 ,
    "Bh" : 107 ,
    "Hs" : 108 ,
    "Mt" : 109 ,
    "Ds" : 110 ,
    "Rg" : 111 ,
    "Cn" : 112 ,
    "Nh" : 113 ,
    "Fl" : 114 ,
    "Mc" : 115 ,
    "Lv" : 116 ,
    "Ts" : 117 ,
    "Og" : 118 ,
}

def mol_to_dict(
        path : pathlib.Path or str
    ):

    state = 0
    # 0 : not read
    # 1 : reading header
    # 2 : reading count line
    # 3 : reading atom
    # 4 : reading connection
    # else : no need to care

    atom_dat = None
    connection_dat = None

    with open(path,'r') as f:
        state = 1
        line_cnt_state = 0 # line read in current state

        atom_cnt = 0
        connection_cnt = 0

        for lineno, line in enumerate(f):

            line_cnt_state +=1
            if state == 0:
                pass
            elif state == 1: # reading header
                if line_cnt_state == 3:
                    state+=1
                    line_cnt_state=0
            elif state == 2:
                words = line.split()
                atom_cnt, connection_cnt = int(line[0:3]), int(line[3:6])

                atom_dat = np.zeros(
                    atom_cnt,
                    dtype=[('x','f4'),('y','f4'),('z','f4'),('proton_no','i8')]
                )

                connection_dat = np.zeros(
                    connection_cnt,
                    dtype=[('v1','i8'),('v2','i8'),('conn_type','f4')]
                )

                state+=1
                line_cnt_state = 0
            elif state == 3:
                words = line.split()

                x,y,z = float(words[0]),float(words[1]),float(words[2])
                atom_symbol = str(words[3]).strip()

                atom_dat[line_cnt_state-1]= x,y,z, symbol_to_proton[atom_symbol]

                if line_cnt_state == atom_cnt:
                    state+=1
                    line_cnt_state = 0
            elif state == 4:

                v1, v2, conn_type = int(line[0:3]),int(line[3:6]),float(line[6:9])

                connection_dat[line_cnt_state-1]=v1,v2,conn_type

                if line_cnt_state == connection_cnt:
                    state+=1
                    line_cnt_state = 0
            else:
                break

        
    return atom_dat, connection_dat

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument('path_to_mol',type=pathlib.Path)

    args = parser.parse_args()

    atom_dat, conn_dat = mol_to_dict(args.path_to_mol)

    print(atom_dat)
    print(conn_dat)
    
    
