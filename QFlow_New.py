import numpy as np
from numpy import einsum
from itertools import combinations
from CCSD import *
from DUCC import *
# from FCI import *
import copy
import sys
import os
np.set_printoptions(threshold=sys.maxsize)


def read_Hamiltonian(file_name):
  
    print("Reading Hamiltonian")
    print("-------------------")

    with open(file_name, 'r') as file1:
        lines = file1.readlines()

    neles, norbs = map(int, lines[0].split())

    print("Electrons = ", neles)
    print("Orbitals  = ", norbs)

    # Allocate spatial one- and two-electron integrals
    oei = np.zeros((norbs, norbs))
    tei = np.zeros((norbs, norbs, norbs, norbs))
    rep_e = 0.0

    for line in lines:
        tokens = line.split()
        if len(tokens) == 3:
            p, q, val = tokens
            p, q = int(p) - 1, int(q) - 1 # Shift to python indexing

            val = float(val)
            oei[p, q] = val
            oei[q, p] = val
    
        elif len(tokens) == 5:
            p, q, r, s, val = tokens
            p, q, r, s = int(p) - 1, int(q) - 1, int(r) - 1, int(s) - 1, 

            val = float(val)
            if p == q == r == s == -1:
                rep_e = val

            else:   #(11|22) Format
                tei[p, q, r, s] = val
                tei[q, p, r, s] = val
                tei[p, q, s, r] = val
                tei[q, p, s, r] = val
                tei[r, s, p, q] = val
                tei[r, s, q, p] = val
                tei[s, r, p, q] = val
                tei[s, r, q, p] = val
            
    print("Computing Hartree-Fock energy ...")
    nocc = neles // 2
    o = slice(None, nocc)

    one_energy = 2.0 * einsum('ii', oei[o, o])
    two_energy = 2.0 * einsum('iijj', tei[o, o, o, o]) - 1.0 * einsum('ijji', tei[o, o, o, o])
    
    print("One Electron Energy = {: 5.8f}".format(one_energy))
    print("Two Electron Energy = {: 5.8f}".format(two_energy))
    print("Repulsion Energy    = {: 5.8f}".format(rep_e))
    print("Total SCF Energy    = {: 5.8f}".format(one_energy + two_energy + rep_e))

    del o, one_energy, two_energy

    return neles, norbs, oei, tei, rep_e

def spin_orb_Hamiltonian(norbs,neles,oei,tei,rep_e):
  
    soei = np.zeros((2 * norbs, 2 * norbs))
    stei = np.zeros((2 * norbs, 2 * norbs, 2 * norbs, 2 * norbs))

    for p in range(norbs):
        for q in range(norbs):
        # Populate 1-body coefficients. Require p and q have same spin.
            soei[2 * p, 2 * q] = oei[p, q]
            soei[2 * p + 1, 2 * q + 1] = oei[p, q]

        # Continue looping to prepare 2-body coefficients.
            for r in range(norbs):
                for s in range(norbs):
                    # [ij|kl] format, i.e. [11|22]
                    # Same spin
                    stei[2 * p, 2 * q, 2 * r, 2 * s] = tei[p, q, r, s]
                    stei[2 * p + 1, 2 * q + 1, 2 * r + 1, 2 * s + 1] = tei[p, q, r, s]
                    # # Mixed spin
                    stei[2 * p, 2 * q, 2 * r + 1, 2 * s + 1] = tei[p, q, r, s]
                    stei[2 * p + 1, 2 * q + 1, 2 * r, 2 * s] = tei[p, q, r, s]

    print("\nRe-computing Hartree-Fock energy ...")
    n = np.newaxis
    o = slice(None, neles)
    v = slice(neles, None)
 
    one_energy = 1.0 * einsum('ii', soei[o, o])
    two_energy = 0.5 * einsum('iijj', stei[o, o, o, o]) - 0.5 * einsum('ijji', stei[o, o, o, o])
        
    print("One Electron Energy = {: 5.8f}".format(one_energy))
    print("Two Electron Energy = {: 5.8f}".format(two_energy))
    print("Repulsion Energy    = {: 5.8f}".format(rep_e))
    print("Total SCF Energy    = {: 5.8f}".format(one_energy + two_energy + rep_e))

    # Antisymmetrize the two-electron integrals
    # <ij||kl> format with <12||12> ordering
    # <ij||kl> = <ij|kl> - <ij|lk>
    #          = [ik|jl] - [il|jk]
    gtei = np.einsum('ikjl', stei) - np.einsum('iljk', stei)

    print("\nHartree-Fock energy with antismmetrized tei (gtei)")
    two_energy = 0.5 * einsum('ijij', gtei[o, o, o, o])
    print("Two Electron Energy = {: 5.8f}".format(two_energy))
    print("Total SCF Energy    = {: 5.8f}".format(one_energy + two_energy + rep_e))

    # Form Fock operator
    fock = soei + np.einsum('piqi->pq', gtei[:, o, :, o])
    print("\n\tOrbital Energies")
    print("\t----------------")
    for p in range(norbs):
        print("{}     {: 5.6f}  {: 5.6f}".format(p + 1, fock[2 * p, 2 * p], fock[2 * p + 1, 2 * p + 1]))

    del one_energy, two_energy, soei

    return gtei, fock, o, v, n

def form_active_spaces(neles,norbs,nao,nav):
    nocc, nvirt = neles // 2, norbs - (neles // 2)
    occ_so_list = []
    virt_so_list = []

    for _ in range(nocc):
        occ_so_list.append(_)
    occ_combinations = list(combinations(occ_so_list[::-1], nao))

    for _ in range(nocc,nvirt+nocc):
        virt_so_list.append(_)
    virt_combinations = list(combinations(virt_so_list, nav))

    all_combinations = []
    for occ_set in occ_combinations:
        for virt_set in virt_combinations:
            all_combinations.append([list(occ_set)[::-1], list(virt_set), list(occ_set + virt_set)])

    print("Number of Active Space Combinations = ", len(all_combinations))

    return all_combinations

def get_key(val,xacc_dict):
   
    for key, value in xacc_dict.items():
        if val == value:
            return key
 
    return "key doesn't exist"

def main():

    print_thresh = 0.00000000001
    
    neles, norbs, oei, tei, rep_e = read_Hamiltonian("Bare")
    gtei, fock, o, v, n = spin_orb_Hamiltonian(norbs,neles,oei,tei,rep_e)
    
    # Set active occupied and virtual for the subproblems
    nao = 2
    nav = 2
    naso = 2 * (nao + nav)
    
    print("\nActive Space Occupied = ", nao)
    print("Active Space Virtuals = ", nav)
    
    o_as = slice(None, nao*2)
    # v_as = slice(nao*2, None)
    
    # GET ALL COMBINATIONS OF ACTIVE SPACES
    all_combos = form_active_spaces(neles,norbs,nao,nav)
    
    # CREATE THE MASTER POOL OF AMPLITUDES AND EXCITATIONS
    Master_T1_Excitations = []
    Master_T2_Excitations = []
    nsvirt = 2 * norbs - neles
    for i in range(neles):
        for a in range(nsvirt):
            Master_T1_Excitations.append([a,i])
            for j in range(neles):
                for b in range(nsvirt):
                    Master_T2_Excitations.append([a,b,i,j])
    
    Master_T1_Amplitudes = np.zeros((nsvirt, neles))
    Master_T2_Amplitudes = np.zeros((nsvirt, nsvirt, neles, neles))


# ***************************************************************************************************************************
# CCSD FOR DEBUGGING

    # t1f = np.zeros((nsvirt, neles))
    # t2f = np.zeros((nsvirt, nsvirt, neles, neles))

    # # INITIAL CCSD CALCULATION FOR TESTING
    # #     - SHOULD BE COMMENTED FOR ACTUAL RUNS 
    
    # eps = np.zeros((2 * norbs))
    # for p in range((2 * norbs)):
    #     eps[p] = fock[p,p]

    # print("Orbital Energies = ",eps)

    # d1_ai = 1 / (-eps[v, n] + eps[n, o])
    # d2_abij = 1 / (-eps[v, n, n, n] - eps[n, v, n, n] +
    #                 eps[n, n, o, n] + eps[n, n, n, o])

    # print("\nCOUPLED-CLUSTER ITERATIONS")
    # print('*'*25)
    # t1f, t2f = CC_kernel(rep_e, np.zeros((nsvirt, neles)), np.zeros((nsvirt, nsvirt, neles, neles)), fock, gtei, o, v, d1_ai, d2_abij)

# ***************************************************************************************************************************

# __________________________________________________________
#  ______     ______   __         ______     __     __       
# /\  __ \   /\  ___\ /\ \       /\  __ \   /\ \  _ \ \      
# \ \ \/\_\  \ \  __\ \ \ \____  \ \ \/\ \  \ \ \/ ".\ \     
#  \ \___\_\  \ \_\    \ \_____\  \ \_____\  \ \__/".~\_\    
#   \/___/_/   \/_/     \/_____/   \/_____/   \/_/   \/_/    
# __________________________________________________________    

    for cycle in range(1):
        print("\nCYCLE ", cycle)
        print("---------")

        Running_T1_Excitations = copy.copy(Master_T2_Excitations)
        Running_T2_Excitations = copy.copy(Master_T2_Excitations)

        for combo in all_combos:
        
            print("\nCombo =",combo)

            # SETUP ACTIVE SPACE INDEX LISTS
            ActSO_o = []
            ActSO_v = []
            for index in combo[0]:
                ActSO_o.append(2*(index))
                ActSO_o.append(2*(index)+1)
            for index in combo[1]:
                ActSO_v.append(2*(index))
                ActSO_v.append(2*(index)+1)

            combined_ActSO = ActSO_o + ActSO_v
            print("[Act_O] [Act_V] Spin Orbitals =", ActSO_o, ActSO_v)

            # CREATE T_EXT FOR COMBO FROM MASTER LIST.
            T1_ext = copy.copy(Master_T1_Amplitudes)
            T2_ext = copy.copy(Master_T2_Amplitudes)

            for i in range(neles,nsvirt+neles):
                for j in range(neles):
                    if (i in ActSO_v) and (j in ActSO_o):
                        T1_ext[(i-neles), j] = 0.0

            for i in range(neles,nsvirt+neles):
                for j in range(neles,nsvirt+neles):
                    for k in range(neles):
                        for l in range(neles):
                            if (i in ActSO_v) and (j in ActSO_v) and (k in ActSO_o) and (l in ActSO_o):
                                T2_ext[(i-neles), (j-neles), k, l] = 0.0


# ***************************************************************************************************************************

            # GET THE TRANSFORMED FOCK AND V
            Fock_Trans = np.zeros(((2 * norbs), (2 * norbs)))
            V_Trans = np.zeros(((2 * norbs), (2 * norbs), (2 * norbs), (2 * norbs)))

            # DUCC ROUTINES:
            H_0(Fock_Trans, V_Trans, T1_ext, T2_ext, fock, gtei, o, v)
            F_1(Fock_Trans, V_Trans, T1_ext, T2_ext, fock, gtei, o, v)
            V_1(Fock_Trans, V_Trans, T1_ext, T2_ext, fock, gtei, o, v)
            F_2(Fock_Trans, V_Trans, T1_ext, T2_ext, fock, gtei, o, v)
            # NOT TESTED:
            # V_2(Fock_Trans, V_Trans, T1_ext, T2_ext, fock, gtei, o, v)
            # F_3(Fock_Trans, V_Trans, T1_ext, T2_ext, fock, gtei, o, v)

            # FILL IN ELEMENTS
            Fock_Trans[v,o] = np.einsum('ia->ai', Fock_Trans[o,v])
            V_Trans[o,o,v,o] = np.einsum('ijka->jiak', V_Trans[o,o,o,v])
            V_Trans[o,v,o,o] = np.einsum('ijka->kaij', V_Trans[o,o,o,v])
            V_Trans[v,o,o,o] = np.einsum('ijka->akji', V_Trans[o,o,o,v])
            V_Trans[v,v,o,o] = np.einsum('ijab->abij', V_Trans[o,o,v,v])
            V_Trans[v,o,v,o] = np.einsum('iajb->aibj', V_Trans[o,v,o,v])
            V_Trans[o,v,v,o] = -1.0 * np.einsum('iajb->iabj', V_Trans[o,v,o,v])
            V_Trans[v,o,o,v] = -1.0 * np.einsum('iajb->aijb', V_Trans[o,v,o,v])
            V_Trans[v,o,v,v] = np.einsum('iabc->aicb', V_Trans[o,v,v,v])
            V_Trans[v,v,o,v] = np.einsum('iabc->bcia', V_Trans[o,v,v,v])
            V_Trans[v,v,v,o] = np.einsum('iabc->cbai', V_Trans[o,v,v,v])

# ***************************************************************************************************************************
# CREATING/FILLING ACTIVE SPACE TRANFORMED OPERATORS 

            oei_trans_as = np.zeros((naso, naso))
            fock_as = np.zeros((naso, naso))
            gtei_as = np.zeros((naso, naso, naso, naso))

            for p in range(naso):
                for q in range(naso):
                    fock_as[p, q] = Fock_Trans[combined_ActSO[p], combined_ActSO[q]]
                    for r in range(naso):
                        for s in range(naso):
                            gtei_as[p, q, r, s] = V_Trans[combined_ActSO[p], combined_ActSO[q], combined_ActSO[r], combined_ActSO[s]]

            for p in range(naso):
                for q in range(naso):
                      oei_trans_as[p, q] = fock_as[p, q]
                      for i in range(2 * nao):
                          oei_trans_as[p, q] += -1.0 * gtei_as[p, i, q, i]

            HF_E_trans = np.einsum('ii', Fock_Trans[o, o]) - 0.5 * np.einsum('ijij', V_Trans[o, o, o, o])
            HF_E_trans_as = np.einsum('ii', fock_as[o_as, o_as]) - 0.5 * np.einsum('ijij', gtei_as[o_as, o_as, o_as, o_as])
            adjusted_rep_e = HF_E_trans - HF_E_trans_as + rep_e
        
            print("Total Transformed SCF Energy  = {: 5.8f}".format(HF_E_trans + rep_e))
            print("Active Transformed SCF Energy = {: 5.8f}".format(HF_E_trans_as + rep_e))
            print("Adjusted Repulsion Energy     = {: 5.8f}".format(adjusted_rep_e))

# ***************************************************************************************************************************

            filename = '_'.join([str(index + 1) for index in combo[0]] + [str(index + 1) for index in combo[1]]) + "-xacc"

            # xacc_dict = {i: int((i // 2) + (i % 2) * (naso/2)) for i in range(naso)}
            ActSO_Total = ActSO_o + ActSO_v
            xacc_dict = {i: int((index // 2) + (index % 2) * (naso/2)) for index, i in enumerate(ActSO_Total)}
            print(xacc_dict)

            with open(filename, 'w') as f:

                for i in xacc_dict.values():
                    for j in xacc_dict.values():
                        
                        if(abs(oei_trans_as[i, j]) > print_thresh):
                            f.write("({: 12.10f},0){}^ {} +\n".format(oei_trans_as[i, j], i, j))

                        for k in xacc_dict.values():
                            for l in xacc_dict.values():

                                if(abs(gtei_as[i, j, k, l]) > print_thresh):
                                    f.write("({: 12.10f},0){}^ {}^ {} {} +\n".format(0.25 * (gtei_as[i, j, k, l]), i, j, l, k))

                f.write("({: 12.10f},0)".format(adjusted_rep_e))

# ***************************************************************************************************************************
            print("Excuting uccsd --n-orbitals {} --n-electrons {} --hamiltonian {} > {}.out".format(naso, nao*2 ,filename, filename))  
            os.system("uccsd --n-orbitals {} --n-electrons {} --hamiltonian {} > {}.out".format(naso, nao*2, filename, filename))  

# ***************************************************************************************************************************



if __name__ == "__main__":
    main()
