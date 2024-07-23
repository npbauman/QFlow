import numpy as np
from numpy import einsum

def ccsd_energy(t1, t2, f, g, o, v):

    #   HF ENERGY

    #	  1.0*f(i,i)
    energy = 1.0 * einsum('ii', f[o, o])
    #	 -(1.0/2.0)*<i,j||i,j>
    energy += -0.5 * einsum('ijij', g[o, o, o, o])

    # print("SCF Energy  from CC Calculation = ", energy)
    #   CC CORRELATION

    #   1.0*f(m,e)*t1(e,m)
    energy += 1.0 * einsum('me,em', f[o, v], t1, optimize=True)
    #   (1.0/4.0)*v(m,n,e,f)*t2(e,f,m,n)
    energy += 0.25 * einsum('mnef,efmn', g[o, o, v, v], t2, optimize=True)
    #   (1.0/2.0)*v(m,n,e,f)*t1(e,m)*t1(f,n)
    energy += 0.5 * einsum('mnef,em,fn', g[o, o, v, v], t1, t1, optimize=True)

    return energy

def singles_residual(t1, t2, f, g, o, v):

    #   1.0*f(a,i)
    singles_res = 1.0 * einsum('ai->ai', f[v, o])
    #   -1.0*f(m,i)*t1(a,m)
    singles_res += -1.0 * einsum('mi,am->ai', f[o, o], t1, optimize=True)
    #   1.0*f(a,e)*t1(e,i)
    singles_res += 1.0 * einsum('ae,ei->ai', f[v, v], t1, optimize=True)
    #   1.0*f(m,e)*t2(a,e,i,m)
    singles_res += 1.0 * einsum('me,aeim->ai', f[o, v], t2, optimize=True)
    #   -1.0*f(m,e)*t1(e,i)*t1(a,m)
    singles_res += -1.0 * einsum('me,ei,am->ai', f[o, v], t1, t1, optimize=True)
    #   1.0*v(i,e,a,m)*t1(e,m)
    singles_res += 1.0 * einsum('ieam,em->ai', g[o, v, v, o], t1, optimize=True)
    #   -(1.0/2.0)*v(m,n,i,e)*t2(a,e,m,n)
    singles_res += -0.5 * einsum('mnie,aemn->ai', g[o, o, o, v], t2, optimize=True)
    #   (1.0/2.0)*v(m,a,f,e)*t2(e,f,i,m)
    singles_res += 0.5 * einsum('mafe,efim->ai', g[o, v, v, v], t2, optimize=True)
    #   -1.0*v2(m,n,i,e)*t1(a,m)*t1(e,n)
    singles_res += -1.0 * einsum('mnie,am,en->ai', g[o, o, o, v], t1, t1, optimize=True)
    #   1.0*v2(m,a,f,e)*t1(e,i)*t1(f,m)
    singles_res += 1.0 * einsum('mafe,ei,fm->ai', g[o, v, v, v], t1, t1, optimize=True)
    #   -(1.0/2.0)*v2(m,n,e,f)*t1(e,i)*t2(a,f,m,n)
    singles_res += -0.5 * einsum('mnef,ei,afmn->ai', g[o, o, v, v], t1, t2, optimize=True)
    #   -(1.0/2.0)*v2(m,n,e,f)*t1(a,m)*t2(e,f,i,n)
    singles_res += -0.5 * einsum('mnef,am,efin->ai', g[o, o, v, v], t1, t2, optimize=True)
    #   1.0*v2(m,n,e,f)*t1(e,m)*t2(a,f,i,n)
    singles_res += 1.0 * einsum('mnef,em,afin->ai', g[o, o, v, v], t1, t2, optimize=True)
    #   -1.0*v2ijab(m,n,e,f)*t1(e,i)*t1(a,m)*t1(f,n)
    singles_res += -1.0 * einsum('mnef,ei,am,fn->ai', g[o, o, v, v], t1, t1, t1, optimize=True)

    return singles_res

def doubles_residual(t1, t2, f, g, o, v):

    #  1.0 * v(a, b, i, j)
    doubles_res =  1.0 * einsum('abij->abij', g[v, v, o, o])
    #  1.0 * f(m, i) * t2(a, b, j, m)
    doubles_res +=  1.0 * einsum('mi,abjm->abij', f[o, o], t2, optimize=True)
    #  -1.0 * f(a, e) * t2(b, e, i, j)
    doubles_res +=  -1.0 * einsum('ae,beij->abij', f[v, v], t2, optimize=True)
    #  -1.0 * f(m, j) * t2(a, b, i, m)
    doubles_res +=  -1.0 * einsum('mj,abim->abij', f[o, o], t2, optimize=True)
    #  1.0 * f(b, e) * t2(a, e, i, j)
    doubles_res +=  1.0 * einsum('be,aeij->abij', f[v, v], t2, optimize=True)
    #  1.0 * f(m, e) * t1(e, i) * t2(a, b, j, m)
    doubles_res +=  1.0 * einsum('me,ei,abjm->abij', f[o, v], t1, t2, optimize=True)
    #  1.0 * f(m, e) * t1(a, m) * t2(b, e, i, j)
    doubles_res +=  1.0 * einsum('me,am,beij->abij', f[o, v], t1, t2, optimize=True)
    #  -1.0 * f(m, e) * t1(e, j) * t2(a, b, i, m)
    doubles_res +=  -1.0 * einsum('me,ej,abim->abij', f[o, v], t1, t2, optimize=True)
    #  -1.0 * f(m, e) * t1(b, m) * t2(a, e, i, j)
    doubles_res +=  -1.0 * einsum('me,bm,aeij->abij', f[o, v], t1, t2, optimize=True)
    #  -1.0 * v(a, m, i, j) * t1(b, m)
    doubles_res +=  -1.0 * einsum('amij,bm->abij', g[v, o, o, o], t1, optimize=True)
    #  1.0 * v(a, b, i, e) * t1(e, j)
    doubles_res +=  1.0 * einsum('abie,ej->abij', g[v, v, o, v], t1, optimize=True)
    #  1.0 * v(b, m, i, j) * t1(a, m)
    doubles_res +=  1.0 * einsum('bmij,am->abij', g[v, o, o, o], t1, optimize=True)
    #  -1.0 * v(a, b, j, e) * t1(e, i)
    doubles_res +=  -1.0 * einsum('abje,ei->abij', g[v, v, o, v], t1, optimize=True)
    #  1.0 * v(a, m, i, e) * t2(b, e, j, m)
    doubles_res += 1.0 * einsum('amie,bejm->abij', g[v, o, o, v], t2, optimize=True)
    #  (1.0/2.0) * v(m, n, i, j) * t2(a, b, m, n)
    doubles_res +=  0.5 * einsum('mnij,abmn->abij', g[o, o, o, o], t2, optimize=True)
    #  1.0 * v(b, m, e, i) * t2(a, e, j, m)
    doubles_res +=  1.0 * einsum('bmei,aejm->abij', g[v, o, v, o], t2, optimize=True)
    #  1.0 * v(a, m, e, j) * t2(b, e, i, m)
    doubles_res +=  1.0 * einsum('amej,beim->abij', g[v, o, v, o], t2, optimize=True)
    #  (1.0/2.0) * v(a, b, e, f) * t2(e, f, i, j)
    doubles_res +=  0.5 * einsum('abef,efij->abij', g[v, v, v, v], t2, optimize=True)
    #  -1.0 * v(b, m, e, j) * t2(a, e, i, m)
    doubles_res +=  -1.0 * einsum('bmej,aeim->abij', g[v, o, v, o], t2, optimize=True)
    #  1.0 * v(a, m, e, i) * t1(e, j) * t1(b, m)
    doubles_res +=  1.0 * einsum('amei,ej,bm->abij', g[v, o, v, o], t1, t1, optimize=True)
    #  1.0 * vl(m, n, i, j) * t1(a, m) * t1(b, n)
    doubles_res +=  1.0 * einsum('mnij,am,bn->abij', g[o, o, o, o], t1, t1, optimize=True)
    #  -1.0 * v(b, m, e, i) * t1(a, m) * t1(e, j)
    doubles_res +=  -1.0 * einsum('bmei,am,ej->abij', g[v, o, v, o], t1, t1, optimize=True)
    #  -1.0 * v(a, m, e, j) * t1(e, i) * t1(b, m)
    doubles_res +=  -1.0 * einsum('amej,ei,bm->abij', g[v, o, v, o], t1, t1, optimize=True)
    #  1.0 * v(a, b, e, f) * t1(e, i) * t1(f, j)
    doubles_res +=  1.0 * einsum('abef,ei,fj->abij', g[v, v, v, v], t1, t1, optimize=True)
    #   1.0 * v(b, m, e, j) * t1(e, i) * t1(a, m)
    doubles_res +=   1.0 * einsum('bmej,ei,am->abij', g[v, o, v, o], t1, t1, optimize=True)
    #  -1.0 * v(m, n, i, e) * t1(a, m) * t2(b, e, j, n)
    doubles_res +=  -1.0 * einsum('mnie,am,bejn->abij', g[o, o, o, v], t1, t2, optimize=True)
    #  (1.0/2.0) * v(m, n, i, e) * t1(e, j) * t2(a, b, m, n)
    doubles_res +=  0.5 * einsum('mnie,ej,abmn->abij', g[o, o, o, v], t1, t2, optimize=True)
    #  1.0 * v(m, n, i, e) * t1(b, m) * t2(a, e, j, n)
    doubles_res +=  1.0 * einsum('mnie,bm,aejn->abij', g[o, o, o, v], t1, t2, optimize=True)
    #  -1.0 * v(m, n, i, e) * t1(e, m) * t2(a, b, j, n)
    doubles_res +=  -1.0 * einsum('mnie,em,abjn->abij', g[o, o, o, v], t1, t2, optimize=True)
    #  1.0 * v(m, a, f, e) * t1(e, i) * t2(b, f, j, m)
    doubles_res +=  1.0 * einsum('mafe,ei,bfjm->abij', g[o, v, v, v], t1, t2, optimize=True)
    #  -(1.0/2.0) * v(m, n, j, e) * t1(e, i) * t2(a, b, m, n)
    doubles_res +=  -0.5 * einsum('mnje,ei,abmn->abij', g[o, o, o, v], t1, t2, optimize=True)
    #  -1.0 * v(m, b, f, e) * t1(e, i) * t2(a, f, j, m)
    doubles_res +=  -1.0 * einsum('mbfe,ei,afjm->abij', g[o, v, v, v], t1, t2, optimize=True)
    #  -1.0 * v(m, a, f, e) * t1(e, j) * t2(b, f, i, m)
    doubles_res +=  -1.0 * einsum('mafe,ej,bfim->abij', g[o, v, v, v], t1, t2, optimize=True)
    #  -(1.0/2.0) * v(m, a, f, e) * t1(b, m) * t2(e, f, i, j)
    doubles_res +=  -0.5 * einsum('mafe,bm,efij->abij', g[o, v, v, v], t1, t2, optimize=True)
    #  1.0 * v(m, a, f, e) * t1(e, m) * t2(b, f, i, j)
    doubles_res +=  1.0 * einsum('mafe,em,bfij->abij', g[o, v, v, v], t1, t2, optimize=True)
    #  1.0 * v(m, n, j, e) * t1(a, m) * t2(b, e, i, n)
    doubles_res +=  1.0 * einsum('mnje,am,bein->abij', g[o, o, o, v], t1, t2, optimize=True)
    #  (1.0/2.0) * v(m, b, f, e) * t1(a, m) * t2(e, f, i, j)
    doubles_res += 0.5 * einsum('mbfe,am,efij->abij', g[o, v, v, v], t1, t2, optimize=True)
    #  -1.0 * v(m, n, j, e) * t1(b, m) * t2(a, e, i, n)
    doubles_res +=  -1.0 * einsum('mnje,bm,aein->abij', g[o, o, o, v], t1, t2, optimize=True)
    #  1.0 * v(m, n, j, e) * t1(e, m) * t2(a, b, i, n)
    doubles_res +=  1.0 * einsum('mnje,em,abin->abij', g[o, o, o, v], t1, t2, optimize=True)
    #  1.0 * v(m, b, f, e) * t1(e, j) * t2(a, f, i, m)
    doubles_res +=  1.0 * einsum('mbfe,ej,afim->abij', g[o, v, v, v], t1, t2, optimize=True)
    #  -1.0 * v(m, b, f, e) * t1(e, m) * t2(a, f, i, j)
    doubles_res +=  -1.0 * einsum('mbfe,em,afij->abij', g[o, v, v, v], t1, t2, optimize=True)
    #  -(1.0/2.0) * v(m, n, e, f) * t2(a, e, i, j) * t2(b, f, m, n)
    doubles_res +=  -0.5 * einsum('mnef,aeij,bfmn->abij', g[o, o, v, v], t2, t2, optimize=True)
    #  -(1.0/2.0) * v(m, n, e, f) * t2(a, b, i, m) * t2(e, f, j, n)
    doubles_res +=  -0.5 * einsum('mnef,abim,efjn->abij', g[o, o, v, v], t2, t2, optimize=True)
    #  1.0 * v(m, n, e, f) * t2(a, e, i, m) * t2(b, f, j, n)
    doubles_res +=  1.0 * einsum('mnef,aeim,bfjn->abij', g[o, o, v, v], t2, t2, optimize=True)
    #  (1.0/2.0) * v(m, n, e, f) * t2(b, e, i, j) * t2(a, f, m, n)
    doubles_res +=  0.5 * einsum('mnef,beij,afmn->abij', g[o, o, v, v], t2, t2, optimize=True)
    #  (1.0/4.0) * v(m, n, e, f) * t2(e, f, i, j) * t2(a, b, m, n)
    doubles_res +=  0.25 * einsum('mnef,efij,abmn->abij', g[o, o, v, v], t2, t2, optimize=True)
    #  -1.0 * v(m, n, e, f) * t2(b, e, i, m) * t2(a, f, j, n)
    doubles_res +=  -1.0 * einsum('mnef,beim,afjn->abij', g[o, o, v, v], t2, t2, optimize=True)
    #  -(1.0/2.0) * v(m, n, e, f) * t2(e, f, i, m) * t2(a, b, j, n)
    doubles_res +=  -0.5 * einsum('mnef,efim,abjn->abij', g[o, o, v, v], t2, t2, optimize=True)
    #  1.0 * v(m, n, i, e) * t1(a, m) * t1(e, j) * t1(b, n)
    doubles_res +=  1.0 * einsum('mnie,am,ej,bn->abij', g[o, o, o, v], t1, t1, t1, optimize=True)
    #  -1.0 * v(m, a, f, e) * t1(e, i) * t1(f, j) * t1(b, m)
    doubles_res +=  -1.0 * einsum('mafe,ei,fj,bm->abij', g[o, v, v, v], t1, t1, t1, optimize=True)
    #  -1.0 * v(m, n, j, e) * t1(e, i) * t1(a, m) * t1(b, n)
    doubles_res +=  -1.0 * einsum('mnje,ei,am,bn->abij', g[o, o, o, v], t1, t1, t1, optimize=True)
    #  1.0 * v(m, b, f, e) * t1(e, i) * t1(a, m) * t1(f, j)
    doubles_res +=  1.0 * einsum('mbfe,ei,am,fj->abij', g[o, v, v, v], t1, t1, t1, optimize=True)
    #  -1.0 * v(m, n, e, f) * t1(e, i) * t1(a, m) * t2(b, f, j, n)
    doubles_res +=  -1.0 * einsum('mnef,ei,am,bfjn->abij', g[o, o, v, v], t1, t1, t2, optimize=True)
    #  (1.0/2.0) * v(m, n, e, f) * t1(e, i) * t1(f, j) * t2(a, b, m, n)
    doubles_res +=  0.5 * einsum('mnef,ei,fj,abmn->abij', g[o, o, v, v], t1, t1, t2, optimize=True)
    #  1.0 * v(m, n, e, f) * t1(e, i) * t1(b, m) * t2(a, f, j, n)
    doubles_res +=  1.0 * einsum('mnef,ei,bm,afjn->abij', g[o, o, v, v], t1, t1, t2, optimize=True)
    #  -1.0 * v(m, n, e, f) * t1(e, i) * t1(f, m) * t2(a, b, j, n)
    doubles_res +=  -1.0 * einsum('mnef,ei,fm,abjn->abij', g[o, o, v, v], t1, t1, t2, optimize=True)
    #  1.0 * v(m, n, e, f) * t1(a, m) * t1(e, j) * t2(b, f, i, n)
    doubles_res +=  1.0 * einsum('mnef,am,ej,bfin->abij', g[o, o, v, v], t1, t1, t2, optimize=True)
    #  (1.0/2.0) * v(m, n, e, f) * t1(a, m) * t1(b, n) * t2(e, f, i, j)
    doubles_res +=  0.5 * einsum('mnef,am,bn,efij->abij', g[o, o, v, v], t1, t1, t2, optimize=True)
    #  -1.0 * v(m, n, e, f) * t1(a, m) * t1(e, n) * t2(b, f, i, j)
    doubles_res +=  -1.0 * einsum('mnef,am,en,bfij->abij', g[o, o, v, v], t1, t1, t2, optimize=True)
    #  -1.0 * v(m, n, e, f) * t1(e, j) * t1(b, m) * t2(a, f, i, n)
    doubles_res +=  -1.0 * einsum('mnef,ej,bm,afin->abij', g[o, o, v, v], t1, t1, t2, optimize=True)
    #  1.0 * v(m, n, e, f) * t1(e, j) * t1(f, m) * t2(a, b, i, n)
    doubles_res +=  1.0 * einsum('mnef,ej,fm,abin->abij', g[o, o, v, v], t1, t1, t2, optimize=True)
    #  1.0 * v(m, n, e, f) * t1(b, m) * t1(e, n) * t2(a, f, i, j)
    doubles_res +=  1.0 * einsum('mnef,bm,en,afij->abij', g[o, o, v, v], t1, t1, t2, optimize=True)
    #  1.0 * v(m, n, e, f) * t1(e, i) * t1(a, m) * t1(f, j) * t1(b, n)
    doubles_res +=  1.0 * einsum('mnef,ei,am,fj,bn->abij', g[o, o, v, v], t1, t1, t1, t1, optimize=True)

    return doubles_res

def CC_kernel(rep_e, t1, t2, fock, g, o, v, d1_ai, d2_abij, max_iter=100, thresh=1.0E-11):

    e_ai = np.reciprocal(d1_ai)
    e_abij = np.reciprocal(d2_abij)
    old_energy = ccsd_energy(t1, t2, fock, g, o, v)
    for idx in range(max_iter):

        singles_res = singles_residual(t1, t2, fock, g, o, v) + e_ai * t1
        doubles_res = doubles_residual(t1, t2, fock, g, o, v) + e_abij * t2

        new_singles = singles_res * d1_ai
        new_doubles = doubles_res * d2_abij

        current_energy = ccsd_energy(new_singles, new_doubles, fock, g, o, v)
        delta_e = np.abs(old_energy - current_energy)

        if delta_e < thresh:
            print("FINAL ENERGY = ",old_energy + rep_e)
            return new_singles, new_doubles
        else:
            t1 = new_singles
            t2 = new_doubles
            old_energy = current_energy
            # print("\tIteration {: 5d}\t{: 5.15f}\t{: 5.15f}".format(idx, old_energy + rep_e, delta_e))
    else:
        print("Did not converge")
        return new_singles, new_doubles
