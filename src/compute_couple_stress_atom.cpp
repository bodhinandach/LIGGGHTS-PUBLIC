/* ----------------------------------------------------------------------
    This is the

    ██╗     ██╗ ██████╗  ██████╗  ██████╗ ██╗  ██╗████████╗███████╗
    ██║     ██║██╔════╝ ██╔════╝ ██╔════╝ ██║  ██║╚══██╔══╝██╔════╝
    ██║     ██║██║  ███╗██║  ███╗██║  ███╗███████║   ██║   ███████╗
    ██║     ██║██║   ██║██║   ██║██║   ██║██╔══██║   ██║   ╚════██║
    ███████╗██║╚██████╔╝╚██████╔╝╚██████╔╝██║  ██║   ██║   ███████║
    ╚══════╝╚═╝ ╚═════╝  ╚═════╝  ╚═════╝ ╚═╝  ╚═╝   ╚═╝   ╚══════╝®

    DEM simulation engine, released by
    DCS Computing Gmbh, Linz, Austria
    http://www.dcs-computing.com, office@dcs-computing.com

    LIGGGHTS® is part of CFDEM®project:
    http://www.liggghts.com | http://www.cfdem.com

    Core developer and main author:
    Christoph Kloss, christoph.kloss@dcs-computing.com

    LIGGGHTS® is open-source, distributed under the terms of the GNU Public
    License, version 2 or later. It is distributed in the hope that it will
    be useful, but WITHOUT ANY WARRANTY; without even the implied warranty
    of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. You should have
    received a copy of the GNU General Public License along with LIGGGHTS®.
    If not, see http://www.gnu.org/licenses . See also top-level README
    and LICENSE files.

    LIGGGHTS® and CFDEM® are registered trade marks of DCS Computing GmbH,
    the producer of the LIGGGHTS® software and the CFDEM®coupling software
    See http://www.cfdem.com/terms-trademark-policy for details.

-------------------------------------------------------------------------
    Contributing author and copyright for this file:
    This file is from LAMMPS, but has been modified. Copyright for
    modification:

    Copyright 2012-     DCS Computing GmbH, Linz
    Copyright 2009-2012 JKU Linz

    Copyright of original file:
    LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
    http://lammps.sandia.gov, Sandia National Laboratories
    Steve Plimpton, sjplimp@sandia.gov

    Copyright (2003) Sandia Corporation.  Under the terms of Contract
    DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
    certain rights in this software.  This software is distributed under
    the GNU General Public License.
------------------------------------------------------------------------- */

#include <stdlib.h>
#include <string.h>
#include <cmath>        // std::atan2
#include "compute_couple_stress_atom.h"
#include "atom.h"
#include "update.h"
#include "comm.h"
#include "force.h"
#include "pair.h"
#include "bond.h"
#include "angle.h"
#include "dihedral.h"
#include "improper.h"
#include "kspace.h"
#include "modify.h"
#include "fix.h"
#include "memory.h"
#include "error.h"

using namespace LAMMPS_NS;

/* ---------------------------------------------------------------------- */

ComputeCoupleStressAtom::ComputeCoupleStressAtom(LAMMPS *lmp, int &iarg, int narg, char **arg) :
  Compute(lmp, iarg, narg, arg)
{
  if (narg < iarg) error->all(FLERR,"Illegal compute couplestress/atom command");

  peratom_flag = 1;
  size_peratom_cols = 9;
  pressatomflag = 1;
  timeflag = 1;
  comm_forward = 9;
  comm_reverse = 9;

  if (narg == iarg) {
    keflag = 1;
    pairflag = 1;
    bondflag = angleflag = dihedralflag = improperflag = 1;
    kspaceflag = 1;
    fixflag = 1;
    cylflag = 0;
  } else {
    keflag = 0;
    pairflag = 0;
    bondflag = angleflag = dihedralflag = improperflag = 0;
    kspaceflag = 0;
    fixflag = 0;
    cylflag = 0;
    while (iarg < narg) {
      if (strcmp(arg[iarg],"ke") == 0) keflag = 1;
      else if (strcmp(arg[iarg],"pair") == 0) pairflag = 1;
      else if (strcmp(arg[iarg],"cyl") == 0) cylflag = 1; // FIXME: Only work for x axis rotation
      else if (strcmp(arg[iarg],"bond") == 0) bondflag = 1;
      else if (strcmp(arg[iarg],"angle") == 0) angleflag = 1;
      else if (strcmp(arg[iarg],"dihedral") == 0) dihedralflag = 1;
      else if (strcmp(arg[iarg],"improper") == 0) improperflag = 1;
      else if (strcmp(arg[iarg],"kspace") == 0) kspaceflag = 1;
      else if (strcmp(arg[iarg],"fix") == 0) fixflag = 1;
      else if (strcmp(arg[iarg],"virial") == 0) {
        pairflag = 1;
        bondflag = angleflag = dihedralflag = improperflag = 1;
        kspaceflag = fixflag = 1;
      } else error->all(FLERR,"Illegal compute couplestress/atom command");
      iarg++;
    }
  }

  nmax = 0;
  stress = NULL;
}

/* ---------------------------------------------------------------------- */

ComputeCoupleStressAtom::~ComputeCoupleStressAtom()
{
  memory->destroy(stress);
}

/* ---------------------------------------------------------------------- */

void ComputeCoupleStressAtom::compute_peratom()
{
  int i,j;
  double onemass;

  invoked_peratom = update->ntimestep;
  if (update->vflag_atom != invoked_peratom)
    error->all(FLERR,"Per-atom virial was not tallied on needed timestep");

  // grow local stress array if necessary
  // needs to be atom->nmax in length

  if (atom->nmax > nmax) {
    memory->destroy(stress);
    nmax = atom->nmax;
    memory->create(stress,nmax,9,"couplestress/atom:stress");
    array_atom = stress;
  }

  // npair includes ghosts if either newton flag is set
  //   b/c some bonds/dihedrals call pair::ev_tally with pairwise info
  // nbond includes ghosts if newton_bond is set
  // ntotal includes ghosts if either newton flag is set
  // KSpace includes ghosts if tip4pflag is set

  int nlocal = atom->nlocal;
  int npair = nlocal;
  int nbond = nlocal;
  int ntotal = nlocal;
  int nkspace = nlocal;
  if (force->newton) npair += atom->nghost;
  if (force->newton_bond) nbond += atom->nghost;
  if (force->newton) ntotal += atom->nghost;
  if (force->kspace && force->kspace->tip4pflag) nkspace += atom->nghost;

  // clear local stress array

  for (i = 0; i < ntotal; i++)
    for (j = 0; j < 9; j++)
      stress[i][j] = 0.0;

  // add in per-atom contributions from each force

  if (pairflag && force->pair) {
    double **muatom = force->pair->muatom;
    for (i = 0; i < npair; i++)
      for (j = 0; j < 9; j++)
        stress[i][j] += muatom[i][j];
  }

  if (bondflag && force->bond) {
    double **vatom = force->bond->vatom;
    for (i = 0; i < nbond; i++) {
      for (j = 0; j < 6; j++)
        stress[i][j] += vatom[i][j];
      for (j = 6; j < 9; j++)
        stress[i][j] += vatom[i][j-3];
    }
  }

  if (angleflag && force->angle) {
    double **vatom = force->angle->vatom;
    for (i = 0; i < nbond; i++){
      for (j = 0; j < 6; j++)
        stress[i][j] += vatom[i][j];
      for (j = 6; j < 9; j++)
        stress[i][j] += vatom[i][j-3];
    }
  }

  if (dihedralflag && force->dihedral) {
    double **vatom = force->dihedral->vatom;
    for (i = 0; i < nbond; i++){
      for (j = 0; j < 6; j++)
        stress[i][j] += vatom[i][j];
      for (j = 6; j < 9; j++)
        stress[i][j] += vatom[i][j-3];
    }
  }

  if (improperflag && force->improper) {
    double **vatom = force->improper->vatom;
    for (i = 0; i < nbond; i++){
      for (j = 0; j < 6; j++)
        stress[i][j] += vatom[i][j];
      for (j = 6; j < 9; j++)
        stress[i][j] += vatom[i][j-3];
    }
  }

  if (kspaceflag && force->kspace) {
    double **vatom = force->kspace->vatom;
    for (i = 0; i < nkspace; i++) {
      for (j = 0; j < 6; j++)
        stress[i][j] += vatom[i][j];
      for (j = 6; j < 9; j++)
        stress[i][j] += vatom[i][j-3];
    }
  }
  // add in per-atom contributions from relevant fixes
  // skip if vatom = NULL
  // possible during setup phase if fix has not initialized its vatom yet
  // e.g. fix ave/spatial defined before fix shake,
  //   and fix ave/spatial uses a per-atom stress from this compute as input

  if (fixflag) {
    for (int ifix = 0; ifix < modify->nfix; ifix++)
      if (modify->fix[ifix]->virial_flag) {
        double **vatom = modify->fix[ifix]->vatom;
        if (vatom)
          for (i = 0; i < nlocal; i++) {
            for (j = 0; j < 6; j++)
              stress[i][j] += vatom[i][j];
            for (j = 6; j < 9; j++)
              stress[i][j] += vatom[i][j-3];
          }
      }
  }

  // communicate ghost virials between neighbor procs

  if (force->newton || (force->kspace && force->kspace->tip4pflag))
    comm->reverse_comm_compute(this);

  // zero virial of atoms not in group
  // only do this after comm since ghost contributions must be included

  int *mask = atom->mask;

  for (i = 0; i < nlocal; i++)
    if (!(mask[i] & groupbit)) {
      stress[i][0] = 0.0;
      stress[i][1] = 0.0;
      stress[i][2] = 0.0;
      stress[i][3] = 0.0;
      stress[i][4] = 0.0;
      stress[i][5] = 0.0;
      stress[i][6] = 0.0;
      stress[i][7] = 0.0;
      stress[i][8] = 0.0;
    }

  // include kinetic energy term for each atom in group
  // mvv2e converts mv^2 to energy

  if (keflag) {
    // double **v = atom->v;
    // double *mass = atom->mass;
    // double *rmass = atom->rmass;
    // int *type = atom->type;
    // double mvv2e = force->mvv2e;

    // double *radius = atom->radius;
    // double **omega = atom->omega;
    // double inertia = 0.4;
    // double pfactor = 4.0/3.0;

    // if (rmass) {
    //   for (i = 0; i < nlocal; i++)
    //     if (mask[i] & groupbit) {
    //       onemass = mvv2e * pfactor * inertia * rmass[i] * radius[i] * radius[i];
    //       stress[i][0] += onemass*v[i][0]*omega[i][0];
    //       stress[i][1] += onemass*v[i][1]*omega[i][1];
    //       stress[i][2] += onemass*v[i][2]*omega[i][2];
    //       stress[i][3] += onemass*v[i][0]*omega[i][1];
    //       stress[i][4] += onemass*v[i][0]*omega[i][2];
    //       stress[i][5] += onemass*v[i][1]*omega[i][2];
    //       stress[i][6] += onemass*v[i][1]*omega[i][0];
    //       stress[i][7] += onemass*v[i][2]*omega[i][0];
    //       stress[i][8] += onemass*v[i][2]*omega[i][1];
    //     }

    // } else {
    //   for (i = 0; i < nlocal; i++)
    //     if (mask[i] & groupbit) {
    //       onemass = mvv2e * pfactor * inertia * mass[type[i]] * radius[i] * radius[i];
    //       stress[i][0] += onemass*v[i][0]*omega[i][0];
    //       stress[i][1] += onemass*v[i][1]*omega[i][1];
    //       stress[i][2] += onemass*v[i][2]*omega[i][2];
    //       stress[i][3] += onemass*v[i][0]*omega[i][1];
    //       stress[i][4] += onemass*v[i][0]*omega[i][2];
    //       stress[i][5] += onemass*v[i][1]*omega[i][2];
    //       stress[i][6] += onemass*v[i][1]*omega[i][0];
    //       stress[i][7] += onemass*v[i][2]*omega[i][0];
    //       stress[i][8] += onemass*v[i][2]*omega[i][1];
    //     }
    // }
  }

  // Convert to cylindrical coordinates
  if (cylflag){
      double **x = atom->x;
      for (i = 0; i < nlocal; i++){
          double theta = std::atan2(x[i][2], x[i][1]);
          double rotation[3][3];
          rotation[0][0] = 1.0;
          rotation[0][1] = 0.0;
          rotation[0][2] = 0.0;

          rotation[1][0] = 0.0;
          rotation[1][1] = std::cos(theta);
          rotation[1][2] = -std::sin(theta);

          rotation[2][0] = 0.0;
          rotation[2][1] = std::sin(theta);
          rotation[2][2] = std::cos(theta);

          double stress_mat[3][3];
          stress_mat[0][0] = stress[i][0];
          stress_mat[0][1] = stress[i][3];
          stress_mat[0][2] = stress[i][4];

          stress_mat[1][0] = stress[i][6];
          stress_mat[1][1] = stress[i][1];
          stress_mat[1][2] = stress[i][5];

          stress_mat[2][0] = stress[i][7];
          stress_mat[2][1] = stress[i][8];
          stress_mat[2][2] = stress[i][2];

          // Initializing temporary mult to 0.
          double mult[3][3];
          for(unsigned p = 0; p < 3; ++p)
              for(unsigned q = 0; q < 3; ++q)
              {
                  mult[p][q]=0.0;
              }

          // mult = stress * rot
          for(unsigned p = 0; p < 3; ++p)
              for(unsigned q = 0; q < 3; ++q)
                  for(unsigned o = 0; o < 3; ++o)
                  {
                      mult[p][q] += stress_mat[p][o] * rotation[o][q];
                  }

          // Initializing new stress matrix to 0.
          double new_stress[3][3];
          for(unsigned p = 0; p < 3; ++p)
              for(unsigned q = 0; q < 3; ++q)
              {
                  new_stress[p][q]=0.0;
              }

          // new_stress = rot.transpose() * mult
          for(unsigned p = 0; p < 3; ++p)
              for(unsigned q = 0; q < 3; ++q)
                  for(unsigned o = 0; o < 3; ++o)
                  {
                      new_stress[p][q] += rotation[o][p] * mult[o][q];
                  }

          // Cylindrical stress
          stress[i][0] = new_stress[0][0];
          stress[i][1] = new_stress[1][1];
          stress[i][2] = new_stress[2][2];
          stress[i][3] = new_stress[0][1];
          stress[i][4] = new_stress[0][2];
          stress[i][5] = new_stress[1][2];
          stress[i][6] = new_stress[1][0];
          stress[i][7] = new_stress[2][0];
          stress[i][8] = new_stress[2][1];
      }
  }

  // convert to stress*volume units = -pressure*volume

  double nktv2p = -force->nktv2p;
  for (i = 0; i < nlocal; i++)
    if (mask[i] & groupbit) {
      stress[i][0] *= nktv2p;
      stress[i][1] *= nktv2p;
      stress[i][2] *= nktv2p;
      stress[i][3] *= nktv2p;
      stress[i][4] *= nktv2p;
      stress[i][5] *= nktv2p;
      stress[i][6] *= nktv2p;
      stress[i][7] *= nktv2p;
      stress[i][8] *= nktv2p;
    }
}

/* ---------------------------------------------------------------------- */

int ComputeCoupleStressAtom::pack_reverse_comm(int n, int first, double *buf)
{
  int i,m,last;

  m = 0;
  last = first + n;
  for (i = first; i < last; i++) {
    buf[m++] = stress[i][0];
    buf[m++] = stress[i][1];
    buf[m++] = stress[i][2];
    buf[m++] = stress[i][3];
    buf[m++] = stress[i][4];
    buf[m++] = stress[i][5];
    buf[m++] = stress[i][6];
    buf[m++] = stress[i][7];
    buf[m++] = stress[i][8];
  }
  return 9;
}

/* ---------------------------------------------------------------------- */

void ComputeCoupleStressAtom::unpack_reverse_comm(int n, int *list, double *buf)
{
  int i,j,m;

  m = 0;
  for (i = 0; i < n; i++) {
    j = list[i];
    stress[j][0] += buf[m++];
    stress[j][1] += buf[m++];
    stress[j][2] += buf[m++];
    stress[j][3] += buf[m++];
    stress[j][4] += buf[m++];
    stress[j][5] += buf[m++];
    stress[j][6] += buf[m++];
    stress[j][7] += buf[m++];
    stress[j][8] += buf[m++];
  }
}

/* ----------------------------------------------------------------------
   memory usage of local atom-based array
------------------------------------------------------------------------- */

double ComputeCoupleStressAtom::memory_usage()
{
  double bytes = nmax*9 * sizeof(double);
  return bytes;
}

/* ---------------------------------------------------------------------- */

int ComputeCoupleStressAtom::pack_comm(int n, int *list, double *buf,
                             int pbc_flag, int *pbc)
{
    int i,j;
    //we dont need to account for pbc here
    int m = 0;
    for (i = 0; i < n; i++) {
      j = list[i];
      buf[m++] = stress[j][0];
      buf[m++] = stress[j][1];
      buf[m++] = stress[j][2];
      buf[m++] = stress[j][3];
      buf[m++] = stress[j][4];
      buf[m++] = stress[j][5];
      buf[m++] = stress[j][6];
      buf[m++] = stress[j][7];
      buf[m++] = stress[j][8];
    }
    return 9;
}

/* ---------------------------------------------------------------------- */

void ComputeCoupleStressAtom::unpack_comm(int n, int first, double *buf)
{
  int i,m,last;
  m = 0;
  last = first + n;
  for (i = first; i < last; i++) {
      stress[i][0] = buf[m++];
      stress[i][1] = buf[m++];
      stress[i][2] = buf[m++];
      stress[i][3] = buf[m++];
      stress[i][4] = buf[m++];
      stress[i][5] = buf[m++];
      stress[i][6] = buf[m++];
      stress[i][7] = buf[m++];
      stress[i][8] = buf[m++];
  }

}
