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
#include "compute_strainenergy_atom.h"
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

ComputeStrainEnergyAtom::ComputeStrainEnergyAtom(LAMMPS *lmp, int &iarg, int narg, char **arg) :
  Compute(lmp, iarg, narg, arg)
{
  if (narg < iarg) error->all(FLERR,"Illegal compute strainenergy/atom command");

  peratom_flag = 1;
  size_peratom_cols = 1;
  pressatomflag = 1;
  timeflag = 1;
  comm_forward = 1;
  comm_reverse = 1;

  if (narg == iarg) {
    pairflag = 1;
  } else {
    pairflag = 0;
    while (iarg < narg) {
      if (strcmp(arg[iarg],"pair") == 0) pairflag = 1;
      else if (strcmp(arg[iarg],"virial") == 0) {
        pairflag = 1;
      } else error->all(FLERR,"Illegal compute strainenergy/atom command");
      iarg++;
    }
  }

  nmax = 0;
  strainenergy = NULL;
}

/* ---------------------------------------------------------------------- */

ComputeStrainEnergyAtom::~ComputeStrainEnergyAtom()
{
  memory->destroy(strainenergy);
}

/* ---------------------------------------------------------------------- */

void ComputeStrainEnergyAtom::compute_peratom()
{
  int i,j;
  double onemass;

  invoked_peratom = update->ntimestep;
  if (update->vflag_atom != invoked_peratom)
    error->all(FLERR,"Per-atom strainenergy was not tallied on needed timestep");

  // grow local stress array if necessary
  // needs to be atom->nmax in length

  if (atom->nmax > nmax) {
    memory->destroy(strainenergy);
    nmax = atom->nmax;
    memory->create(strainenergy,nmax,1,"strainenergy/atom:strainenergy");
    array_atom = strainenergy;
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

  // clear local strainenergy array

  for (i = 0; i < ntotal; i++)
    strainenergy[i][0] = 0.0;

  // add in per-atom contributions from each force

  if (pairflag && force->pair) {
    double *seatom = force->pair->seatom;
    for (i = 0; i < npair; i++)
      strainenergy[i][0] += seatom[i];
  }

  // communicate ghost virials between neighbor procs

  if (force->newton || (force->kspace && force->kspace->tip4pflag))
    comm->reverse_comm_compute(this);

  // zero virial of atoms not in group
  // only do this after comm since ghost contributions must be included

  int *mask = atom->mask;

  for (i = 0; i < nlocal; i++)
    if (!(mask[i] & groupbit)) {
      strainenergy[i][0] = 0.0;
    }

}

/* ---------------------------------------------------------------------- */

int ComputeStrainEnergyAtom::pack_reverse_comm(int n, int first, double *buf)
{
  int i,m,last;

  m = 0;
  last = first + n;
  for (i = first; i < last; i++) {
    buf[m++] = strainenergy[i][0];
  }
  return 1;
}

/* ---------------------------------------------------------------------- */

void ComputeStrainEnergyAtom::unpack_reverse_comm(int n, int *list, double *buf)
{
  int i,j,m;

  m = 0;
  for (i = 0; i < n; i++) {
    j = list[i];
    strainenergy[j][0] += buf[m++];
  }
}

/* ----------------------------------------------------------------------
   memory usage of local atom-based array
------------------------------------------------------------------------- */

double ComputeStrainEnergyAtom::memory_usage()
{
  double bytes = nmax*1 * sizeof(double);
  return bytes;
}

/* ---------------------------------------------------------------------- */

int ComputeStrainEnergyAtom::pack_comm(int n, int *list, double *buf,
                             int pbc_flag, int *pbc)
{
    int i,j;
    //we dont need to account for pbc here
    int m = 0;
    for (i = 0; i < n; i++) {
      j = list[i];
      buf[m++] = strainenergy[j][0];
    }
    return 1;
}

/* ---------------------------------------------------------------------- */

void ComputeStrainEnergyAtom::unpack_comm(int n, int first, double *buf)
{
  int i,m,last;
  m = 0;
  last = first + n;
  for (i = first; i < last; i++) {
      strainenergy[i][0] = buf[m++];
  }
}
