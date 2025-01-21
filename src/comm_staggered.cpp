// clang-format off
/* ----------------------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   https://www.lammps.org/, Sandia National Laboratories
   LAMMPS development team: developers@lammps.org

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

#include "comm_staggered.h"

#include "atom.h"
#include "atom_vec.h"
#include "bond.h"
#include "compute.h"
#include "domain.h"
#include "dump.h"
#include "error.h"
#include "fix.h"
#include "memory.h"
#include "neighbor.h"
#include "pair.h"

#include <cmath>
#include <cstring>

using namespace LAMMPS_NS;

static constexpr double BUFFACTOR = 1.5;
static constexpr int BUFMIN = 1024;
static constexpr double EPSILON = 1.0e-6;
static constexpr int DELTA_PROCS = 16;


/* ---------------------------------------------------------------------- */
//IMPORTANT: we *MUST* pass "*oldcomm" to the Comm initializer here, as
//           the code below *requires* that the (implicit) copy constructor
//           for Comm is run and thus creating a shallow copy of "oldcomm".
//           The call to Comm::copy_arrays() then converts the shallow copy
//           into a deep copy of the class with the new layout.
//           @todo search 'comm->layout' and adapt for LAYOUT_STAGGERED
//           @bug search 'comm->layout' and adapt for LAYOUT_STAGGERED

CommStaggered::CommStaggered(LAMMPS *lmp, Comm *oldcomm, char *arg) : CommTiled(lmp,oldcomm)
{
#ifdef DEBUG_COMM_STAGGERED
  char logfile[50];
  sprintf(logfile, "scratch/comm_staggered_%i.log", me);
  fp = fopen(logfile, "w");
  fprintf(fp, "new log proc %i\n", me);
  fflush(fp);
#endif

  if (oldcomm->layout == Comm::LAYOUT_TILED)
    error->all(FLERR,"Cannot change to comm_style staggered from tiled layout");

  style = Comm::STAGGERED;
  if (strlen(arg) != 3) error->all(FLERR, "expected 3 chars instead of {}", strlen(arg));
  for (int i=0; i<3; i++) {
    if (arg[i] == 'x') staggered2spatial[i] = 0;
    else if (arg[i] == 'y') staggered2spatial[i] = 1;
    else if (arg[i] == 'z') staggered2spatial[i] = 2;
    else error->all(FLERR, "expected x,y or z instead of {} in staggered config", arg[i]);
    spatial2staggered[staggered2spatial[i]] = i;
  }

  ///> @todo adapt copy_arrays
  init_buffers_staggered();

#ifdef DEBUG_COMM_STAGGERED
  fprintf(fp, "procgrid %i %i %i\nmyloc %i %i %i\ngrid2proc[myloc] %i\n", procgrid[0], procgrid[1], procgrid[2], myloc[0], myloc[1], myloc[2], grid2proc[myloc[0]][myloc[1]][myloc[2]]);
  fflush(fp);
#endif
}

/**
  * Free allocated memory specific to staggered.
  */

CommStaggered::~CommStaggered()
{
  if (staggered_proc2grid) memory->destroy(staggered_proc2grid);
  if (staggered_grid2proc) memory->destroy(staggered_grid2proc);
  if (layer_splits) memory->destroy(layer_splits);
  if (row_splits) memory->destroy(row_splits);
  if (cell_splits) memory->destroy(cell_splits);
#ifdef DEBUG_COMM_STAGGERED
  fclose(fp);
#endif
}

/**
  * initialise comm buffers and other data structs local to CommStaggered but not to CommTiled
  */

void CommStaggered::init_buffers_staggered()
{
#ifdef DEBUG_COMM_STAGGERED
  fprintf(fp, "Hello from init_buffers_staggered\n");
  fflush(fp);
#endif

  // allocate memory and store constant quantities
  for (int i=0; i<3; i++) {
    staggered_procgrid[spatial2staggered[i]] = procgrid[i];
    staggered_myloc[spatial2staggered[i]] = myloc[i];
  }
  n_layers = staggered_procgrid[0];
  n_rows = staggered_procgrid[1];
  n_cells = staggered_procgrid[2];

  memory->create(layer_splits, n_layers+1, "comm:layer_splits");
  memory->create(row_splits, n_layers, n_rows+1, "comm:row_splits");
  memory->create(cell_splits, n_layers, n_rows, n_cells+1, "comm:cell_splits");
  memory->create(staggered_grid2proc, n_layers, n_rows, n_cells, "comm:staggered_grid2proc");
  memory->create(staggered_proc2grid, nprocs, 3, "comm:staggered_proc2grid");

  int tmploc[3];
  for (tmploc[0]=0; tmploc[0]<n_layers; tmploc[0]++) {
    for (tmploc[1]=0; tmploc[1]<n_rows;   tmploc[1]++) {
      for (tmploc[2]=0; tmploc[2]<n_cells;  tmploc[2]++) {
        staggered_grid2proc[tmploc[0]][tmploc[1]][tmploc[2]] = grid2proc[tmploc[spatial2staggered[0]]]
                                                   [tmploc[spatial2staggered[1]]]
                                                   [tmploc[spatial2staggered[2]]];
      }
    }
  }
  int gatherbuffer[nprocs*3];
  gatherbuffer[me*3+0] = staggered_myloc[0];
  gatherbuffer[me*3+1] = staggered_myloc[1];
  gatherbuffer[me*3+2] = staggered_myloc[2];
  MPI_Allgather(MPI_IN_PLACE,3,MPI_INT,gatherbuffer,3,MPI_INT,world);
  for (int i=0; i<nprocs; i++) {
    staggered_proc2grid[i][0] = gatherbuffer[i*3+0];
    staggered_proc2grid[i][1] = gatherbuffer[i*3+1];
    staggered_proc2grid[i][2] = gatherbuffer[i*3+2];
  }
}

/**
  * setup spatial-decomposition communication patterns
  * function of neighbor cutoff(s) & cutghostuser & current box size and tiling
  */

void CommStaggered::setup()
{
#ifdef DEBUG_COMM_STAGGERED
  fprintf(fp, "Hello from CommStaggered::setup\n");
  fprintf(fp, "mysplit x %f %f y %f %f z %f %f\n", mysplit[0][0], mysplit[0][1], mysplit[1][0], mysplit[1][1], mysplit[2][0], mysplit[2][1]);
  fflush(fp);
#endif

  int i,j,n;

  // domain properties used in setup method and methods it calls

  dimension = domain->dimension;
  int *periodicity = domain->periodicity;
  int ntypes = atom->ntypes;

  if (triclinic == 0) {
    prd = domain->prd;
    boxlo = domain->boxlo;
    boxhi = domain->boxhi;
    sublo = domain->sublo;
    subhi = domain->subhi;
  } else {
    prd = domain->prd_lamda;
    boxlo = domain->boxlo_lamda;
    boxhi = domain->boxhi_lamda;
    sublo = domain->sublo_lamda;
    subhi = domain->subhi_lamda;
  }

  // set function pointers

  if (layout == Comm::LAYOUT_STAGGERED) {
    box_drop = &CommStaggered::box_drop_staggered;
    box_other = &CommStaggered::box_other_staggered;
    box_touch = &CommStaggered::box_touch_staggered;
    point_drop = &CommStaggered::point_drop_staggered;
  } else {
    // set brick for this class
    box_drop = &CommStaggered::box_drop_brick;
    box_other = &CommStaggered::box_other_brick;
    box_touch = &CommStaggered::box_touch_brick;
    point_drop = &CommStaggered::point_drop_brick;
    // set brick for parent class
    CommTiled::box_drop = &CommStaggered::box_drop_brick;
    CommTiled::box_other = &CommStaggered::box_other_brick;
    CommTiled::box_touch = &CommStaggered::box_touch_brick;
    CommTiled::point_drop = &CommStaggered::point_drop_brick;
  }

  // if RCB decomp exists and just changed, gather needed global RCB info

#ifdef DEBUG_COMM_STAGGERED
  fprintf(fp, "call coord2proc_setup? comm->layout %i layout %i (uniform %i, nonuniform %i, tiled %i, staggered %i\n", comm->layout, layout, Comm::LAYOUT_UNIFORM, Comm::LAYOUT_NONUNIFORM, Comm::LAYOUT_TILED , Comm::LAYOUT_STAGGERED);
  fflush(fp);
#endif
  if (layout == Comm::LAYOUT_STAGGERED) coord2proc_setup();

  // set cutoff for comm forward and comm reverse
  // check that cutoff < any periodic box length

  if (mode == Comm::MULTI) {
    double **cutcollectionsq = neighbor->cutcollectionsq;

    // build collection array for atom exchange
    neighbor->build_collection(0);

    // If using multi/reduce, communicate particles a distance equal
    // to the max cutoff with equally sized or smaller collections
    // If not, communicate the maximum cutoff of the entire collection
    for (i = 0; i < ncollections; i++) {
      if (cutusermulti) {
        cutghostmulti[i][0] = cutusermulti[i];
        cutghostmulti[i][1] = cutusermulti[i];
        cutghostmulti[i][2] = cutusermulti[i];
      } else {
        cutghostmulti[i][0] = 0.0;
        cutghostmulti[i][1] = 0.0;
        cutghostmulti[i][2] = 0.0;
      }

      for (j = 0; j < ncollections; j++){
        if (multi_reduce && (cutcollectionsq[j][j] > cutcollectionsq[i][i])) continue;
        cutghostmulti[i][0] = MAX(cutghostmulti[i][0],sqrt(cutcollectionsq[i][j]));
        cutghostmulti[i][1] = MAX(cutghostmulti[i][1],sqrt(cutcollectionsq[i][j]));
        cutghostmulti[i][2] = MAX(cutghostmulti[i][2],sqrt(cutcollectionsq[i][j]));
      }
    }
  }

  if (mode == Comm::MULTIOLD) {
    double *cuttype = neighbor->cuttype;
    for (i = 1; i <= ntypes; i++) {
      double tmp = 0.0;
      if (cutusermultiold) tmp = cutusermultiold[i];
      cutghostmultiold[i][0] = MAX(tmp,cuttype[i]);
      cutghostmultiold[i][1] = MAX(tmp,cuttype[i]);
      cutghostmultiold[i][2] = MAX(tmp,cuttype[i]);
    }
  }

  double cut = get_comm_cutoff();
  if ((cut == 0.0) && (me == 0))
    error->warning(FLERR,"Communication cutoff is 0.0. No ghost atoms "
                   "will be generated. Atoms may get lost.");

  if (triclinic == 0) cutghost[0] = cutghost[1] = cutghost[2] = cut;
  else {
    double *h_inv = domain->h_inv;
    double length0,length1,length2;
    length0 = sqrt(h_inv[0]*h_inv[0] + h_inv[5]*h_inv[5] + h_inv[4]*h_inv[4]);
    cutghost[0] = cut * length0;
    length1 = sqrt(h_inv[1]*h_inv[1] + h_inv[3]*h_inv[3]);
    cutghost[1] = cut * length1;
    length2 = h_inv[2];
    cutghost[2] = cut * length2;
    if (mode == Comm::MULTI) {
      for (i = 0; i < ncollections; i++) {
        cutghostmulti[i][0] *= length0;
        cutghostmulti[i][1] *= length1;
        cutghostmulti[i][2] *= length2;
      }
    }

    if (mode == Comm::MULTIOLD) {
      for (i = 1; i <= ntypes; i++) {
        cutghostmultiold[i][0] *= length0;
        cutghostmultiold[i][1] *= length1;
        cutghostmultiold[i][2] *= length2;
      }
    }
  }

  if ((periodicity[0] && cutghost[0] > prd[0]) ||
      (periodicity[1] && cutghost[1] > prd[1]) ||
      (dimension == 3 && periodicity[2] && cutghost[2] > prd[2]))
    error->all(FLERR,"Communication cutoff for comm_style tiled "
               "cannot exceed periodic box length");

  // if cut = 0.0, set to epsilon to induce nearest neighbor comm
  // this is b/c sendproc is used below to infer touching exchange procs
  // exchange procs will be empty (leading to lost atoms) if sendproc = 0
  // will reset sendproc/etc to 0 after exchange is setup, down below

  int cutzero = 0;
  if (cut == 0.0) {
    cutzero = 1;
    cut = MIN(prd[0],prd[1]);
    if (dimension == 3) cut = MIN(cut,prd[2]);
    cut *= EPSILON*EPSILON;
    cutghost[0] = cutghost[1] = cutghost[2] = cut;
  }

  // setup forward/reverse communication
  // loop over 6 swap directions
  // determine which procs I will send to and receive from in each swap
  // done by intersecting ghost box with all proc sub-boxes it overlaps
  // sets nsendproc, nrecvproc, sendproc, recvproc
  // sets sendother, recvother, sendself, pbc_flag, pbc, sendbox
  // resets nprocmax

  int noverlap1,indexme;
  double lo1[3],hi1[3],lo2[3],hi2[3];
  int one,two;

  int iswap = 0;
#ifdef DEBUG_COMM_STAGGERED
  fprintf(fp, "mysplit x %f %f y %f %f z %f %f\n", mysplit[0][0], mysplit[0][1], mysplit[1][0], mysplit[1][1], mysplit[2][0], mysplit[2][1]);
  fprintf(fp, "boxlo + mysplit*prd x %f %f y %f %f z %f %f\n",
      boxlo[0]+domain->prd[0]*mysplit[0][0], boxlo[0]+domain->prd[0]*mysplit[0][1],
      boxlo[1]+domain->prd[1]*mysplit[1][0], boxlo[1]+domain->prd[1]*mysplit[1][1],
      boxlo[2]+domain->prd[2]*mysplit[2][0], boxlo[2]+domain->prd[2]*mysplit[2][1]);
  fprintf(fp, "sublo %f %f %f subhi %f %f %f\n", sublo[0], sublo[1], sublo[2], subhi[0], subhi[1], subhi[2]);
  fflush(fp);
#endif
  for (int idim = 0; idim < dimension; idim++) {
    for (int idir = 0; idir < 2; idir++) {

      // one = first ghost box in same periodic image
      // two = second ghost box wrapped across periodic boundary
      // either may not exist

      one = 1;
      lo1[0] = sublo[0]; lo1[1] = sublo[1]; lo1[2] = sublo[2];
      hi1[0] = subhi[0]; hi1[1] = subhi[1]; hi1[2] = subhi[2];
      if (idir == 0) {
        lo1[idim] = sublo[idim] - cutghost[idim];
        hi1[idim] = sublo[idim];
      } else {
        lo1[idim] = subhi[idim];
        hi1[idim] = subhi[idim] + cutghost[idim];
      }

      two = 0;
      if (idir == 0 && periodicity[idim] && lo1[idim] < boxlo[idim]) two = 1;
      if (idir == 1 && periodicity[idim] && hi1[idim] > boxhi[idim]) two = 1;

      if (two) {
        lo2[0] = sublo[0]; lo2[1] = sublo[1]; lo2[2] = sublo[2];
        hi2[0] = subhi[0]; hi2[1] = subhi[1]; hi2[2] = subhi[2];
        if (idir == 0) {
          lo2[idim] = lo1[idim] + prd[idim];
          hi2[idim] = boxhi[idim];
          if (sublo[idim] == boxlo[idim]) one = 0;
          //else if (layout == Comm::LAYOUT_STAGGERED && staggered_myloc[spatial2staggered[idim]] == 0) one = 0; ///>@todo use? rather workaround than robust fix
        } else {
          lo2[idim] = boxlo[idim];
          hi2[idim] = hi1[idim] - prd[idim];
          if (subhi[idim] == boxhi[idim]) one = 0;
          //else if (layout == Comm::LAYOUT_STAGGERED && staggered_myloc[spatial2staggered[idim]] == staggered_procgrid[spatial2staggered[idim]] -1) one = 0; ///>@todo use ? more workaround than robust fix
        }
      }

      if (one) {
        if (idir == 0) lo1[idim] = MAX(lo1[idim],boxlo[idim]);
        else hi1[idim] = MIN(hi1[idim],boxhi[idim]);
        if (lo1[idim] == hi1[idim]) one = 0; ///> @fixme dim 0 dir 1 box_drop one 1; Hello from box_drop_staggered lo 1.000000 0.205773 0.169099 hi 1.000000 0.404332 0.338174
#ifdef DEBUG_COMM_STAGGERED
        fprintf(fp, "one exists idir %i idim %i lo1[idim] %.13g hi1[idim] %.13g boxlo[idim] %.13g boxhi[idim] %.13g width %.13g hi1>lo1 %i hi1<lo1 %i hi1==lo1 %i\n", idir, idim, lo1[idim], hi1[idim], boxlo[idim], boxhi[idim], hi1[idim]-lo1[idim], hi1[idim]>lo1[idim], hi1[idim]<lo1[idim], hi1[idim]==lo1[idim]);
        fflush(fp);
#endif
      }
      if (one && idir == 0 && layout == Comm::LAYOUT_STAGGERED && staggered_myloc[spatial2staggered[idim]] == 0) error->one(FLERR, "one set for communication in direction 0, but there is no processor");
      if (one && idir == 1 && layout == Comm::LAYOUT_STAGGERED && staggered_myloc[spatial2staggered[idim]] == staggered_procgrid[spatial2staggered[idim]] -1) error->one(FLERR, "one set for communication in direction 1, but there is no processor");

      // noverlap = # of overlaps of box1/2 with procs via box_drop()
      // overlap = list of overlapping procs
      // if overlap with self, indexme = index of me in list

      indexme = -1;
      noverlap = 0;
#ifdef DEBUG_COMM_STAGGERED
      fprintf(fp, "start calculating neighbours in dimension %i and direction %i\n", idim, idir);
      fflush(fp);
#endif
      //if (idim == 2) {
      //  printf("[%i]: clear direction %i (myloc %i %i %i) cutghost %f\n", me, idir, staggered_myloc[0], staggered_myloc[1], staggered_myloc[2], cutghost[idim]);
      //  neighbours_z->clear(idir);
      //}
#ifdef DEBUG_COMM_STAGGERED
      fprintf(fp, "box_drop one %i\n", one);
      fflush(fp);
#endif
      if (one) (this->*box_drop)(idim,lo1,hi1,indexme);
      noverlap1 = noverlap;
#ifdef DEBUG_COMM_STAGGERED
      fprintf(fp, "box_drop two %i\n", two);
      fflush(fp);
#endif
      if (two) (this->*box_drop)(idim,lo2,hi2,indexme);
#ifdef DEBUG_COMM_STAGGERED
      fprintf(fp, "calculated %i neighbours in dimension %i and direction %i\n", noverlap, idim, idir);
      for (int i=0; i<noverlap; i++) fprintf(fp, "overlap[%i] = %i\n", i, overlap[i]);
      fflush(fp);
#endif

      // if self is in overlap list, move it to end of list

      if (indexme >= 0) {
        int tmp = overlap[noverlap-1];
        overlap[noverlap-1] = overlap[indexme];
        overlap[indexme] = tmp;
      }

      // reallocate 2nd dimensions of all send/recv arrays, based on noverlap
      // # of sends of this swap = # of recvs of iswap +/- 1

      if (noverlap > nprocmax[iswap]) {
        int oldmax = nprocmax[iswap];
        while (nprocmax[iswap] < noverlap) nprocmax[iswap] += DELTA_PROCS;
        grow_swap_send(iswap,nprocmax[iswap],oldmax);
        if (idir == 0) grow_swap_recv(iswap+1,nprocmax[iswap]);
        else grow_swap_recv(iswap-1,nprocmax[iswap]);
      }

      // overlap how has list of noverlap procs
      // includes PBC effects

      if (noverlap && overlap[noverlap-1] == me) sendself[iswap] = 1;
      else sendself[iswap] = 0;
      if (noverlap && noverlap-sendself[iswap]) sendother[iswap] = 1;
      else sendother[iswap] = 0;

      nsendproc[iswap] = noverlap;
      for (i = 0; i < noverlap; i++) sendproc[iswap][i] = overlap[i];

      if (idir == 0) {
        recvother[iswap+1] = sendother[iswap];
        nrecvproc[iswap+1] = noverlap;
        for (i = 0; i < noverlap; i++) recvproc[iswap+1][i] = overlap[i];
      } else {
        recvother[iswap-1] = sendother[iswap];
        nrecvproc[iswap-1] = noverlap;
        for (i = 0; i < noverlap; i++) recvproc[iswap-1][i] = overlap[i];
      }

      // compute sendbox for each of my sends
      // obox = intersection of ghostbox with other proc's sub-domain
      // sbox = what I need to send to other proc
      //      = sublo to MIN(sublo+cut,subhi) in idim, for idir = 0
      //      = MIN(subhi-cut,sublo) to subhi in idim, for idir = 1
      //      = obox in other 2 dims
      // if sbox touches other proc's sub-box boundaries in lower dims,
      //   extend sbox in those lower dims to include ghost atoms
      // single mode and multi mode

      double oboxlo[3],oboxhi[3],sbox[6],sbox_multi[6],sbox_multiold[6];

      if (mode == Comm::SINGLE) {
        for (i = 0; i < noverlap; i++) {
          pbc_flag[iswap][i] = 0;
          pbc[iswap][i][0] = pbc[iswap][i][1] = pbc[iswap][i][2] =
            pbc[iswap][i][3] = pbc[iswap][i][4] = pbc[iswap][i][5] = 0;

          (this->*box_other)(idim,idir,overlap[i],oboxlo,oboxhi);

          if (i < noverlap1) {
            sbox[0] = MAX(oboxlo[0],lo1[0]);
            sbox[1] = MAX(oboxlo[1],lo1[1]);
            sbox[2] = MAX(oboxlo[2],lo1[2]);
            sbox[3] = MIN(oboxhi[0],hi1[0]);
            sbox[4] = MIN(oboxhi[1],hi1[1]);
            sbox[5] = MIN(oboxhi[2],hi1[2]);
          } else {
            pbc_flag[iswap][i] = 1;
            if (idir == 0) pbc[iswap][i][idim] = 1;
            else pbc[iswap][i][idim] = -1;
            if (triclinic) {
              if (idim == 1) pbc[iswap][i][5] = pbc[iswap][i][idim];
              if (idim == 2) pbc[iswap][i][4] = pbc[iswap][i][3] = pbc[iswap][i][idim];
            }
            sbox[0] = MAX(oboxlo[0],lo2[0]);
            sbox[1] = MAX(oboxlo[1],lo2[1]);
            sbox[2] = MAX(oboxlo[2],lo2[2]);
            sbox[3] = MIN(oboxhi[0],hi2[0]);
            sbox[4] = MIN(oboxhi[1],hi2[1]);
            sbox[5] = MIN(oboxhi[2],hi2[2]);
          }

          if (idir == 0) {
            sbox[idim] = sublo[idim];
            if (i < noverlap1)
              sbox[3+idim] = MIN(sbox[3+idim]+cutghost[idim],subhi[idim]);
            else
              sbox[3+idim] = MIN(sbox[3+idim]-prd[idim]+cutghost[idim],subhi[idim]);
          } else {
            if (i < noverlap1) sbox[idim] = MAX(sbox[idim]-cutghost[idim],sublo[idim]);
            else sbox[idim] = MAX(sbox[idim]+prd[idim]-cutghost[idim],sublo[idim]);
            sbox[3+idim] = subhi[idim];
          }

          if (idim >= 1) {
            if (sbox[0] == oboxlo[0]) sbox[0] -= cutghost[0];
            if (sbox[3] == oboxhi[0]) sbox[3] += cutghost[0];
          }
          if (idim == 2) {
            if (sbox[1] == oboxlo[1]) sbox[1] -= cutghost[1];
            if (sbox[4] == oboxhi[1]) sbox[4] += cutghost[1];
          }

          memcpy(sendbox[iswap][i],sbox,6*sizeof(double));
        }
      }

      if (mode == Comm::MULTI) {
        for (i = 0; i < noverlap; i++) {
          pbc_flag[iswap][i] = 0;
          pbc[iswap][i][0] = pbc[iswap][i][1] = pbc[iswap][i][2] =
            pbc[iswap][i][3] = pbc[iswap][i][4] = pbc[iswap][i][5] = 0;

          (this->*box_other)(idim,idir,overlap[i],oboxlo,oboxhi);

          if (i < noverlap1) {
            sbox[0] = MAX(oboxlo[0],lo1[0]);
            sbox[1] = MAX(oboxlo[1],lo1[1]);
            sbox[2] = MAX(oboxlo[2],lo1[2]);
            sbox[3] = MIN(oboxhi[0],hi1[0]);
            sbox[4] = MIN(oboxhi[1],hi1[1]);
            sbox[5] = MIN(oboxhi[2],hi1[2]);
          } else {
            pbc_flag[iswap][i] = 1;
            if (idir == 0) pbc[iswap][i][idim] = 1;
            else pbc[iswap][i][idim] = -1;
            if (triclinic) {
              if (idim == 1) pbc[iswap][i][5] = pbc[iswap][i][idim];
              if (idim == 2) pbc[iswap][i][4] = pbc[iswap][i][3] = pbc[iswap][i][idim];
            }
            sbox[0] = MAX(oboxlo[0],lo2[0]);
            sbox[1] = MAX(oboxlo[1],lo2[1]);
            sbox[2] = MAX(oboxlo[2],lo2[2]);
            sbox[3] = MIN(oboxhi[0],hi2[0]);
            sbox[4] = MIN(oboxhi[1],hi2[1]);
            sbox[5] = MIN(oboxhi[2],hi2[2]);
          }

          for (int icollection = 0; icollection < ncollections; icollection++) {
            sbox_multi[0] = sbox[0];
            sbox_multi[1] = sbox[1];
            sbox_multi[2] = sbox[2];
            sbox_multi[3] = sbox[3];
            sbox_multi[4] = sbox[4];
            sbox_multi[5] = sbox[5];
            if (idir == 0) {
              sbox_multi[idim] = sublo[idim];
              if (i < noverlap1)
                sbox_multi[3+idim] =
                  MIN(sbox_multi[3+idim]+cutghostmulti[icollection][idim],subhi[idim]);
              else
                sbox_multi[3+idim] =
                  MIN(sbox_multi[3+idim]-prd[idim]+cutghostmulti[icollection][idim],subhi[idim]);
            } else {
              if (i < noverlap1)
                sbox_multi[idim] =
                  MAX(sbox_multi[idim]-cutghostmulti[icollection][idim],sublo[idim]);
              else
                sbox_multi[idim] =
                  MAX(sbox_multi[idim]+prd[idim]-cutghostmulti[icollection][idim],sublo[idim]);
              sbox_multi[3+idim] = subhi[idim];
            }

            if (idim >= 1) {
              if (sbox_multi[0] == oboxlo[0])
                sbox_multi[0] -= cutghostmulti[icollection][idim];
              if (sbox_multi[3] == oboxhi[0])
                sbox_multi[3] += cutghostmulti[icollection][idim];
            }
            if (idim == 2) {
              if (sbox_multi[1] == oboxlo[1])
                sbox_multi[1] -= cutghostmulti[icollection][idim];
              if (sbox_multi[4] == oboxhi[1])
                sbox_multi[4] += cutghostmulti[icollection][idim];
            }

            memcpy(sendbox_multi[iswap][i][icollection],sbox_multi,6*sizeof(double));
          }
        }
      }

      if (mode == Comm::MULTIOLD) {
        for (i = 0; i < noverlap; i++) {
          pbc_flag[iswap][i] = 0;
          pbc[iswap][i][0] = pbc[iswap][i][1] = pbc[iswap][i][2] =
            pbc[iswap][i][3] = pbc[iswap][i][4] = pbc[iswap][i][5] = 0;

          (this->*box_other)(idim,idir,overlap[i],oboxlo,oboxhi);

          if (i < noverlap1) {
            sbox[0] = MAX(oboxlo[0],lo1[0]);
            sbox[1] = MAX(oboxlo[1],lo1[1]);
            sbox[2] = MAX(oboxlo[2],lo1[2]);
            sbox[3] = MIN(oboxhi[0],hi1[0]);
            sbox[4] = MIN(oboxhi[1],hi1[1]);
            sbox[5] = MIN(oboxhi[2],hi1[2]);
          } else {
            pbc_flag[iswap][i] = 1;
            if (idir == 0) pbc[iswap][i][idim] = 1;
            else pbc[iswap][i][idim] = -1;
            if (triclinic) {
              if (idim == 1) pbc[iswap][i][5] = pbc[iswap][i][idim];
              if (idim == 2) pbc[iswap][i][4] = pbc[iswap][i][3] = pbc[iswap][i][idim];
            }
            sbox[0] = MAX(oboxlo[0],lo2[0]);
            sbox[1] = MAX(oboxlo[1],lo2[1]);
            sbox[2] = MAX(oboxlo[2],lo2[2]);
            sbox[3] = MIN(oboxhi[0],hi2[0]);
            sbox[4] = MIN(oboxhi[1],hi2[1]);
            sbox[5] = MIN(oboxhi[2],hi2[2]);
          }

          for (int itype = 1; itype <= atom->ntypes; itype++) {
            sbox_multiold[0] = sbox[0];
            sbox_multiold[1] = sbox[1];
            sbox_multiold[2] = sbox[2];
            sbox_multiold[3] = sbox[3];
            sbox_multiold[4] = sbox[4];
            sbox_multiold[5] = sbox[5];
            if (idir == 0) {
              sbox_multiold[idim] = sublo[idim];
              if (i < noverlap1)
                sbox_multiold[3+idim] =
                  MIN(sbox_multiold[3+idim]+cutghostmultiold[itype][idim],subhi[idim]);
              else
                sbox_multiold[3+idim] =
                  MIN(sbox_multiold[3+idim]-prd[idim]+cutghostmultiold[itype][idim],subhi[idim]);
            } else {
              if (i < noverlap1)
                sbox_multiold[idim] =
                  MAX(sbox_multiold[idim]-cutghostmultiold[itype][idim],sublo[idim]);
              else
                sbox_multiold[idim] =
                  MAX(sbox_multiold[idim]+prd[idim]-cutghostmultiold[itype][idim],sublo[idim]);
              sbox_multiold[3+idim] = subhi[idim];
            }

            if (idim >= 1) {
              if (sbox_multiold[0] == oboxlo[0])
                sbox_multiold[0] -= cutghostmultiold[itype][idim];
              if (sbox_multiold[3] == oboxhi[0])
                sbox_multiold[3] += cutghostmultiold[itype][idim];
            }
            if (idim == 2) {
              if (sbox_multiold[1] == oboxlo[1])
                sbox_multiold[1] -= cutghostmultiold[itype][idim];
              if (sbox_multiold[4] == oboxhi[1])
                sbox_multiold[4] += cutghostmultiold[itype][idim];
            }

            memcpy(sendbox_multiold[iswap][i][itype],sbox_multiold,6*sizeof(double));
          }
        }
      }

      iswap++;
    }
  }


  // setup exchange communication = subset of forward/reverse comm procs
  // loop over dimensions
  // determine which procs I will exchange with in each dimension
  // subset of procs that touch my proc in forward/reverse comm
  // sets nexchproc & exchproc, resets nexchprocmax

  int proc;

  for (int idim = 0; idim < dimension; idim++) {

    // overlap = list of procs that touch my sub-box in idim
    // proc can appear twice in list if touches in both directions
    // 2nd add-to-list checks to ensure each proc appears exactly once

    noverlap = 0;
    iswap = 2*idim;
    n = nsendproc[iswap];
    for (i = 0; i < n; i++) {
      proc = sendproc[iswap][i];
      if (proc == me) continue;
      if ((this->*box_touch)(proc,idim,0)) {
        if (noverlap == maxoverlap) {
          maxoverlap += DELTA_PROCS;
          memory->grow(overlap,maxoverlap,"comm:overlap");
        }
        overlap[noverlap++] = proc;
      }
    }
    noverlap1 = noverlap;
    iswap = 2*idim+1;
    n = nsendproc[iswap];

    MPI_Barrier(world);

    for (i = 0; i < n; i++) {
      proc = sendproc[iswap][i];
      if (proc == me) continue;
      if ((this->*box_touch)(proc,idim,1)) {
        for (j = 0; j < noverlap1; j++)
          if (overlap[j] == proc) break;
        if (j < noverlap1) continue;
        if (noverlap == maxoverlap) {
          maxoverlap += DELTA_PROCS;
          memory->grow(overlap,maxoverlap,"comm:overlap");
        }
        overlap[noverlap++] = proc;
      }
    }

    MPI_Barrier(world);

    // reallocate exchproc and exchnum if needed based on noverlap

    if (noverlap > nexchprocmax[idim]) {
      while (nexchprocmax[idim] < noverlap) nexchprocmax[idim] += DELTA_PROCS;
      delete [] exchproc[idim];
      exchproc[idim] = new int[nexchprocmax[idim]];
      delete [] exchnum[idim];
      exchnum[idim] = new int[nexchprocmax[idim]];
    }

    nexchproc[idim] = noverlap;
    for (i = 0; i < noverlap; i++) exchproc[idim][i] = overlap[i];
#ifdef DEBUG_COMM_STAGGERED
    fprintf(fp, "neighbours dimension %i\n", idim);
    for (i = 0; i < noverlap; i++) {
      fprintf(fp, "idim %i i %i proc %i\n", idim, i, exchproc[idim][i]);
      if (exchproc[idim][i] < 0 || exchproc[idim][i] >= nprocs) {
        error->one(FLERR, "exchproc value {} unexpected for nprocs {}", exchproc[idim][i], nprocs);
      }
    }
    fflush(fp);
#endif
  }

  // reset sendproc/etc to 0 if cut is really 0.0

  if (cutzero) {
    for (i = 0; i < nswap; i++) {
      nsendproc[i] = nrecvproc[i] =
        sendother[i] = recvother[i] = sendself[i] = 0;
    }
  }

  // reallocate MPI Requests as needed

  int nmax = 0;
  for (i = 0; i < nswap; i++) nmax = MAX(nmax,nprocmax[i]);
  for (i = 0; i < dimension; i++) nmax = MAX(nmax,nexchprocmax[i]);
  if (nmax > maxrequest) {
    maxrequest = nmax;
    delete [] requests;
    requests = new MPI_Request[maxrequest];
  }
}

/**
  * exchange: move atoms to correct processors
  * atoms exchanged with procs that touch sub-box in each of 3 dims
  * send out atoms that have left my box, receive ones entering my box
  * atoms will be lost if not inside a touching proc's box
  *   can happen if atom moves outside of non-periodic boundary
  *   or if atom moves more than one proc away
  * this routine called before every reneighboring
  * for triclinic, atoms must be in lamda coords (0-1) before exchange is called
  */

void CommStaggered::exchange()
{
#ifdef DEBUG_COMM_STAGGERED
  fprintf(fp, "Hello from CommStaggered::exchange\n");
  fflush(fp);
#endif
  if (layout != Comm::LAYOUT_STAGGERED) {
    CommTiled::exchange();
    return;
  }
#ifdef DEBUG_COMM_STAGGERED
  fprintf(fp, "start exchange\n");
  fflush(fp);
#endif
  int i,m,nexch,nsend,nrecv,nlocal,proc,offset;
  double lo,hi,value;
  double **x;
  AtomVec *avec = atom->avec;

  // clear global->local map for owned and ghost atoms
  // b/c atoms migrate to new procs in exchange() and
  //   new ghosts are created in borders()
  // map_set() is done at end of borders()
  // clear ghost count and any ghost bonus data internal to AtomVec

  if (map_style != Atom::MAP_NONE) atom->map_clear();
  atom->nghost = 0;
  atom->avec->clear_bonus();

  // ensure send buf has extra space for a single atom
  // only need to reset if a fix can dynamically add to size of single atom

  if (maxexchange_fix_dynamic) {
    // TODO: remove commented code
    //int bufextra_old = bufextra;
    //init_exchange();
    //if (bufextra > bufextra_old) grow_send(maxsend+bufextra,2);
    init_exchange();
    if (bufextra > bufextra_max) {
      grow_send(maxsend+bufextra,2);
      bufextra = bufextra_max;
    }
  }

  // domain properties used in exchange method and methods it calls
  // subbox bounds for orthogonal or triclinic

  if (triclinic == 0) {
    prd = domain->prd;
    boxlo = domain->boxlo;
    boxhi = domain->boxhi;
    sublo = domain->sublo;
    subhi = domain->subhi;
  } else {
    prd = domain->prd_lamda;
    boxlo = domain->boxlo_lamda;
    boxhi = domain->boxhi_lamda;
    sublo = domain->sublo_lamda;
    subhi = domain->subhi_lamda;
  }

  // loop over dimensions

  dimension = domain->dimension;

  ///> @todo Loop over dimensions starting with layers for comm_style = staggered?
  ///>       Could cause bugs for brick functions.
  for (int dim = 0; dim < dimension; dim++) {

    // fill buffer with atoms leaving my box, using < and >=
    // when atom is deleted, fill it in with last atom

    x = atom->x;
    lo = sublo[dim];
    hi = subhi[dim];
    nlocal = atom->nlocal;
    i = nsend = 0;

    while (i < nlocal) {
      if (x[i][dim] < lo || x[i][dim] >= hi) {
        if (nsend > maxsend) grow_send(nsend,1);
        proc = (this->*point_drop)(dim,x[i]);
        if (proc != me) {
          buf_send[nsend++] = proc;
          nsend += avec->pack_exchange(i,&buf_send[nsend]);
        } else {
          // DEBUG statment
          ///> @todo remove debug statement
          error->warning(FLERR,"Losing atom in CommStaggered::exchange() send, "
                         "likely bad dynamics");
        }
        avec->copy(nlocal-1,i,1);
        nlocal--;
      } else i++;
    }
    atom->nlocal = nlocal;

    // send and recv atoms from neighbor procs that touch my sub-box in dim
    // no send/recv with self
    // send size of message first
    // receiver may receive multiple messages, realloc buf_recv if needed

    nexch = nexchproc[dim];
    if (!nexch) continue;
#ifdef DEBUG_COMM_STAGGERED
    for (m = 0; m < nexch; m++) fprintf(fp, "step 0 m %i exchproc %i\n", m, exchproc[dim][m]);
    fflush(fp);
#endif

    for (m = 0; m < nexch; m++) {
#ifdef DEBUG_COMM_STAGGERED
      fprintf(fp, "recv 1 INT from %i in dim %i\n", exchproc[dim][m], dim);
#endif
      MPI_Irecv(&exchnum[dim][m],1,MPI_INT,exchproc[dim][m],2*dim,world,&requests[m]);
    }
#ifdef DEBUG_COMM_STAGGERED
    fflush(fp);
#endif
    for (m = 0; m < nexch; m++) {
#ifdef DEBUG_COMM_STAGGERED
      fprintf(fp, "send 1 INT (%i) to %i in dim %i\n", nsend, exchproc[dim][m], dim);
      fflush(fp);
#endif
      MPI_Send(&nsend,1,MPI_INT,exchproc[dim][m],2*dim,world);
    }
#ifdef DEBUG_COMM_STAGGERED
    fprintf(fp, "start Waitall 2\n");
    fflush(fp);
#endif
    MPI_Waitall(nexch,requests,MPI_STATUS_IGNORE);
#ifdef DEBUG_COMM_STAGGERED
    for (m = 0; m < nexch; m++) fprintf(fp, "step 1 m %i exchproc %i exchnum %i\n", m, exchproc[dim][m], exchnum[dim][m]);
    fprintf(fp, "end Waitall 2\n");
    fflush(fp);
#endif

    nrecv = 0;
    for (m = 0; m < nexch; m++) nrecv += exchnum[dim][m];
    if (nrecv > maxrecv) grow_recv(nrecv);

    offset = 0;
    for (m = 0; m < nexch; m++) {
#ifdef DEBUG_COMM_STAGGERED
      fprintf(fp, "recv %i DOUBLESs from %i in dim %i\n", exchnum[dim][m], exchproc[dim][m], dim);
#endif
      MPI_Irecv(&buf_recv[offset],exchnum[dim][m],MPI_DOUBLE,exchproc[dim][m],2*dim+1,world,&requests[m]); ///>@debug count may be zero
      offset += exchnum[dim][m];
    }
#ifdef DEBUG_COMM_STAGGERED
    fflush(fp);
#endif
    for (m = 0; m < nexch; m++) {
#ifdef DEBUG_COMM_STAGGERED
      fprintf(fp, "send %i DOUBLEs to %i in dim %i\n", nsend, exchproc[dim][m], dim);
      fflush(fp);
#endif
      MPI_Send(buf_send,nsend,MPI_DOUBLE,exchproc[dim][m],2*dim+1,world); ///>@debug count may be zero
    }
#ifdef DEBUG_COMM_STAGGERED
    fprintf(fp, "start Waitall 3\n");
    fflush(fp);
#endif
    MPI_Waitall(nexch,requests,MPI_STATUS_IGNORE);
#ifdef DEBUG_COMM_STAGGERED
    fprintf(fp, "end Waitall 3\n");
    fflush(fp);
#endif

    // check incoming atoms to see if I own it and they are in my box
    // if so, add to my list
    // box check is only for this dimension,
    //   atom may be passed to another proc in later dims

    m = 0;
    while (m < nrecv) {
      proc = static_cast<int> (buf_recv[m++]);
      if (proc == me) {
        value = buf_recv[m+dim+1];
        if (value >= lo && value < hi) {
          m += avec->unpack_exchange(&buf_recv[m]);
          continue;
        } else {
          // DEBUG statment
          ///> @todo remove debug statement
          error->warning(FLERR,"Losing atom in CommStaggered::exchange() recv");
        }
      }
      m += static_cast<int> (buf_recv[m]);
    }
  }

  if (atom->firstgroupname) atom->first_reorder();
#ifdef DEBUG_COMM_STAGGERED
  fprintf(fp, "end exchange\n");
  fflush(fp);
#endif
}

/**
  * Determine overlap list of Noverlap procs the lo/hi box overlaps.
  * overlap = non-zero area in common between box and proc sub-domain.
  * Recursive method for traversing an RCB tree of cuts.
  * No need to split lo/hi box as recurse b/c OK if box extends outside RCB box.
  * @todo Rewrite and consider the fact, that the requested neighbours are in neighbouring layers, rows, cells
  *       The communication order is still tiled, so one cannot use staggered specific benefits yet.
  * @param[in] lo left box boundaries in spatial box coordinates
  * @param[in] hi right box boundaries in spatial box coordinates
  * @note Writes overlaping processors to array overlap.
  *       The number of processors in this array is noverlap.
  * @param[out] indexme index of own processor in overlap list
  * @todo lo can be outside of box. Is this problematic?
  * @todo hi can be outside of box. Is this problematic?
  */

void CommStaggered::box_drop_staggered(int /*idim*/, double *lo, double *hi, int &indexme)
{
  int layer_lo, layer_hi, layer_i;
  int row_lo, row_hi, row_i;
  int cell_lo, cell_hi, cell_i;
  int rank_overlap;
#ifdef DEBUG_COMM_STAGGERED
  fprintf(fp, "Hello from box_drop_staggered lo %f %f %f hi %f %f %f\n",
      lo[0]/domain->prd[0], lo[1]/domain->prd[1], lo[2]/domain->prd[2],
      hi[0]/domain->prd[0], hi[1]/domain->prd[1], hi[2]/domain->prd[2]);
  fflush(fp);
#endif

  // calculate layer range
  box_drop_1d(lo[staggered2spatial[0]], hi[staggered2spatial[0]], layer_splits, n_layers,
             staggered2spatial[0], layer_lo, layer_hi);
#ifdef DEBUG_COMM_STAGGERED
  fprintf(fp, "layer_lo %i layer_hi %i\n", layer_lo, layer_hi);
  fflush(fp);
#endif
  for (int i_layer=layer_lo; i_layer<=layer_hi; i_layer++) {
    // calculate row range
    box_drop_1d(lo[staggered2spatial[1]], hi[staggered2spatial[1]], row_splits[i_layer], n_rows,
                staggered2spatial[1], row_lo, row_hi);
#ifdef DEBUG_COMM_STAGGERED
    fprintf(fp, "layer %i row_lo %i row_hi %i\n", i_layer, row_lo, row_hi);
    fflush(fp);
#endif
    //neighbours_z->set_row_range(row_lo, row_hi);

    for (int i_row=row_lo; i_row<=row_hi; i_row++) {
      // calculate cell range
      box_drop_1d(lo[staggered2spatial[2]], hi[staggered2spatial[2]], cell_splits[i_layer][i_row],
                  n_cells, staggered2spatial[2], cell_lo, cell_hi);
#ifdef DEBUG_COMM_STAGGERED
      fprintf(fp, "layer %i row %i cell_lo %i cell_hi %i\n", i_layer, i_row, cell_lo, cell_hi);
      fflush(fp);
#endif

      for (int i_cell=cell_lo; i_cell<=cell_hi; i_cell++) {
        // this cell overlaps with the given box
        rank_overlap = staggered_grid2proc[i_layer][i_row][i_cell];

        // allocate more memory if required
        if (noverlap == maxoverlap) {
          maxoverlap += DELTA_PROCS;
          memory->grow(overlap,maxoverlap,"comm:overlap");
        }

        if (rank_overlap == me) indexme = noverlap;
        overlap[noverlap++] = rank_overlap;
#ifdef DEBUG_COMM_STAGGERED
        fprintf(fp, "found rank_overlap %i\n", rank_overlap);
        fflush(fp);
#endif
      }
    }
  }
#ifdef DEBUG_COMM_STAGGERED
  fprintf(fp, "Bye from box_drop_staggered.\n");
  fflush(fp);
#endif
}

/**
  * @brief return other box owned by proc as lo/hi corner pts
  * @param[in] proc rank of a processor
  * @param[out] lo low boundaries of processor in box units
  * @param[out] hi high boundaries of processor in box units
  */

void CommStaggered::box_other_staggered(int /*idim*/, int /*idir*/, int proc, double *lo, double *hi)
{
#ifdef DEBUG_COMM_STAGGERED
  fprintf(fp, "Hello from box_other_staggered\n");
  fflush(fp);
#endif
  int * loc = staggered_proc2grid[proc];

  lo[staggered2spatial[0]] = boxlo[staggered2spatial[0]] + prd[staggered2spatial[0]]*layer_splits[loc[0]];
  if (loc[0]+1 == n_layers) hi[staggered2spatial[0]] = boxhi[staggered2spatial[0]];
  else hi[staggered2spatial[0]] = boxlo[staggered2spatial[0]] + prd[staggered2spatial[0]]*layer_splits[loc[0]+1];

  lo[staggered2spatial[1]] = boxlo[staggered2spatial[1]] + prd[staggered2spatial[1]]*row_splits[loc[0]][loc[1]];
  if (loc[1]+1 == n_rows) hi[staggered2spatial[1]] = boxhi[staggered2spatial[1]];
  else hi[staggered2spatial[1]] = boxlo[staggered2spatial[1]] + prd[staggered2spatial[1]]*row_splits[loc[0]][loc[1]+1];

  lo[staggered2spatial[2]] = boxlo[staggered2spatial[2]] + prd[staggered2spatial[2]]*cell_splits[loc[0]][loc[1]][loc[2]];
  if (loc[2]+1 == n_cells) hi[staggered2spatial[2]] = boxhi[staggered2spatial[2]];
  else hi[staggered2spatial[2]] = boxlo[staggered2spatial[2]] + prd[staggered2spatial[2]]*cell_splits[loc[0]][loc[1]][loc[2]+1];

}

/**
  * @brief return other box owned by proc as lo/hi corner pts
  * @param[in] proc rank of a processor
  * @param[out] lo low boundaries of processor in [0,1] units
  * @param[out] hi high boundaries of processor in [0,1] units
  */

void CommStaggered::box_other_mysplit(int proc, double *lo, double *hi)
{
#ifdef DEBUG_COMM_STAGGERED
  fprintf(fp, "Hello from box_other_mysplit\n");
  fflush(fp);
#endif
  int * loc = staggered_proc2grid[proc];

  lo[staggered2spatial[0]] = layer_splits[loc[0]];
  hi[staggered2spatial[0]] = layer_splits[loc[0]+1];

  lo[staggered2spatial[1]] = row_splits[loc[0]][loc[1]];
  hi[staggered2spatial[1]] = row_splits[loc[0]][loc[1]+1];

  lo[staggered2spatial[2]] = cell_splits[loc[0]][loc[1]][loc[2]];
  hi[staggered2spatial[2]] = cell_splits[loc[0]][loc[1]][loc[2]+1];
}

/**
  * return 1 if proc's box touches me, else 0
  * @param[in] proc other processor
  * @param[in] idim dimension in which the touch is tested
  * @param[in] idir direction in which the touch is tested (0 = left of my processor, 1 = right)
  * @return 1 if neighbous in idim and idir, 0 otherwise
  */

int CommStaggered::box_touch_staggered(int proc, int idim, int idir)
{
#ifdef DEBUG_COMM_STAGGERED
  fprintf(fp, "Hello from box_touch_staggered\n");
  fflush(fp);
#endif
  // use staggered grid notation
  idim = spatial2staggered[idim];

  // sending to left
  // only touches if proc hi = my lo, or if proc hi = boxhi and my lo = boxlo

  if (idir == 0) {
    if (staggered_proc2grid[proc][idim] + 1 == staggered_myloc[idim])
      return 1;
    else if (staggered_proc2grid[proc][idim] == staggered_procgrid[idim] - 1 &&
             staggered_myloc[idim] == 0)
      return 1;

  // sending to right
  // only touches if proc lo = my hi, or if proc lo = boxlo and my hi = boxhi

  } else {
    if (staggered_proc2grid[proc][idim] == staggered_myloc[idim] + 1)
      return 1;
    else if (staggered_proc2grid[proc][idim] == 0 &&
             staggered_myloc[idim] == staggered_procgrid[idim] -1)
      return 1;
  }

  return 0;
}

/**
  * @brief Determine which proc owns point x.
  * Drop point into staggered mesh.
  * Sending starts with increasing idim.
  * Smaller dimensions that idim are already exchanged.
  * @todo Rewrite function since it depends on the tiled communication routine.
  *       When starting with communictation in layer direction, on can simply
  *       check only the neighbouring layer and so on.
  * @param[in] idim dimension in which the point is outside
  * @param[in] x position of point
  */

int CommStaggered::point_drop_staggered(int idim, double *x)
{
#ifdef DEBUG_COMM_STAGGERED
  fprintf(fp, "Hello from point_drop_staggered\n");
  fflush(fp);
#endif
  double xnew[3];
  xnew[0] = x[0]; xnew[1] = x[1]; xnew[2] = x[2];

  // Ensure that the target processor is a neighbour.
  // idim = 0
  // Set xnew to own boundaries if x is outside in dimensions 1 and 2.
  // idim = 1
  // Set xnew to own boundaries if x is outside in dimensions 2.
  // idim = 2
  // Do not change xnew.


  if (idim == 0) {
    if (xnew[1] < sublo[1] || xnew[1] > subhi[1]) {
      if (closer_subbox_edge(1,x)) xnew[1] = subhi[1];
      else xnew[1] = sublo[1];
    }
  }
  if (idim <= 1) {
    if (xnew[2] < sublo[2] || xnew[2] > subhi[2]) {
      if (closer_subbox_edge(2,x)) xnew[2] = subhi[2];
      else xnew[2] = sublo[2];
    }
  }

  int proc = point_drop_staggered_recurse(xnew);
  double proc_lo[3], proc_hi[3];
  if (proc == me) return me;

  // x communication
  if (idim == 0) {
    box_other_mysplit(proc, proc_lo, proc_hi);

    int done = 1;
    //  proc y lo == me y hi
    if (proc_lo[1] == mysplit[1][1]) {
      // proc is right of me in y
      xnew[1] -= EPSILON * (subhi[1]-sublo[1]);
      // move atom inside of own proc
      // -> atom is not send to this neighbour (in this step)
      // atom could be sent in y communication
      done = 0;
      // -> drop point again to find different processor
    }
    if (proc_lo[2] == mysplit[2][1]) {
      // proc is right of me in z
      xnew[2] -= EPSILON * (subhi[2]-sublo[2]);
      // change xnew to inside in z
      done = 0;
      // -> drop point again
    }
    if (!done) {
      proc = point_drop_staggered_recurse(xnew);
      box_other_mysplit(proc, proc_lo, proc_hi);
      done = 1;
      // proc y lo == me y hi
      // -> proc is right of me in y
      if (proc_lo[1] == mysplit[1][1]) {
        // set atom inside in y
        xnew[1] -= EPSILON * (subhi[1]-sublo[1]);
        // drop again
        done = 0;
      }
      // proc z lo == me z hi
      // -> proc is right of me in z
      if (proc_lo[2] == mysplit[2][1]) {
        // set atom inside in z
        xnew[2] -= EPSILON * (subhi[2]-sublo[2]);
        // drop again
        done = 0;
      }
      if (!done) proc = point_drop_staggered_recurse(xnew);
      ///> ->The atom is shifted inside of the own box in y and z if required to find a neighbour in x direction
    }
  } else if (idim == 1) {
    // proc z lo == me z hi
    // -> proc is right of me in z
    box_other_mysplit(proc, proc_lo, proc_hi);
    if (proc_lo[2] == mysplit[2][1]) {
      xnew[2] -= EPSILON * (subhi[2]-sublo[2]);
      proc = point_drop_staggered_recurse(xnew);
    }
  }

  return proc;
}

/**
  * @brief Drop point in staggered mesh in dimension dim.
  * @note Do not use for box drop since the boundary conditions differ.
  *       A point is inside for p in [splitlo, splithi).
  *       A box point is inside for p in (splitlo, splithi).
  * @param[in] value to sort into staggered mesh in box coordinatos
  * @param[in] splitlist list of splits e.g. layer_splits
  * @param[in] listlen length of splitlist
  * @param[in] dim spatial dimension of value
  * @return i in staggered grid in the given dimension
  */

int CommStaggered::point_drop_1d(double value, double * splitlist, int listlen, int dim)
{
#ifdef DEBUG_COMM_STAGGERED
  fprintf(fp, "Hello from value drop dim %i value %f listlen %i\n", dim, value/domain->prd[dim], listlen);
  fflush(fp);
#endif
  double cut;
  int procmid;

  int proclower = 0;
  int procupper = listlen -1;
  while (proclower != procupper) {
    procmid = proclower + (procupper - proclower) / 2 + 1;
    cut = boxlo[dim] + prd[dim] * splitlist[procmid];
#ifdef DEBUG_COMM_STAGGERED
    fprintf(fp, "proclower %i procupper %i cut %f\n", proclower, procupper, cut/domain->prd[dim]);
    fflush(fp);
#endif
    if (value < cut) procupper = procmid -1;
    else proclower = procmid;
  }
#ifdef DEBUG_COMM_STAGGERED
  fprintf(fp, "proc %i\n", proclower);
  fflush(fp);
#endif
  return proclower;
}

/**
  * @brief Drop 1d box in staggered mesh in dimension dim.
  * @note Do not use for point drop since the boundary conditions differ.
  *       A point is inside for p in [splitlo, splithi).
  *       A box point is inside for p in (splitlo, splithi).
  * @param[in] lo lower value to sort into staggered mesh in box coordinatos
  * @param[in] hi lower value to sort into staggered mesh in box coordinatos
  * @param[in] splitlist list of splits e.g. layer_splits
  * @param[in] listlen length of splitlist
  * @param[in] dim spatial dimension of value
  * @param[out] rtn_lo processor i in staggered grid in the given dimension of value lo
  * @param[out] rtn_hi processor i in staggered grid in the given dimension of value hi
  */

void CommStaggered::box_drop_1d(double lo, double hi, double * splitlist, int listlen, int dim, int & rtn_lo, int & rtn_hi)
{
#ifdef DEBUG_COMM_STAGGERED
  fprintf(fp, "Hello from box drop 1d dim %i lo %f hi %f listlen %i\n", dim, lo/domain->prd[dim], hi/domain->prd[dim], listlen);
  fflush(fp);
#endif
  double cut;
  int procmid, proclower, procupper;

  // drop lo
  proclower = 0;
  procupper = listlen -1;
  while (proclower != procupper) {
    procmid = proclower + (procupper - proclower) / 2 + 1;
    cut = boxlo[dim] + prd[dim] * splitlist[procmid];
#ifdef DEBUG_COMM_STAGGERED
    fprintf(fp, "proclower %i procupper %i cut %f\n", proclower, procupper, cut/domain->prd[dim]);
    fflush(fp);
#endif
    if (lo < cut) procupper = procmid -1;
    else proclower = procmid;
  }
  rtn_lo = proclower;
  ///> @todo do not return touch only boxes?

  // drop hi
  // hi > lo -> rtn_hi >= rtn_lo
  procupper = listlen -1;
  while (proclower != procupper) {
    procmid = proclower + (procupper - proclower) / 2 + 1;
    cut = boxlo[dim] + prd[dim] * splitlist[procmid];
#ifdef DEBUG_COMM_STAGGERED
    fprintf(fp, "proclower %i procupper %i cut %f\n", proclower, procupper, cut/domain->prd[dim]);
    fflush(fp);
#endif
    if (hi > cut ) proclower = procmid;
    else procupper = procmid -1;
  }
  rtn_hi = proclower;
  ///> @todo do not return touch only boxes?

#ifdef DEBUG_COMM_STAGGERED
  fprintf(fp, "Bye from box drop 1d proc lo %i proc hi %i\n", rtn_lo, rtn_hi);
  fflush(fp);
#endif
}

/**
  * drop in staggered mesh
  * @param[in] x point in box coordinates
  * @return rank of processor
  */

int CommStaggered::point_drop_staggered_recurse(double *x)
{
#ifdef DEBUG_COMM_STAGGERED
  fprintf(fp, "Hello from point_drop_staggered_recurse\n");
  fflush(fp);
#endif
  int i_layer, i_row, i_cell;
  // find layer
  i_layer = point_drop_1d(x[staggered2spatial[0]], layer_splits,                n_layers, staggered2spatial[0]);
  i_row   = point_drop_1d(x[staggered2spatial[1]], row_splits[i_layer],         n_rows,   staggered2spatial[1]);
  i_cell  = point_drop_1d(x[staggered2spatial[2]], cell_splits[i_layer][i_row], n_cells,  staggered2spatial[2]);

#ifdef DEBUG_COMM_STAGGERED
  fprintf(fp, "Bye from point_drop_staggered_recurse: dropped: x %f y %f z %f proc %i\n", x[0], x[1], x[2], staggered_grid2proc[i_layer][i_row][i_cell]);
  fflush(fp);
#endif
  return staggered_grid2proc[i_layer][i_row][i_cell];
}

/**
  * probably wrong line: if RCB decomp exists and just changed, gather needed global RCB info
  *
  * required input: procgrid, myloc, spatial2staggered, staggered2spatial, grid2proc
  */

void CommStaggered::coord2proc_setup()
{
#ifdef DEBUG_COMM_STAGGERED
  fprintf(fp, "Hello from coord2proc_setup\n");
  fflush(fp);
#endif
  if (!staggerednew) return;
#ifdef DEBUG_COMM_STAGGERED
  fprintf(fp, "Hello from CommStaggered::coord2proc_setup\n");
  fflush(fp);
#endif
  staggerednew = 0;

  // gather lower boundaries of all processors
  double gatherbuffer[nprocs*3];
  gatherbuffer[me*3+spatial2staggered[0]] = mysplit[0][0];
  gatherbuffer[me*3+spatial2staggered[1]] = mysplit[1][0];
  gatherbuffer[me*3+spatial2staggered[2]] = mysplit[2][0];
  // TODO use gatherv to prevent double sendings?
  MPI_Allgather(MPI_IN_PLACE,3,MPI_DOUBLE,gatherbuffer,3,MPI_DOUBLE,world);

  // store splits
  for (int il=0; il<n_layers; il++) {
    layer_splits[il] = gatherbuffer[staggered_grid2proc[il][0][0]*3+0];
    for (int ir=0; ir<n_rows; ir++) {
      row_splits[il][ir] = gatherbuffer[staggered_grid2proc[il][ir][0]*3+1];
      for (int ic=0; ic<n_cells; ic++) {
        cell_splits[il][ir][ic] = gatherbuffer[staggered_grid2proc[il][ir][ic]*3+2];
      }
      cell_splits[il][ir][n_cells] = 1;
    }
    row_splits[il][n_rows] = 1;
  }
  layer_splits[n_layers] = 1;
}

/**
  * Determine which proc owns atom with coord x[3] based on current decomp.
  * x will be in box (orthogonal) or lamda coords (triclinic).
  * if layout = UNIFORM or NONUNIFORM, invoke parent method.
  * if layout = STAGGERED, use point_drop_recurse().
  * @return owning proc ID, ignore igx,igy,igz
  */

int CommStaggered::coord2proc(double *x, int &igx, int &igy, int &igz)
{
#ifdef DEBUG_COMM_STAGGERED
  fprintf(fp, "Hello from coord2proc with comm->layout %i layout %i (uniform %i, nonuniform %i, tiled %i, staggered %i\n", comm->layout, layout, Comm::LAYOUT_UNIFORM, Comm::LAYOUT_NONUNIFORM, Comm::LAYOUT_TILED , Comm::LAYOUT_STAGGERED);
  fflush(fp);
  int p = -1;
  if (layout != Comm::LAYOUT_STAGGERED) {
    p = Comm::coord2proc(x,igx,igy,igz);
    fprintf(fp, "layout != staggered and p = %i\n", p);
    fflush(fp);
  } else {
    p = point_drop_staggered_recurse(x);
    fprintf(fp, "layout == staggered and p = %i\n", p);
    fflush(fp);
  }
  return p;
#else
  if (layout != Comm::LAYOUT_STAGGERED) return Comm::coord2proc(x,igx,igy,igz);
  return point_drop_staggered_recurse(x);
#endif
}

/* ----------------------------------------------------------------------
   return # of bytes of allocated memory
------------------------------------------------------------------------- */

double CommStaggered::memory_usage()
{
#ifdef DEBUG_COMM_STAGGERED
  fprintf(fp, "Hello from memory_usage\n");
  fflush(fp);
#endif
  double bytes = 0;
  return bytes;
}
