/* -*- c++ -*- ----------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   https://www.lammps.org/, Sandia National Laboratories
   LAMMPS development team: developers@lammps.org

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

#ifndef LMP_COMM_STAGGERED_H
#define LMP_COMM_STAGGERED_H

#include "comm.h"
#include "comm_tiled.h"

namespace LAMMPS_NS {

class CommStaggered : public CommTiled {
 public:
  CommStaggered(class LAMMPS *, class Comm *, char * arg);

  ~CommStaggered() override;

  void setup() override;                // setup comm pattern
  void exchange() override;             // move atoms to new procs

  void coord2proc_setup() override;
  int coord2proc(double *, int &, int &, int &) override;

  double memory_usage() override;

 private:
  double * layer_splits;                // store splits for layers
  double ** row_splits;                 // store splits for rows
  double *** cell_splits;               // store splits for cells

#ifdef DEBUG_COMM_STAGGERED
  FILE * fp;
#endif

  void init_buffers_staggered();

  // box drop and other functions
  void box_drop_1d(double, double, double *, int , int , int &, int &);
  int point_drop_1d(double, double *, int, int);

  typedef void (CommStaggered::*BoxDropPtr)(int, double *, double *, int &);
  BoxDropPtr box_drop;
  void box_drop_staggered(int, double *, double *, int &);

  typedef void (CommStaggered::*BoxOtherPtr)(int, int, int, double *, double *);
  BoxOtherPtr box_other;
  void box_other_staggered(int, int, int, double *, double *);
  void box_other_mysplit(int, double *, double *);

  typedef int (CommStaggered::*BoxTouchPtr)(int, int, int);
  BoxTouchPtr box_touch;
  int box_touch_staggered(int, int, int);

  typedef int (CommStaggered::*PointDropPtr)(int, double *);
  PointDropPtr point_drop;
  int point_drop_staggered(int, double *);
  int point_drop_staggered_recurse(double *);
};

}    // namespace LAMMPS_NS

#endif
