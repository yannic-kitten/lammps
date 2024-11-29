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

#ifdef FIX_CLASS
// clang-format off
FixStyle(balance/overlay,FixBalanceOverlay);
// clang-format on
#else

#ifndef LMP_FIX_BALANCE_OVERLAY_H
#define LMP_FIX_BALANCE_OVERLAY_H

#include "fix.h"
#include "ALL.hpp"
#include "pointers.h"
#include <mpi.h>

namespace LAMMPS_NS {

/**
 *  Small wrapper for the lammps timers to get time intervals.
 */

class JullTimer : protected Pointers {
  public:
    JullTimer(class LAMMPS *);
    void init();
    void print_times(int);
    double get_timer(int);
    double get_work();
  private:
    double time[4];             // value of lammps timers
    double time_interval[4];    // time interval between two calls

    void set_time();
};

/**
 *  Fix for dynamic load balancing with the ALL-library.
 *  Local load balancing is supported for a staggered and a tensor grid.
 *  Global load balancing is supported for a staggered grid.
 *  Local and global load balancing can be combined.
 */

class FixBalanceOverlay : public Fix {
 public:
  FixBalanceOverlay(class LAMMPS *, int, char **);
  ~FixBalanceOverlay() override;
  int setmask() override;
  void post_constructor() override;
  void init() override;
  void pre_exchange() override;
  void min_pre_exchange() override;
  double compute_vector(int) override;
  double memory_usage() override;

 private:

  ///> @todo: remove kspace_flag (?) // YK
  //int kspace_flag;              // 1 if KSpace solver defined
  bigint lastbalance;           // last timestep balancing was attempted

  double maxloadperproc;        // max load on any processor

  // user set variables

  int nevery;                   // call load balancer after nevery steps
  int gridstyle;                // STAGGERED, TENSOR, UNKNOWN_GRID
  int workstyle;                // NATOMS,TIME,UNKNOWN_WORK,RANK
  int wtflag;                   // use per atom weights 1, otherwise 0
  ///> @todo: add attributes    // YK
  //int sortflag;               // 1 for sorting comm messages                              // YK
  //int reportonly;             // 1 if skipping rebalancing and only computing imbalance   // YK
  double bw;                    // bin width for histogram
  double local_threshold;       // imbalance threshold for tensor and staggered
  double global_threshold;      // imbalance threshold for histogram
  bool rescale_histogram;       // rescale histogram to scalar work
  bool use_bw_threshold;        // bin width threshold set
  bool use_local_lb;            // jull_local (staggered or tensor) required
  bool use_global_lb;           // jull_global (histogram) required
  bool verbose;                 // write stats to log
  int nevery_file_write;        // argument of file

  // calculated variables

  double work;                  // load of this rank
  double imbalance;             // max work / avg work
  double timer_lb;              // duration of load balancing step in seconds
  MPI_File fh;                  // file handle for parallel output
  MPI_Comm x_masters;           // for tensor to gather xsplit
  MPI_Comm y_masters;           // for tensor to gather ysplit
  MPI_Comm z_masters;           // for tensor to gather zsplit
  bigint next_file_write;       // next timestep with file_write
  bool reduce_outvec_flag;      // recaluclate out vector?
  double outvec_timer[3];

  // ALL objects

  ALL::ALL<double, double> *jull_local;   // staggered or tensor
  ALL::ALL<double, double> *jull_global;  // histogram
  ALL::ALL<double, double> *jull_last;    // pointer to last used object or nullptr

  // ALL input

  std::vector<int> n_bins;                      // number of bin of histogram
  std::vector<int> myloc_vec;                   // position in staggered/tensor gird per dimension
  std::vector<int> procgrid_vec;                // size of staggered/tensor grid per dimension
  std::vector<double> minimum_domain_size;      // in box units [0, domain->prd]
  std::vector<ALL::Point<double>> my_vertices;  // in box units [0, domain->prd]

  // class pointers

  class FixStoreAtom *fixstore;     // per-atom weights for histogram stored in FixStore
  class Irregular *irregular;       // for atom migration after boudary update
  // TODO: remove (YK)
  class JullTimer *jull_timer;      // wrapper for lammps timers
  double *weight;                  // ptr to FixStore weight vector

  // functions

  std::vector<ALL::Point<double>> get_comm_vertices();
  std::vector<double> calc_histogram(int);
  std::vector<double> get_sys_size_from_domain();
  double get_work();

  void set_comm_vertices(std::vector<ALL::Point<double>>);

  void balance();
  void calc_imbalance();
  void set_weights();
  void unset_weights();
  void balance_local();
  void balance_global();

  void print_domains();
  void print_histogram(std::vector<double>);

#ifdef DEBUG_COMM_STAGGERED
  FILE * fp;
#endif
};

}    // namespace LAMMPS_NS

#endif
#endif
