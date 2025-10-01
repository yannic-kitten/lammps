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
FixStyle(balance/all,FixBalanceAll);
// clang-format on
#else

#ifndef LMP_FIX_BALANCE_ALL_H
#define LMP_FIX_BALANCE_ALL_H

#include "fix.h"
#include "ALL.hpp"
#include "pointers.h"
#include <mpi.h>
#include <string>

namespace LAMMPS_NS {

class FixBalanceAll : public Fix {
 public:
  FixBalanceAll(class LAMMPS *, int, char **);
  ~FixBalanceAll() override;
  int setmask() override;
  void post_constructor() override;
  void init() override;
  void pre_exchange() override;
  double compute_vector(int) override;
  double memory_usage() override;

 private:

  bigint lastbalance;           // last timestep balancing was attempted

  double maxloadperproc;        // max load on any processor

  // user set variables

  int nevery;                   // call load balancer after nevery steps
  int gridstyle;                // STAGGERED, TENSOR, UNKNOWN_GRID
  int wtflag;                   // use per atom weights 1, otherwise 0
  int varflag;                  // 1 if weight style var(iable) is used
  double bw;                    // bin width for histogram
  double local_threshold;       // imbalance threshold for tensor and staggered
  double global_threshold;      // imbalance threshold for histogram
  bool use_bw_threshold;        // bin width threshold set
  bool use_local_lb;            // all_local (staggered or tensor) required
  bool use_global_lb;           // all_global (histogram) required
  bool verbose;                 // write stats to log

  // calculated variables

  double work;                  // load of this rank
  double imbalance;             // max work / avg work
  double timer_lb;              // duration of load balancing step in seconds
  MPI_Comm x_masters;           // for tensor to gather xsplit
  MPI_Comm y_masters;           // for tensor to gather ysplit
  MPI_Comm z_masters;           // for tensor to gather zsplit
  bool reduce_outvec_flag;      // recaluclate out vector?
  double outvec_timer[3]; // TODO(?): remove
  int stag_cut_order[3];          // order of dimensions to cut for staggered grid methods (default is zyx: [2, 1, 0])
  std::string stag_cut_order_str; // string representation of cut order for staggered grid methods

  // ALL objects

  ALL::ALL<double, double> *all_local;   // staggered or tensor
  ALL::ALL<double, double> *all_global;  // histogram
  ALL::ALL<double, double> *all_last;    // pointer to last used object or nullptr

  // ALL input

  std::vector<int> n_bins;                      // number of bin of histogram
  std::vector<int> myloc_vec;                   // position in staggered/tensor gird per dimension
  std::vector<int> procgrid_vec;                // size of staggered/tensor grid per dimension
  std::vector<double> minimum_domain_size;      // in box units [0, domain->prd]
  std::vector<ALL::Point<double>> my_vertices;  // in box units [0, domain->prd]

  // class pointers

  class FixStoreAtom *fixstore;     // per-atom weights for histogram stored in FixStore
  class Irregular *irregular;       // for atom migration after boudary update

  int nimbalance;                   // number of user-specified weight styles
  class Imbalance **imbalances;     // list of Imb classes, one per weight style
  double *weight;                   // ptr to FixStore weight vector

  // functions

  std::vector<ALL::Point<double>> get_comm_vertices();
  std::vector<double> calc_histogram(int);
  std::vector<double> get_sys_size_from_domain();
  double get_work();

  void set_comm_vertices(std::vector<ALL::Point<double>>);

  void balance();
  void balance_local();
  void balance_global();
  void init_imbalance(int);
  void calc_imbalance();
  void set_weights();
  void unset_weights();

  void weight_storage();

  //void print_domains();
  //void print_histogram(std::vector<double>);
};

}    // namespace LAMMPS_NS

#endif
#endif
