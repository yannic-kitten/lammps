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

#include "fix_balance_all.h"

#include "ALL.hpp"

#include "timer.h"
#include "pointers.h"
#include "pair.h"
#include "fix_store_atom.h"

#include "atom.h"
#include "comm.h"
#include "domain.h"
#include "error.h"
#include "force.h"
#include "group.h"
#include "imbalance.h"
#include "imbalance_group.h"
#include "imbalance_neigh.h"
#include "imbalance_store.h"
#include "imbalance_time.h"
#include "imbalance_var.h"
#include "irregular.h"
#include "kspace.h"
#include "modify.h"
#include "neighbor.h"
#include "memory.h"
#include "update.h"

#include <cstring>
#include <mpi.h>
#include <math.h>

using namespace LAMMPS_NS;
using namespace FixConst;

enum { TENSOR, STAGGERED, UNKNOWN_GRID };
enum { TENSOR_MAX, TENSOR_CLASSIC, NONE };

// clang-format off

/**
 *  create class and parse arguments in LAMMPS script.
 *  Syntax:
 *
 *  fix ID group-ID balance/all keyword args ...
 *
 *  required keyword/arg pairs
 *
 *      every arg = nevery
 *          nevery = perform dynamic load balancing every this many steps
 *      grid style args = define grid
 *          style = staggered or tensor
 *              staggered args = none
 *              tensor args = max or classic
 *                max = use TENSOR_MAX method of ALL
 *                classic = use TENSOR method of ALL
 *
 *      weight style args = use weighted particle counts for the balancing
 *          style = group or neigh or time or var or store
 *              group args = Ngroup group1 weight1 group2 weight2 ...
 *                Ngroup = number of groups with assigned weights
 *                group1, group2, ... = group IDs
 *                weight1, weight2, ...   = corresponding weight factors
 *              neigh factor = compute weight based on number of neighbors
 *                factor = scaling factor (> 0)
 *              time factor = compute weight based on time spend computing
 *                factor = scaling factor (> 0)
 *              var name = take weight from atom-style variable
 *                name = name of the atom-style variable
 *              store name = store weight in custom atom property defined by fix property/atom command
 *                name = atom property name (without d_ prefix)
 *
 *  At least one of the following keyword/arg pairs is required.
 *  Only one (or none) load balancer is used in a load balancing step.
 *
 *      global args = trigger_threshold bin_width (use histogram balancing; only for staggered)
 *          trigger_threshold = float
 *              float = apply load balancer if the imbalance of the load is above this threshold
 *          bin_width = approximate width of bin (arbitrary positive value)
 *      local args = trigger_threshold (use local balancing)
 *          trigger_threshold = float
 *              float = apply load balancer if the imbalance of the load is above this threshold
 *
 *  optional keyword arg pairs
 *
 *      verbose args = none (verbose output to log/screen)
 */

FixBalanceAll::FixBalanceAll(LAMMPS *lmp, int narg, char **arg) :
  Fix(lmp, narg, arg), all_local(nullptr), irregular(nullptr), all_global(nullptr)
{
  if (narg < 6) error->all(FLERR,"Illegal fix balance/all command");

  imbalances = nullptr;
  fixstore = nullptr;

  box_change = BOX_CHANGE_DOMAIN;
  pre_exchange_migrate = 1;

  vector_flag = 1;
  size_vector = 6;
  extvector = 0;

  // parse required arguments

  if (domain->triclinic) error->all(FLERR,"triclinic domains are not yet supported by ALL");

  // set defaul values

  gridstyle = UNKNOWN_GRID;
  int tensorstyle = NONE;
  nevery = -1;
  bw = -1;
  local_threshold = -1;
  global_threshold = -1;
  use_local_lb = false;
  use_global_lb = false;
  verbose = false;
  wtflag = 0;
  varflag = 0;

  // count max number of weight settings

  nimbalance = 0;
  for (int i = 3; i < narg; i++)
    if (strcmp(arg[i],"weight") == 0) nimbalance++;
  if (nimbalance) imbalances = new Imbalance*[nimbalance];
  nimbalance = 0;

  // parse arguments
  for (int iarg = 3; iarg < narg; ++iarg) {
    if (strcmp(arg[iarg],"every") == 0) {
      if (iarg+1 >= narg) error->all(FLERR, "balance/all: every requires an integer");
      nevery = utils::inumeric(FLERR,arg[iarg+1],false,lmp);
      iarg++;
    } else if (strcmp(arg[iarg],"grid") == 0) {
      if (iarg+1 >= narg) error->all(FLERR, "balance/all: grid requires one argument");
      if (gridstyle != UNKNOWN_GRID) error->all(FLERR, "balance/all: multiple grid styles defined");
      if (strcmp(arg[iarg+1],"staggered") == 0) gridstyle = STAGGERED;
      else if (strcmp(arg[iarg+1],"tensor") == 0) {
        gridstyle = TENSOR;
        if (iarg+2 >= narg) error->all(FLERR, "balance/all: grid tensor requires one argument");
        if (strcmp(arg[iarg+2],"classic") == 0) tensorstyle = TENSOR_CLASSIC;
        else if (strcmp(arg[iarg+2],"max") == 0) tensorstyle = TENSOR_MAX;
        else error->all(FLERR, "balance/all: unknown grid tensor argument {}", arg[iarg+2]);
        iarg++;
      } else error->all(FLERR, "balance/all: unknown grid argument {}", arg[iarg+1]);
      iarg++;
    } else if (strcmp(arg[iarg],"weight") == 0) {
      wtflag = 1;
      Imbalance *imb;
      int nopt = 0;
      if (strcmp(arg[iarg+1],"group") == 0) {
        imb = new ImbalanceGroup(lmp);
        nopt = imb->options(narg-iarg,arg+iarg+2);
        imbalances[nimbalance++] = imb;
      } else if (strcmp(arg[iarg+1],"time") == 0) {
        imb = new ImbalanceTime(lmp);
        nopt = imb->options(narg-iarg,arg+iarg+2);
        imbalances[nimbalance++] = imb;
      } else if (strcmp(arg[iarg+1],"neigh") == 0) {
        imb = new ImbalanceNeigh(lmp);
        nopt = imb->options(narg-iarg,arg+iarg+2);
        imbalances[nimbalance++] = imb;
      } else if (strcmp(arg[iarg+1],"var") == 0) {
        varflag = 1;
        imb = new ImbalanceVar(lmp);
        nopt = imb->options(narg-iarg,arg+iarg+2);
        imbalances[nimbalance++] = imb;
      } else if (strcmp(arg[iarg+1],"store") == 0) {
        imb = new ImbalanceStore(lmp);
        nopt = imb->options(narg-iarg,arg+iarg+2);
        imbalances[nimbalance++] = imb;
      } else {
        error->all(FLERR,"Unknown fix balance/all weight method: {}", arg[iarg+1]);
      }
      iarg += 1+nopt;
    } else if (strcmp(arg[iarg],"verbose") == 0) {
      verbose = true;
    } else if (strcmp(arg[iarg],"global") == 0) {
      if (iarg+2 >= narg) error->all(FLERR, "balance/all: histogram requires two arguments");
      use_global_lb = true;
      global_threshold = utils::numeric(FLERR,arg[iarg+1],false,lmp);
      bw = utils::numeric(FLERR,arg[iarg+2],false,lmp);
      iarg += 2;
    } else if (strcmp(arg[iarg],"local") == 0) {
      if (iarg+1 >= narg) error->all(FLERR, "balance/all: threshold requires one argument");
      use_local_lb = true;
      local_threshold = utils::numeric(FLERR,arg[iarg+1],false,lmp);
      iarg++;
    } else error->all(FLERR,"balance/all: unknown argument '{}'", arg[iarg]);
  }

  // check arguments
  if (nevery < 1) error->all(FLERR,"balance/all: nevery > 0 required");
  if (gridstyle == UNKNOWN_GRID) error->all(FLERR, "balance/all: grid style required");
  if (use_global_lb) {
    if (gridstyle != STAGGERED) error->all(FLERR, "balance/all: histogram requires staggered grid");
    if (bw <= 0) error->all(FLERR, "balance/all: bw not positive");
  }
  if (! (use_local_lb || use_global_lb)) error->all(FLERR, "balance/all: at least local, global required");
  if (use_local_lb && use_global_lb) {
    if (local_threshold >= global_threshold) error->all(FLERR, "balance/all: local threshold < global threshold required");
  }

  // set comm style dependent on gridstyle ? see Input::comm_style()
  // probably not since there is no running simulation in input (in contrast to here)
  if (gridstyle == STAGGERED && comm->style != 2)
    error->all(FLERR,"balance/all: comm_style staggered required for staggered grid");
  if (gridstyle == TENSOR && comm->style != 0)
    error->all(FLERR,"balance/all: comm_style brick required for tensor grid");

  if (gridstyle == STAGGERED) {
    all_local = new ALL::ALL<double, double>(ALL::STAGGERED, 3, 0);
  } else if (gridstyle == TENSOR) {
    if (tensorstyle == TENSOR_MAX) all_local = new ALL::ALL<double, double>(ALL::TENSOR_MAX, 3, 0);
    else if (tensorstyle == TENSOR_CLASSIC) all_local = new ALL::ALL<double, double>(ALL::TENSOR, 3, 0);
  }

  if (use_global_lb) {
    all_global = new ALL::ALL<double, double>(ALL::HISTOGRAM, 3, 0);
    n_bins = std::vector<int>(3, -1);
  }

  all_last = nullptr;

  global_freq = 1; // nevery;

  irregular = new Irregular(lmp);

  x_masters = MPI_COMM_NULL;
  y_masters = MPI_COMM_NULL;
  z_masters = MPI_COMM_NULL;

  timer_lb = imbalance = maxloadperproc = -1;

  force_reneighbor = 1;
  lastbalance = -1;
  next_reneighbor = -1;

  reduce_outvec_flag = false;
  for (int i = 0; i < 3; ++i) outvec_timer[i] = -1;
}

/**
 *  Deconstructor. Free communicators and delete allocated memory.
 */

FixBalanceAll::~FixBalanceAll()
{
  all_last = nullptr;
  if (all_local) delete all_local;
  if (all_global) delete all_global;
  delete irregular;

  if (x_masters != MPI_COMM_NULL) MPI_Comm_free(&x_masters);
  if (y_masters != MPI_COMM_NULL) MPI_Comm_free(&y_masters);
  if (z_masters != MPI_COMM_NULL) MPI_Comm_free(&z_masters);

  for (int i = 0; i < nimbalance; i++) delete imbalances[i];
  delete[] imbalances;

  // check nfix in case all fixes have already been deleted
  if (fixstore && modify->nfix) modify->delete_fix(fixstore->id);
  fixstore = nullptr;

}

/**
 *  Post constructor. Setup weight storage.
 */

void FixBalanceAll::post_constructor()
{
  if (wtflag) weight_storage();
}

/**
 *  For lammps. This fix is just called in pre_exchange.
 *  @return mask
 */

int FixBalanceAll::setmask()
{
  int mask = 0;
  mask |= PRE_EXCHANGE;
  return mask;
}

/**
 *  Initialise calculated variables and setup ALL objects.
 *  Setup initial grid.
 *  Write information to log.
 */

void FixBalanceAll::init()
{
  // called before every run

  int counter = 0;
  for (int i=0; i<modify->nfix; i++) {
    if (strcmp(modify->fix[i]->style, "balance") == 0) counter++;
    if (strcmp(modify->fix[i]->style, "balance/all") == 0) counter++;
  }
  if (counter > 1) error->all(FLERR, "More than one dynamic load balancing fix");

  // get vectors as input for ALL

  // use existing uniform grid
  if (comm->layout == Comm::LAYOUT_TILED) error->all(FLERR, "balance/all: initialisation from Comm::LAYOUT_TILED not possible.");
  if (comm->layout == Comm::LAYOUT_STAGGERED && gridstyle == TENSOR) error->all(FLERR, "balance/all: initialisation from Comm::STAGGERED not possible.");
  if (gridstyle == STAGGERED && comm->style == 2 && (comm->staggered2spatial[0]!=2 || comm->staggered2spatial[1]!=1 || comm->staggered2spatial[2]!=0))
    error->all(FLERR, "balance/all: comm_style staggered not zyx");
  if (domain->dimension == 2 && comm->procgrid[2] != 1) error->all(FLERR,"balance/all: 2D-simulation not possible with {} processors in z-direction", comm->procgrid[2]);
  procgrid_vec.assign(comm->procgrid, comm->procgrid+3);
  myloc_vec.assign(comm->myloc, comm->myloc+3);

  minimum_domain_size = {neighbor->skin, neighbor->skin, neighbor->skin};

  // setup ALL

  if (use_local_lb) {
    all_local->setProcGridParams(myloc_vec, procgrid_vec);
    all_local->setMinDomainSize(minimum_domain_size);
    all_local->setCommunicator(world);
    all_local->setProcTag(comm->me);
    all_local->setup();
  }

  if (use_global_lb) {
    all_global->setProcGridParams(myloc_vec, procgrid_vec);
    all_global->setMinDomainSize(minimum_domain_size);
    all_global->setCommunicator(world);
    all_global->setProcTag(comm->me);
    all_global->setup();
  }

  init_imbalance(1);

  // create communicators for tensor
  if (gridstyle == TENSOR) {
    int color;
    // There would be a way to create the communicator without communication, but split is easier to implement.

    if (x_masters != MPI_COMM_NULL) MPI_Comm_free(&x_masters);
    color = (myloc_vec[1] == 0 && myloc_vec[2] == 0) ? 0 : MPI_UNDEFINED;
    MPI_Comm_split(world, color, myloc_vec[0], &x_masters);

    if (y_masters != MPI_COMM_NULL) MPI_Comm_free(&y_masters);
    color = (myloc_vec[0] == 0 && myloc_vec[2] == 0) ? 0 : MPI_UNDEFINED;
    MPI_Comm_split(world, color, myloc_vec[1], &y_masters);

    if (z_masters != MPI_COMM_NULL) MPI_Comm_free(&z_masters);
    color = (myloc_vec[0] == 0 && myloc_vec[1] == 0) ? 0 : MPI_UNDEFINED;
    MPI_Comm_split(world, color, myloc_vec[1], &z_masters);
  }

  next_reneighbor = (update->ntimestep/nevery)*nevery + nevery;

  if (comm->me == 0) {
    utils::logmesg(lmp, "ALL information ...\n");
    utils::logmesg(lmp, "\tversion: 0.9.3\n");
    if (gridstyle == STAGGERED) utils::logmesg(lmp, "\tgrid: staggered\n");
    else if (gridstyle == TENSOR) utils::logmesg(lmp, "\tgrid: tensor\n");
    utils::logmesg(lmp, "\tnumber of processors: {} x {} y {} z\n", procgrid_vec[0], procgrid_vec[1], procgrid_vec[2]);
    if (use_global_lb) {
      utils::logmesg(lmp, "\tglobal load balancing ...\n");
      utils::logmesg(lmp, "\t\ttrigger threshold: {}\n", global_threshold);
      utils::logmesg(lmp, "\t\tbin width: {}\n", bw);
    }
    if (use_local_lb) {
      utils::logmesg(lmp, "\tlocal load balancing ...\n");
      utils::logmesg(lmp, "\t\ttrigger threshold: {}\n", local_threshold);
    }
    if (verbose) utils::logmesg(lmp, "\tverbose output is used\n");
  }
}

/**
 *  Perform dynamic load balancing if required.
 *  Calls corresponding local or global balancing function.
 */

void FixBalanceAll::pre_exchange()
{
  // return if not a rebalance timestep

  if (update->ntimestep < next_reneighbor) return;

  // next timestep to rebalance
  next_reneighbor = (update->ntimestep/nevery)*nevery + nevery;

  balance();
}

/**
 *  Perform load balancing.
 *  Calls corresponding local or global balancing function.
 */

void FixBalanceAll::balance()
{
  // do not allow rebalancing twice on same timestep
  // even if you wanted to, it can mess up elapsed time

  if (update->ntimestep == lastbalance) return;
  lastbalance = update->ntimestep;

  double timer_lb_start = platform::walltime();

  set_weights();

  work = get_work();

  calc_imbalance();

  // call load-balancing function if required
  if (use_global_lb && imbalance > global_threshold)
    balance_global();
  else if (use_local_lb && imbalance > local_threshold)
    balance_local();
  else {
    if (verbose && comm->me == 0) utils::logmesg(lmp, "balance/all: {} no balance required\n", imbalance);
    all_last = nullptr;
  }
  unset_weights();

  timer_lb = platform::walltime() - timer_lb_start;
  reduce_outvec_flag = true;
}

/**
 *  Perform staggered/tensor load balancing.
 */

void FixBalanceAll::balance_local()
{
  if (verbose && comm->me == 0) utils::logmesg(lmp, "balance/all: {} balance local\n", imbalance);

  all_last = all_local;

  // ensure atoms are in current box & update box via shrink-wrap
  // no exchange() since doesn't matter if atoms are assigned to correct procs

  domain->pbc();
  domain->reset_box();

  // The domain size is changed by adjusting the domain specific variables in comm.h
  // local box is set with
  // comm->{xsplit ysplit zsplit myloc procgrid} for tensor
  // comm->mysplit                               for staggered

  // rebalance with ALL
  my_vertices = get_comm_vertices();
  all_local->setVertices(my_vertices);
  all_local->setWork(work);
  all_local->balance();

  set_comm_vertices(all_local->getVertices());
  domain->set_local_box();

  // not required since the minimum box size is also passed to all
  // domain->subbox_too_small_check(neighbor->skin);

  if (gridstyle == STAGGERED) irregular->migrate_atoms();
  else if (irregular->migrate_check()) irregular->migrate_atoms();

  modify->reset_grid();
  if (force->pair) force->pair->reset_grid();
  if (force->kspace) force->kspace->reset_grid();
}

/**
 *  Perform histogram load balancing in z, y and x.
 */

void FixBalanceAll::balance_global()
{
  if (verbose && comm->me == 0) utils::logmesg(lmp, "balance/all: {} balance global\n", imbalance);

  all_last = all_global;

  // only for staggered
  // histogram method

  // insure atoms are in current box & update box via shrink-wrap
  // no exchange() since doesn't matter if atoms are assigned to correct procs

  domain->pbc();
  domain->reset_box();

  // processors may not have complex/simple particles yet, but get some during layer-balancing
  // -> calculate global average
  // only once since timers are evaluated

  // atoms should be inside of the boundaries for the histogram calculation
  irregular->migrate_atoms();

  for (int dim_balance=2; dim_balance>=0; dim_balance--) {
    // rebalance with ALL
    my_vertices = get_comm_vertices();
    all_global->setVertices(my_vertices);
    all_global->setSysSize(get_sys_size_from_domain());

    all_global->setWork(calc_histogram(dim_balance));
    all_global->setMethodData(n_bins.data());

    all_global->balance();
    set_comm_vertices(all_global->getVertices());

    domain->set_local_box();

    irregular->migrate_atoms();
  }

  modify->reset_grid();
  if (force->pair) force->pair->reset_grid();
  if (force->kspace) force->kspace->reset_grid();

}

/**
 *  Get the size of the simulation box from the domain class.
 *  @return box size in box units with box origin in 0,0,0
 */

std::vector<double> FixBalanceAll::get_sys_size_from_domain()
{
  std::vector<double> sys_size(6);
  sys_size.at(0) = 0;
  sys_size.at(1) = domain->xprd;
  sys_size.at(2) = 0;
  sys_size.at(3) = domain->yprd;
  sys_size.at(4) = 0;
  sys_size.at(5) = domain->zprd;
  return sys_size;
}

/**
 *  Get size of this domain from class domain for setVertices of ALL.
 *  @return size of this domain in ALL format
 */

std::vector<ALL::Point<double>> FixBalanceAll::get_comm_vertices()
{
  std::vector<ALL::Point<double>> vertices(2, ALL::Point<double>(3));
  // comm stores the values in reduced coordinates in [0,1]
  // -> multiply with box length per dimension
  if (comm->layout == Comm::LAYOUT_STAGGERED) {
    // tiled
    vertices[0][0] = comm->mysplit[0][0] * domain->prd[0];
    vertices[0][1] = comm->mysplit[1][0] * domain->prd[1];
    vertices[0][2] = comm->mysplit[2][0] * domain->prd[2];
    vertices[1][0] = comm->mysplit[0][1] * domain->prd[0];
    vertices[1][1] = comm->mysplit[1][1] * domain->prd[1];
    vertices[1][2] = comm->mysplit[2][1] * domain->prd[2];
  } else {
    // uniform
    vertices[0][0] = comm->xsplit[comm->myloc[0]] * domain->prd[0];
    vertices[0][1] = comm->ysplit[comm->myloc[1]] * domain->prd[1];
    vertices[0][2] = comm->zsplit[comm->myloc[2]] * domain->prd[2];
    vertices[1][0] = comm->xsplit[comm->myloc[0]+1] * domain->prd[0];
    vertices[1][1] = comm->ysplit[comm->myloc[1]+1] * domain->prd[1];
    vertices[1][2] = comm->zsplit[comm->myloc[2]+1] * domain->prd[2];
  }

  return vertices;
}

/**
 *  Get load of this domain for imbalance calculation and setWork of
 *  local ALL object.
 *  @return work according to used definition
 */

double FixBalanceAll::get_work()
{
  double work = 0.0;

  if (wtflag) {
    weight = fixstore->vstore;
    int nlocal = atom->nlocal;
    for (int i = 0; i < nlocal; i++)
      work += weight[i];

  } else {
    work = atom->nlocal;
  }

  return work;
}

/**
 *  Set domain boundaries in class comm for tiled or brick layout.
 *  @param[in] vertices returned by ALL object after load balancing step
 *  @note updates layout, staggerednew, mysplit for staggered
 *  @note updates xsplit, ysplit, zsplit for tensor
 */

void FixBalanceAll::set_comm_vertices(std::vector<ALL::Point<double>> vertices)
{
  if (gridstyle == STAGGERED) {
    // set tiled values
    comm->layout = Comm::LAYOUT_STAGGERED;
    // comm stores the values in reduced coordinates in [0,1]
    // -> divide by box length
    comm->staggerednew = 1;

    // just get the vertices
    comm->mysplit[0][0] = vertices[0][0] / domain->prd[0];
    comm->mysplit[1][0] = vertices[0][1] / domain->prd[1];
    comm->mysplit[2][0] = vertices[0][2] / domain->prd[2];
    comm->mysplit[0][1] = vertices[1][0] / domain->prd[0];
    comm->mysplit[1][1] = vertices[1][1] / domain->prd[1];
    comm->mysplit[2][1] = vertices[1][2] / domain->prd[2];

    // prevention of floating point issues
    for (int idim=0; idim<3; idim++) {
      if (myloc_vec[idim] == 0) comm->mysplit[idim][0] = 0;
      if (myloc_vec[idim] == procgrid_vec[idim]-1) comm->mysplit[idim][1] = 1;
    }

  } else {
    // gridstyle == TENSOR

    // rescale to [0:1]
    vertices[1][0] /= domain->prd[0];
    vertices[1][1] /= domain->prd[1];
    vertices[1][2] /= domain->prd[2];

    // xsplit ysplit zsplit contain all values
    // split[0] is not changed, but it is zero anyway
    // gather all upper boundaries on master
    if (x_masters != MPI_COMM_NULL) MPI_Gather(&(vertices[1][0]), 1, MPI_DOUBLE, comm->xsplit+1, 1, MPI_DOUBLE, 0, x_masters);
    if (y_masters != MPI_COMM_NULL) MPI_Gather(&(vertices[1][1]), 1, MPI_DOUBLE, comm->ysplit+1, 1, MPI_DOUBLE, 0, y_masters);
    if (z_masters != MPI_COMM_NULL) MPI_Gather(&(vertices[1][2]), 1, MPI_DOUBLE, comm->zsplit+1, 1, MPI_DOUBLE, 0, z_masters);

    // broadcast all boundaries from master
    MPI_Bcast(comm->xsplit, comm->procgrid[0]+1, MPI_DOUBLE, 0, world);
    MPI_Bcast(comm->ysplit, comm->procgrid[1]+1, MPI_DOUBLE, 0, world);
    MPI_Bcast(comm->zsplit, comm->procgrid[2]+1, MPI_DOUBLE, 0, world);

  }
}

/**
 *  Calculate imbalance based on the current scalar work.
 *  @note updates imbalance
 */

void FixBalanceAll::calc_imbalance()
{
  double max, avg;
  MPI_Allreduce(&work, &avg, 1, MPI_DOUBLE, MPI_SUM, world);
  avg /= comm->nprocs;
  MPI_Allreduce(&work, &max, 1, MPI_DOUBLE, MPI_MAX, world);

  maxloadperproc = max;

  if ((max < 0 || avg < 0 || max < avg) && comm->me == 0) error->warning(FLERR, "cannot calculate imbalance with max={} avg={}", max, avg);

  if (max == 0) {
    imbalance = -1;
  } else {
    imbalance = max / avg;
  }
}

/**
 * invoke init() for each Imbalance class
 * flag = 0 for call from Balance, 1 for call from FixBalance
 */

void FixBalanceAll::init_imbalance(int flag = 1)
{
  if (!wtflag) return;
  for (int n = 0; n < nimbalance; n++) imbalances[n]->init(flag);
}

/**
 *  allocate per-particle weight storage for histogram via FixStoreAtom
 *  fix could already be allocated if fix balance is re-specified
 */

void FixBalanceAll::weight_storage()
{
  std::string cmd;
  cmd = id;
  cmd += "HISTOGRAM_WEIGHTS";
  fixstore = dynamic_cast<FixStoreAtom *>(modify->get_fix_by_id(cmd));
  if (!fixstore) fixstore = dynamic_cast<FixStoreAtom *>(modify->add_fix(cmd + " all STORE/ATOM 1 0 0 0"));

  // do not carry weights with atoms during normal atom migration
  fixstore->disable = 1;
}

/**
  *  set weight for each particle
  */

void FixBalanceAll::set_weights()
{
  if (!wtflag) return;
  weight = fixstore->vstore;

  int nlocal = atom->nlocal;
  for (int i = 0; i < nlocal; i++) weight[i] = 1.0;
  for (int n = 0; n < nimbalance; n++) imbalances[n]->compute(weight);

  // weights need to migrate with atoms
  fixstore->disable = 0;
}

/**
  *  prevent further migration of weights
  */

void FixBalanceAll::unset_weights()
{
  if (!wtflag) return;

  // weights should not migrate with atoms
  fixstore->disable = 1;
}

/**
 *  Calculate histogram for setWork of histogram balancing.
 *  @note updates n_bins
 *  @param[in] dimension in which the histogram is calculated
 *  @return histogram for setWork
 */

std::vector<double> FixBalanceAll::calc_histogram(int dimension)
{
  // calculate the histogram width in a way that lower and upper box boundary match with a bin boundary
  int n_bins_global = std::ceil(domain->prd[dimension] / bw);
  const double bin_width = domain->prd[dimension] / n_bins_global;

  double lb = std::ceil(my_vertices[0][dimension] / bin_width) * bin_width;
  double ub = std::ceil(my_vertices[1][dimension] / bin_width) * bin_width;

  double overlap = 0; // bin -1 which is send to lower neighbour

  n_bins.at(dimension) = (int) (std::round((ub -lb) / bin_width) + 1e-4);

  std::vector<double> work_vec(n_bins.at(dimension), 0.0);

  // compute histogram of work load
  double ** x = atom->x;

  // the work per atom is constant
  // -> set work before iterating over atoms
  double work_atom = 1.0;
  double *weight = nullptr;
  if (wtflag) weight = fixstore->vstore;

  for (int i = 0; i < atom->nlocal; i++) {

    // calculate bin of atom
    const int idx = std::floor(((x[i][dimension] - domain->boxlo[dimension] - lb) / bin_width));

    // use individual weight or previously defined work
    if (wtflag) work_atom = weight[i];

    // update corresponding bin
    if (idx >= 0 && idx < n_bins.at(dimension)) {
      work_vec.at(idx) += work_atom;
    } else if (idx == -1) {
      overlap += work_atom;
    } else if (idx == n_bins.at(dimension) && fabs(x[i][dimension] - my_vertices[1][dimension]) < 1e-6) {
      // floating point issue, just use the last bin
      work_vec.at(n_bins.at(dimension) - 1) += work_atom;
    } else {
      error->one(FLERR, "balance/all: unexpected histogram bin {} for histogram of size {} x {} lb {} bin_width {} boxlo {} myvert_lo {} myvert_hi {}", idx, n_bins.at(dimension), x[i][dimension], lb, bin_width, domain->boxlo[dimension], my_vertices[0][dimension], my_vertices[1][dimension]);
    }
  }

  // calculate ranks of neighbours
  //myloc_vec.assign(comm->myloc, comm->myloc+3);
  int rank_left = MPI_PROC_NULL;
  int rank_right = MPI_PROC_NULL;
  int loc_ngh[3];
  loc_ngh[0] = myloc_vec[0];
  loc_ngh[1] = myloc_vec[1];
  loc_ngh[2] = myloc_vec[2];
  // left neighbour
  loc_ngh[dimension] -= 1;
  if (loc_ngh[dimension] >= 0)
    rank_left = comm->grid2proc[loc_ngh[0]][loc_ngh[1]][loc_ngh[2]];
  // right neighbour
  loc_ngh[dimension] += 2;
  if (loc_ngh[dimension] < procgrid_vec[dimension])
    rank_right = comm->grid2proc[loc_ngh[0]][loc_ngh[1]][loc_ngh[2]];

  // exchange overlapping workload (histograms might overlap
  // over the domain boundaries

  MPI_Request sreq, rreq;
  MPI_Status ssta, rsta;

  double recv_work = 0;

  MPI_Isend(&overlap, 1, MPI_DOUBLE, rank_left, 0, world, &sreq);
  MPI_Irecv(&recv_work, 1, MPI_DOUBLE, rank_right, 0, world, &rreq);
  MPI_Wait(&rreq, &rsta);
  MPI_Wait(&sreq, &ssta);

  work_vec.at(n_bins.at(dimension) - 1) += recv_work;

  return work_vec;
}

/**
 *  For lammps output only.
 *  @param[in] i index of output vector
 *  @return requested value
 */

double FixBalanceAll::compute_vector(int i)
{
  // TODO: remove communication and lb timing (doing more than required)
  if (/*remove*/reduce_outvec_flag/*elsewhere*/) {
    double reducebuffer_s, reducebuffer_r;
    reducebuffer_s = /*remove!*/timer_lb/*elsewhere*/;

    // calc min
    MPI_Allreduce(&reducebuffer_s, &reducebuffer_r, 1, MPI_DOUBLE, MPI_MIN, world);
    outvec_timer[0] = reducebuffer_r;

    // calc avg
    MPI_Allreduce(&reducebuffer_s, &reducebuffer_r, 1, MPI_DOUBLE, MPI_SUM, world);
    outvec_timer[1] = reducebuffer_r / comm->nprocs;

    // calc max
    MPI_Allreduce(&reducebuffer_s, &reducebuffer_r, 1, MPI_DOUBLE, MPI_MAX, world);
    outvec_timer[2] = reducebuffer_r;

    reduce_outvec_flag = false;
  }

  if (i == 0) return maxloadperproc;
  if (i == 1) return imbalance;
  if (i <= 4) return outvec_timer[i-3];
  return -1;
}

/**
 *  For lammps stats only.
 *  @return # of bytes of allocated memory
 */

double FixBalanceAll::memory_usage()
{
  double bytes = irregular->memory_usage();
  return bytes;
}

// TODO: remove
//
///**
// *  Write the given histogram to standard output.
// *  @param[in] work_vec histogram to be printed
// */
//
//void FixBalanceAll::print_histogram(std::vector<double> work_vec)
//{
//  printf("[%i]: histogram", comm->me);
//  for (double w : work_vec) printf(" %f", w);
//  printf("\n");
//}
//
///**
// *  Print domain boundaries to standard output.
// *  @note needs to be called by all ranks
// */
//
//void FixBalanceAll::print_domains()
//{
//  if (comm->me == 0) {
//    if (comm->layout == Comm::LAYOUT_TILED) {
//      printf("LBSTAT LAYOUT_TILED\n");
//    } else {
//      printf("LBSTAT LAYOUT_TENSOR\n");
//      printf("LBSTAT procgrid %i %i %i\n", comm->procgrid[0], comm->procgrid[1], comm->procgrid[1]);
//      printf("LBSTAT user_procgrid %i %i %i\n", comm->user_procgrid[0], comm->user_procgrid[1], comm->user_procgrid[1]);
//      printf("LBSTAT xsplit %f", comm->xsplit[0]);
//      for (int i_proc=1; i_proc<=comm->procgrid[0]; i_proc++)
//        printf(" %f", comm->xsplit[i_proc]);
//      printf("\nLBSTAT ysplit %f", comm->ysplit[0]);
//      for (int i_proc=1; i_proc<=comm->procgrid[1]; i_proc++)
//        printf(" %f", comm->ysplit[i_proc]);
//      printf("\nLBSTAT zsplit %f", comm->zsplit[0]);
//      for (int i_proc=1; i_proc<=comm->procgrid[2]; i_proc++)
//        printf(" %f", comm->zsplit[i_proc]);
//      printf("\n");
//    }
//  }
//
//  for (int i_proc=0; i_proc<comm->nprocs; i_proc++) {
//    MPI_Barrier(world);
//    if (i_proc != comm->me) { continue; }
//    printf("LBSTAT processor %i\n", i_proc);
//
//    if (comm->layout == Comm::LAYOUT_TILED) {
//    // public settings specific to layout = TILED
//
//      printf("LBSTAT rcbnew %i\n", comm->rcbnew);
//      printf("LBSTAT rcbcutdim %i\n", comm->rcbcutdim);
//      printf("LBSTAT rcbcutfrac %f\n", comm->rcbcutfrac);
//      for (int i_dim=0; i_dim<3; i_dim++)
//        printf("LBSTAT mysplit %i %f %f\n", i_dim, comm->mysplit[i_dim][0], comm->mysplit[i_dim][1]);
//
//    } else {
//      // public settings specific to layout = UNIFORM, NONUNIFORM
//
//      printf("LBSTAT myloc %i %i %i\n", comm->myloc[0], comm->myloc[1], comm->myloc[2]);
//      for (int i_dim=0; i_dim<3; i_dim++)
//        printf("LBSTAT procneigh %i %i %i\n", i_dim, comm->procneigh[i_dim][0], comm->procneigh[i_dim][1]);
//
//    }
//  }
//}
//
