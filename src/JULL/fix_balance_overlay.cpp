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

#include "fix_balance_overlay.h"

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
enum { NATOMS, TIME, UNKNOWN_WORK, RANK };
enum { TENSOR_MAX, TENSOR_CLASSIC, NONE };

// clang-format off

/**
 *  create class and parse arguments in LAMMPS script.
 *  Syntax:
 *
 *  fix ID group-ID balance/overlay keyword args ...
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
 *      load style = define used model for the load
 *          style = time or natoms or rank
 *              time = use measured force-calculation times for balancing
 *              natoms = use number of particles per processor for balancing
 *              rank = use rank as weight per particle (for debugging)
 *
 *  At least one of the following keyword/arg pairs is required.
 *  Only one (or none) load balancer is used in a load balancing step.
 *
 *      global args = trigger_threshold bin_width rescale_histogram (use histogram balancing; only for staggered)
 *          trigger_threshold = float
 *              float = apply load balancer if the imbalance of the load is above this threshold
 *          bin_width = approximate width of bin (arbitrary positive value)
 *          rescale_histogram = true or false (rescale histogram to scalar work)
 *      local args = trigger_threshold (use local balancing)
 *          trigger_threshold = float
 *              float = apply load balancer if the imbalance of the load is above this threshold
 *
 *  optional keyword arg pairs
 *
 *      verbose args = none (verbose output to log/screen)
 *      file arg = filename
 *          filename = write stats per rank to filename
 */

FixBalanceOverlay::FixBalanceOverlay(LAMMPS *lmp, int narg, char **arg) :
  Fix(lmp, narg, arg), jull_local(nullptr), irregular(nullptr), jull_global(nullptr)
{
#ifdef DEBUG_COMM_STAGGERED
  char logfile[50];
  sprintf(logfile, "scratch/balance_overlay_%i.log", comm->me);
  fp = fopen(logfile, "w");
  fprintf(fp, "new log proc %i\n", comm->me);
  fflush(fp);
#endif
  if (narg < 6) error->all(FLERR,"Illegal fix balance/overlay command");

  fixstore = nullptr;

  box_change = BOX_CHANGE_DOMAIN;
  pre_exchange_migrate = 1;

  vector_flag = 1;
  size_vector = 6;
  extvector = 0;

  // parse required arguments

  if (domain->triclinic) error->all(FLERR,"triclinic domains are not supported by ALL");

  jull_timer = nullptr;

  // set defaul values

  gridstyle = UNKNOWN_GRID;
  workstyle = UNKNOWN_WORK;
  int tensorstyle = NONE;
  nevery = -1;
  bw = -1;
  local_threshold = -1;
  global_threshold = -1;
  use_local_lb = false;
  use_global_lb = false;
  verbose = false;
  rescale_histogram = false;
  wtflag = 0;
  fh = MPI_FILE_NULL;
  nevery_file_write = -1;

  // parse arguments
  for (int iarg=3; iarg<narg; iarg++) {
    if (strcmp(arg[iarg],"every") == 0) {
      if (iarg+1 >= narg) error->all(FLERR, "balance/overlay: every requires an integer");
      nevery = utils::inumeric(FLERR,arg[iarg+1],false,lmp);
      iarg++;
    } else if (strcmp(arg[iarg],"grid") == 0) {
      if (iarg+1 >= narg) error->all(FLERR, "balance/overlay: grid requires one argument");
      if (gridstyle != UNKNOWN_GRID) error->all(FLERR, "balance/overlay: multiple grid styles defined");
      if (strcmp(arg[iarg+1],"staggered") == 0) gridstyle = STAGGERED;
      else if (strcmp(arg[iarg+1],"tensor") == 0) {
        gridstyle = TENSOR;
        if (iarg+2 >= narg) error->all(FLERR, "balance/overlay: grid tensor requires one argument");
        if (strcmp(arg[iarg+2],"classic") == 0) tensorstyle = TENSOR_CLASSIC;
        else if (strcmp(arg[iarg+2],"max") == 0) tensorstyle = TENSOR_MAX;
        else error->all(FLERR, "balance/overlay: unknown grid tensor argument {}", arg[iarg+2]);
        iarg++;
      } else error->all(FLERR, "balance/overlay: unknown grid argument {}", arg[iarg+1]);
      iarg++;
    } else if (strcmp(arg[iarg],"load") == 0) {
      if (iarg+1 >= narg) error->all(FLERR, "balance/overlay: load requires one argument");
      iarg++;
      if (workstyle != UNKNOWN_WORK) error->all(FLERR, "balance/overlay: multiple work styles defined");
      if (strcmp(arg[iarg],"time") == 0) {
        workstyle = TIME;
        jull_timer = new JullTimer(lmp);
      } else if (strcmp(arg[iarg],"natoms") == 0) {
        workstyle = NATOMS;
      } else if (strcmp(arg[iarg],"rank") == 0) {
        workstyle = RANK;
      }
    } else if (strcmp(arg[iarg],"verbose") == 0) {
      verbose = true;
    } else if (strcmp(arg[iarg],"global") == 0) {
      if (iarg+3 >= narg) error->all(FLERR, "balance/overlay: histogram requires three arguments");
      use_global_lb = true;
      global_threshold = utils::numeric(FLERR,arg[iarg+1],false,lmp);
      bw = utils::numeric(FLERR,arg[iarg+2],false,lmp);
      if (strcmp(arg[iarg+3],"false") == 0) rescale_histogram = false;
      else if (strcmp(arg[iarg+3],"true") == 0) rescale_histogram = true;
      else error->all(FLERR, "balance/overlay: expected true or false instead of {} for rescale_histogram", arg[iarg+3]);
      iarg += 3;
    } else if (strcmp(arg[iarg],"local") == 0) {
      if (iarg+1 >= narg) error->all(FLERR, "balance/overlay: threshold requires one argument");
      use_local_lb = true;
      local_threshold = utils::numeric(FLERR,arg[iarg+1],false,lmp);
      iarg++;
    } else if (strcmp(arg[iarg],"file") == 0) {
      if (iarg+2 >= narg) error->all(FLERR, "balance/overlay: file requires two arguments");
      if (MPI_File_open(world, arg[iarg+1], MPI_MODE_WRONLY | MPI_MODE_CREATE, MPI_INFO_NULL, &fh))
        error->one(FLERR, "balance/overlay open {} failed", arg[iarg+1]);
      nevery_file_write = utils::inumeric(FLERR,arg[iarg+2],false,lmp);
      if (nevery_file_write < 1) error->all(FLERR, "balance/overlay: nevery_file_write < 1");
      iarg += 2;
    } else error->all(FLERR,"balance/overlay: unknown argument {}", arg[iarg]);
  }

  // check arguments
  if (nevery < 1) error->all(FLERR,"balance/overlay: nevery > 0 required");
  if (gridstyle == UNKNOWN_GRID) error->all(FLERR, "balance/overlay: grid style required");
  if (workstyle == UNKNOWN_WORK) error->all(FLERR, "balance/overlay: work style required");
  if (use_global_lb) {
    if (workstyle == TIME) wtflag = 1;
    if (gridstyle != STAGGERED) error->all(FLERR, "balance/overlay: histogram requires staggered grid");
    if (bw <= 0) error->all(FLERR, "balance/overlay: bw not positive");
  }
  if (! (use_local_lb || use_global_lb)) error->all(FLERR, "balance/overlay: at least local, global required");
  if (use_local_lb && use_global_lb) {
    if (local_threshold >= global_threshold) error->all(FLERR, "balance/overlay: local threshold < global threshold required");
  }

  // set comm style dependent on gridstyle ? see Input::comm_style()
  // probably not since there is no running simulation in input (in contrast to here)
  if (gridstyle == STAGGERED && comm->style != 2)
    error->all(FLERR,"balance/overlay: comm_style staggered required for staggered grid");
  if (gridstyle == TENSOR && comm->style != 0)
    error->all(FLERR,"balance/overlay: comm_style brick required for tensor grid");

  if (gridstyle == STAGGERED)
    jull_local = new ALL::ALL<double, double>(ALL::STAGGERED, 3, 0);
  else if (gridstyle == TENSOR)
    if (tensorstyle == TENSOR_MAX)
      jull_local = new ALL::ALL<double, double>(ALL::TENSOR_MAX, 3, 0);
  else if (tensorstyle == TENSOR_CLASSIC)
      jull_local = new ALL::ALL<double, double>(ALL::TENSOR, 3, 0);

  if (use_global_lb) {
    jull_global = new ALL::ALL<double, double>(ALL::HISTOGRAM, 3, 0);
    n_bins = std::vector<int>(3, -1);
  }

  jull_last = nullptr;

  global_freq = nevery;

  irregular = new Irregular(lmp);

  x_masters = MPI_COMM_NULL;
  y_masters = MPI_COMM_NULL;
  z_masters = MPI_COMM_NULL;

  timer_lb = imbalance = maxloadperproc = -1;

  force_reneighbor = 1;
  lastbalance = -1;
  next_reneighbor = -1;
  next_file_write = -1;

  reduce_outvec_flag = false;
  for (int i = 0; i < 3; ++i) outvec_timer[i] = -1;
}

/**
 *  Deconstructor. Free communicators and delete allocated memory.
 */

FixBalanceOverlay::~FixBalanceOverlay()
{
  jull_last = nullptr;
  if (jull_local) delete jull_local;
  if (jull_global) delete jull_global;
  delete irregular;
  if (jull_timer) delete jull_timer;

  if (fh != MPI_FILE_NULL) MPI_File_close(&fh);

  if (x_masters != MPI_COMM_NULL) MPI_Comm_free(&x_masters);
  if (y_masters != MPI_COMM_NULL) MPI_Comm_free(&y_masters);
  if (z_masters != MPI_COMM_NULL) MPI_Comm_free(&z_masters);

  // check nfix in case all fixes have already been deleted
  if (fixstore && modify->nfix) modify->delete_fix(fixstore->id);
  fixstore = nullptr;

#ifdef DEBUG_COMM_STAGGERED
  fclose(fp);
#endif
}

/**
  * allocate per-particle weight storage for histogram via FixStoreAtom
  * fix could already be allocated if fix balance is re-specified
  */

void FixBalanceOverlay::post_constructor()
{
#ifdef DEBUG_COMM_STAGGERED
  fprintf(fp, "Hello from post_constructor\n");
  fflush(fp);
#endif
  if (! wtflag) return;

  std::string cmd;
  cmd = id;
  cmd += "HISTOGRAM_WEIGHTS";
  fixstore = dynamic_cast<FixStoreAtom *>( modify->get_fix_by_id(cmd));
  if (!fixstore) fixstore = dynamic_cast<FixStoreAtom *>( modify->add_fix(cmd + " all STORE/ATOM 1 0 0 0"));

  // do not carry weights with atoms during normal atom migration
  fixstore->disable = 1;
}

/**
 *  For lammps. This fix is just called in pre_exchange.
 *  @return mask
 */

int FixBalanceOverlay::setmask()
{
#ifdef DEBUG_COMM_STAGGERED
  fprintf(fp, "Hello from setmask\n");
  fflush(fp);
#endif
  int mask = 0;
  mask |= PRE_EXCHANGE;
  mask |= MIN_PRE_EXCHANGE;
  return mask;
}

/**
 *  Initialise calculated variables and setup ALL objects.
 *  Setup initial grid.
 *  Write information to log.
 */

void FixBalanceOverlay::init()
{
#ifdef DEBUG_COMM_STAGGERED
  fprintf(fp, "Hello from init\n");
  fflush(fp);
#endif
  // called before every minimize and run
  // @todo: remove commented code if alternative works!
  //if (force->kspace) kspace_flag = 1;
  //else kspace_flag = 0;

  int counter = 0;
  for (int i=0; i<modify->nfix; i++) {
    if (strcmp(modify->fix[i]->style, "balance") == 0) counter++;
    if (strcmp(modify->fix[i]->style, "balance/overlay") == 0) counter++;
  }
  if (counter > 1) error->all(FLERR, "More than one dynamic load balancing fix");

  // get vectors as input for ALL

  // use existing uniform grid
  if (comm->layout == Comm::LAYOUT_TILED) error->all(FLERR, "balance/overlay: initialisation from Comm::LAYOUT_TILED not possible.");
  if (comm->layout == Comm::LAYOUT_STAGGERED && gridstyle == TENSOR) error->all(FLERR, "balance/overlay: initialisation from Comm::STAGGERED not possible.");
  if (gridstyle == STAGGERED && comm->style == 2 && (comm->staggered2spatial[0]!=2 || comm->staggered2spatial[1]!=1 || comm->staggered2spatial[2]!=0))
    error->all(FLERR, "balance/overlay: comm_style staggered not zyx");
  if (domain->dimension == 2 && comm->procgrid[2] != 1) error->all(FLERR,"balance/overlay: 2D-simulation not possible with {} processors in z-direction", comm->procgrid[2]);
  procgrid_vec.assign(comm->procgrid, comm->procgrid+3);
  myloc_vec.assign(comm->myloc, comm->myloc+3);

  minimum_domain_size = {neighbor->skin, neighbor->skin, neighbor->skin};

  // setup ALL

#ifdef DEBUG_COMM_STAGGERED
  fprintf(fp, "myloc_vec %i %i %i\n", myloc_vec[0], myloc_vec[1], myloc_vec[2]);
  fprintf(fp, "procgrid_vec %i %i %i\n", procgrid_vec[0], procgrid_vec[1], procgrid_vec[2]);
  fflush(fp);
#endif

  if (use_local_lb) {
    jull_local->setProcGridParams(myloc_vec, procgrid_vec);
    jull_local->setMinDomainSize(minimum_domain_size);
    jull_local->setCommunicator(world);
    jull_local->setProcTag(comm->me);
    jull_local->setup();
  }

  if (use_global_lb) {
    jull_global->setProcGridParams(myloc_vec, procgrid_vec);
    jull_global->setMinDomainSize(minimum_domain_size);
    jull_global->setCommunicator(world);
    jull_global->setProcTag(comm->me);
    jull_global->setup();
  }

  if (jull_timer) jull_timer->init();

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
  if (nevery_file_write > 0) next_file_write = (update->ntimestep/nevery_file_write)*nevery_file_write + nevery_file_write;

  if (comm->me == 0) {
    // TODO
    /// @todo Print ALL version number.
    /// Print ALL citation information.
    utils::logmesg(lmp, "ALL information ...\n");
    if (gridstyle == STAGGERED) utils::logmesg(lmp, "\tgrid: staggered\n");
    else if (gridstyle == TENSOR) utils::logmesg(lmp, "\tgrid: tensor\n");
    utils::logmesg(lmp, "\tnumber of processors: {} x {} y {} z\n", procgrid_vec[0], procgrid_vec[1], procgrid_vec[2]);
    if (workstyle == NATOMS) utils::logmesg(lmp, "\twork: number of atoms\n");
    else if (workstyle == TIME) utils::logmesg(lmp, "\twork: time\n");
    else if (workstyle == RANK) utils::logmesg(lmp, "\twork: rank\n");
    if (use_global_lb) {
      utils::logmesg(lmp, "\tglobal load balancing ...\n");
      utils::logmesg(lmp, "\t\ttrigger threshold: {}\n", global_threshold);
      utils::logmesg(lmp, "\t\tbin width: {}\n", bw);
    }
    if (use_local_lb) {
      utils::logmesg(lmp, "\tlocal load balancing ...\n");
      utils::logmesg(lmp, "\t\ttrigger threshold: {}\n", local_threshold);
    }
    if (fh != MPI_FILE_NULL) utils::logmesg(lmp, "\thistogram file is used\n");
    if (verbose) utils::logmesg(lmp, "\tverbose output is used\n");
  }
}

/**
 *  Perform dynamic load balancing if required.
 *  Calls corresponding local or global balancing function.
 */

void FixBalanceOverlay::min_pre_exchange()
{
#ifdef DEBUG_COMM_STAGGERED
  fprintf(fp, "Hello from min_pre_exchange\n");
  fflush(fp);
#endif
  pre_exchange();
}

/**
 *  Perform dynamic load balancing if required.
 *  Calls corresponding local or global balancing function.
 */

void FixBalanceOverlay::pre_exchange()
{
#ifdef DEBUG_COMM_STAGGERED
  fprintf(fp, "Hello from pre_exchange\n");
  fflush(fp);
#endif
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

void FixBalanceOverlay::balance()
{
#ifdef DEBUG_COMM_STAGGERED
  fprintf(fp, "Hello from balance\n");
  fflush(fp);
#endif
  // do not allow rebalancing twice on same timestep
  // even if you wanted to, it can mess up elapsed time

  if (update->ntimestep == lastbalance) return;
  lastbalance = update->ntimestep;

  double timer_lb_start = platform::walltime();

  work = get_work();

  calc_imbalance();

  // call load-balancing function if required
  if (use_global_lb && imbalance > global_threshold)
    balance_global();
  else if (use_local_lb && imbalance > local_threshold)
    balance_local();
  else {
    if (verbose && comm->me == 0) utils::logmesg(lmp, "balance/overlay: {} no balance required\n", imbalance);
    jull_last = nullptr;
  }

  timer_lb = platform::walltime() - timer_lb_start;
  reduce_outvec_flag = true;
}

/**
 *  Perform staggered/tensor load balancing.
 */

void FixBalanceOverlay::balance_local()
{
#ifdef DEBUG_COMM_STAGGERED
  fprintf(fp, "Hello from balance_local\n");
  fflush(fp);
#endif
  if (verbose && comm->me == 0) utils::logmesg(lmp, "balance/overlay: {} balance local\n", imbalance);

  jull_last = jull_local;

  // ensure atoms are in current box & update box via shrink-wrap
  // no exchange() since doesn't matter if atoms are assigned to correct procs

#ifdef DEBUG_COMM_STAGGERED
  fprintf(fp, "domain pbc and reset_box\n");
  fflush(fp);
#endif
  domain->pbc();
  domain->reset_box();

  if (fh != MPI_FILE_NULL && update->ntimestep >= next_file_write && nevery_file_write > 0) {

    next_file_write = (update->ntimestep/nevery_file_write)*nevery_file_write + nevery_file_write;

    double buffer [23];

    buffer[0] = comm->me;
    buffer[1] = 2;
    buffer[2] = work;
    buffer[3] = -1;
    buffer[4] = jull_timer ? jull_timer->get_timer(0) : -1;
    buffer[5] = jull_timer ? jull_timer->get_timer(1) : -1;
    buffer[6] = jull_timer ? jull_timer->get_timer(2) : -1;
    buffer[7] = jull_timer ? jull_timer->get_timer(3) : -1;
    buffer[8] = atom->nlocal;
    buffer[9] = comm->layout == Comm::LAYOUT_STAGGERED ? comm->mysplit[0][0] : -1;
    buffer[10] = comm->layout == Comm::LAYOUT_STAGGERED ? comm->mysplit[0][1] : -1;
    buffer[11] = comm->layout == Comm::LAYOUT_STAGGERED ? comm->mysplit[1][0] : -1;
    buffer[12] = comm->layout == Comm::LAYOUT_STAGGERED ? comm->mysplit[1][1] : -1;
    buffer[13] = comm->layout == Comm::LAYOUT_STAGGERED ? comm->mysplit[2][0] : -1;
    buffer[14] = comm->layout == Comm::LAYOUT_STAGGERED ? comm->mysplit[2][1] : -1;
    buffer[15] = update->ntimestep;
    MPI_File_write_ordered(fh, buffer, 16, MPI_DOUBLE, MPI_STATUS_IGNORE);
  }

  // The domain size is changed by adjusting the domain specific variables in comm.h
  // local box is set with
  // comm->{xsplit ysplit zsplit myloc procgrid} for tensor
  // comm->mysplit                               for staggered

  // rebalance with ALL
  my_vertices = get_comm_vertices();
#ifdef DEBUG_COMM_STAGGERED
  fprintf(fp, "set Vertices and Work for ALL\n");
  fflush(fp);
#endif
  jull_local->setVertices(my_vertices);
  jull_local->setWork(work);
#ifdef DEBUG_COMM_STAGGERED
  fprintf(fp, "balance with ALL\n");
  fflush(fp);
#endif
  jull_local->balance();

  set_comm_vertices(jull_local->getVertices());
#ifdef DEBUG_COMM_STAGGERED
  fprintf(fp, "domain->set_local_box\n");
  fflush(fp);
#endif
  domain->set_local_box();

  // not required since the minimum box size is also passed to all
  // domain->subbox_too_small_check(neighbor->skin);

  // TODO
  /// @todo
  /// Use methods from rebalance with sendprocs to migrate atoms after domain shift?
  /// This requires the target processor of every atom.
  /// Implement this functionality for staggered?
  /// It is probably not worth the effort for tensor since this routine is already optimised by lammps.
#ifdef DEBUG_COMM_STAGGERED
  fprintf(fp, "migrate atoms\n");
  fflush(fp);
#endif
  if (gridstyle == STAGGERED) irregular->migrate_atoms();
  else if (irregular->migrate_check()) irregular->migrate_atoms();

  // @todo: remove commented code if alternative works!
  //if (kspace_flag) force->kspace->setup_grid();
  modify->reset_grid();
  if (force->pair) force->pair->reset_grid();
  if (force->kspace) force->kspace->reset_grid();
}

/**
 *  Perform histogram load balancing in z, y and x.
 */

void FixBalanceOverlay::balance_global()
{
#ifdef DEBUG_COMM_STAGGERED
  fprintf(fp, "Hello from balance_global\n");
  fflush(fp);
#endif
  if (verbose && comm->me == 0) utils::logmesg(lmp, "balance/overlay: {} balance global\n", imbalance);

  jull_last = jull_global;

  // only for staggered
  // histogram method

  // insure atoms are in current box & update box via shrink-wrap
  // no exchange() since doesn't matter if atoms are assigned to correct procs

#ifdef DEBUG_COMM_STAGGERED
  fprintf(fp, "domain pbc and reset_box\n");
  fflush(fp);
#endif
  domain->pbc();
  domain->reset_box();

  // processors may not have complex/simple particles yet, but get some during layer-balancing
  // -> calculate global average
  // only once since timers are evaluated
  set_weights();

#ifdef DEBUG_COMM_STAGGERED
  fprintf(fp, "irregular->migrate_atoms\n");
  fflush(fp);
#endif
  // atoms should be inside of the boundaries for the histogram calculation
  irregular->migrate_atoms();

  for (int dim_balance=2; dim_balance>=0; dim_balance--) {
#ifdef DEBUG_COMM_STAGGERED
    fprintf(fp, "rebalance with ALL in dimension %i\n", dim_balance);
    fflush(fp);
#endif

    // rebalance with ALL
    my_vertices = get_comm_vertices();
#ifdef DEBUG_COMM_STAGGERED
    fprintf(fp, "set Vertices and SysSize for ALL\n");
    fflush(fp);
#endif
    jull_global->setVertices(my_vertices);
    jull_global->setSysSize(get_sys_size_from_domain());

#ifdef DEBUG_COMM_STAGGERED
    fprintf(fp, "set Work and MethodData for ALL\n");
    fflush(fp);
#endif
    jull_global->setWork(calc_histogram(dim_balance));
    jull_global->setMethodData(n_bins.data());

#ifdef DEBUG_COMM_STAGGERED
    fprintf(fp, "balance with ALL\n");
    fflush(fp);
#endif
    jull_global->balance();
    set_comm_vertices(jull_global->getVertices());

#ifdef DEBUG_COMM_STAGGERED
    fprintf(fp, "domain->set_local_box\n");
    fflush(fp);
#endif
    domain->set_local_box();

#ifdef DEBUG_COMM_STAGGERED
    fprintf(fp, "irregular->migrate_atoms\n");
    fflush(fp);
#endif
    irregular->migrate_atoms();
  }
  unset_weights();

  // @todo: remove commented code if alternative works!
  //if (kspace_flag) force->kspace->setup_grid();
  modify->reset_grid();
  if (force->pair) force->pair->reset_grid();
  if (force->kspace) force->kspace->reset_grid();

}

/**
 *  Get the size of the simulation box from the domain class.
 *  @return box size in box units with box origin in 0,0,0
 */

std::vector<double> FixBalanceOverlay::get_sys_size_from_domain()
{
#ifdef DEBUG_COMM_STAGGERED
  fprintf(fp, "Hello from get_sys_size_from_domain x 0 %f y 0 %f z 0 %f\n", domain->xprd, domain->yprd, domain->zprd);
  fflush(fp);
#endif
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

std::vector<ALL::Point<double>> FixBalanceOverlay::get_comm_vertices()
{
#ifdef DEBUG_COMM_STAGGERED
  fprintf(fp, "Hello from get_comm_vertices\n");
  fflush(fp);
#endif
  std::vector<ALL::Point<double>> vertices(2, ALL::Point<double>(3));
  // comm stores the values in reduced coordinates in [0,1]
  // -> multiply with box length per dimension
  if (comm->layout == Comm::LAYOUT_STAGGERED) {
    /// @todo hard code zero and one ? The returned vertices are passed only to an ALL object, which does not care.
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

#ifdef DEBUG_COMM_STAGGERED
  fprintf(fp, "vertices lo %f %f %f hi %f %f %f\n", vertices[0][0], vertices[0][1], vertices[0][2], vertices[1][0], vertices[1][1], vertices[1][2]);
  fflush(fp);
#endif

  return vertices;
}

/**
 *  Get load of this domain for imbalance calculation and setWork of
 *  local ALL object.
 *  @return work according to used definition
 */

double FixBalanceOverlay::get_work()
{
#ifdef DEBUG_COMM_STAGGERED
  fprintf(fp, "Hello from get_work\n");
  fflush(fp);
#endif

  double work = 0;

  if (workstyle == TIME) work = jull_timer->get_work() + 0.1;
  else if (workstyle == NATOMS) work = atom->nlocal;
  else if (workstyle == RANK) work = comm->me;
  //printf("proc %i natoms %i work %f\n", comm->me, atom->nlocal, work);

#ifdef DEBUG_COMM_STAGGERED
  fprintf(fp, "work %f\n", work);
  fflush(fp);
#endif

  return work;
}

/**
 *  Set domain boundaries in class comm for tiled or brick layout.
 *  @param[in] vertices returned by ALL object after load balancing step
 *  @note updates layout, staggerednew, mysplit for staggered
 *  @note updates xsplit, ysplit, zsplit for tensor
 */

void FixBalanceOverlay::set_comm_vertices(std::vector<ALL::Point<double>> vertices)
{
#ifdef DEBUG_COMM_STAGGERED
  fprintf(fp, "Hello from set_comm_vertices\n");
  fprintf(fp, "vertices lo %f %f %f hi %f %f %f\n", vertices[0][0], vertices[0][1], vertices[0][2], vertices[1][0], vertices[1][1], vertices[1][2]);
  fflush(fp);
#endif
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
#ifdef DEBUG_COMM_STAGGERED
    fprintf(fp, "mysplit x before hard coding %f %f y %f %f z %f %f\n", comm->mysplit[0][0], comm->mysplit[0][1], comm->mysplit[1][0], comm->mysplit[1][1], comm->mysplit[2][0], comm->mysplit[2][1]);
  procgrid_vec.assign(comm->procgrid, comm->procgrid+3);
  myloc_vec.assign(comm->myloc, comm->myloc+3);
    fprintf(fp, "myloc %i %i %i procgrid %i %i %i\n", myloc_vec[0], myloc_vec[1], myloc_vec[2], procgrid_vec[0], procgrid_vec[1], procgrid_vec[2]);
    fflush(fp);
#endif
    for (int idim=0; idim<3; idim++) {
      if (myloc_vec[idim] == 0) comm->mysplit[idim][0] = 0;
      if (myloc_vec[idim] == procgrid_vec[idim]-1) comm->mysplit[idim][1] = 1;
    }

#ifdef DEBUG_COMM_STAGGERED
  fprintf(fp, "mysplit x after hard coding %f %f y %f %f z %f %f\n", comm->mysplit[0][0], comm->mysplit[0][1], comm->mysplit[1][0], comm->mysplit[1][1], comm->mysplit[2][0], comm->mysplit[2][1]);
  fflush(fp);
#endif
  } else {
    // gridstyle == TENSOR

    // rescale to [0:1]
    vertices[1][0] /= domain->prd[0];
    vertices[1][1] /= domain->prd[1];
    vertices[1][2] /= domain->prd[2];

    ///> @todo is the communication of vertices for tensor required or done in comm class?
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

void FixBalanceOverlay::calc_imbalance()
{
#ifdef DEBUG_COMM_STAGGERED
  fprintf(fp, "Hello from calc_imbalance\n");
  fflush(fp);
#endif
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
#ifdef DEBUG_COMM_STAGGERED
  fprintf(fp, "imbalance %f\n", imbalance);
  fflush(fp);
#endif
}

/**
  * set weight for each particle
  */

void FixBalanceOverlay::set_weights()
{
#ifdef DEBUG_COMM_STAGGERED
  fprintf(fp, "Hello from set_weights\n");
  fflush(fp);
#endif
  if (!wtflag) return;
  weight = fixstore->vstore;

  int nlocal = atom->nlocal;
  for (int i = 0; i < nlocal; i++) weight[i] = 1.0;
  for (int n = 0; n < nimbalance; n++) imbalances[n]->compute(weight);

  // weights need to migrate with atoms
  fixstore->disable = 0;
}

/**
  * prevent further migration of weights
  */

void FixBalanceOverlay::unset_weights()
{
#ifdef DEBUG_COMM_STAGGERED
  fprintf(fp, "Hello from unset_weights\n");
  fflush(fp);
#endif
  if (!wtflag) return;

  // weights should not migrate with atoms
  fixstore->disable = 1;
}

/**
 *  Calculate histogram for setWork of histogram balancing.
 *  @note writes stats to file if fh is set
 *  @note updates n_bins
 *  @param[in] dimension in which the histogram is calculated
 *  @return histogram for setWork
 */

std::vector<double> FixBalanceOverlay::calc_histogram(int dimension)
{
#ifdef DEBUG_COMM_STAGGERED
  fprintf(fp, "Hello from calc_histogram in dimension %i\n", dimension);
  fflush(fp);
#endif
  // calculate the histogram width in a way that lower and upper box boundary match with a bin boundary
  int n_bins_global = std::ceil(domain->prd[dimension] / bw);
  const double bin_width = domain->prd[dimension] / n_bins_global;

  double lb = std::ceil(my_vertices[0][dimension] / bin_width) * bin_width;
  double ub = std::ceil(my_vertices[1][dimension] / bin_width) * bin_width;

#ifdef DEBUG_COMM_STAGGERED
  fprintf(fp, "n_bins_global %i\nbin_width %f\nlb %f ub%f\n", n_bins_global, bin_width, lb, ub);
  fflush(fp);
#endif

  double overlap = 0; // bin -1 which is send to lower neighbour

  n_bins.at(dimension) = (int) (std::round((ub -lb) / bin_width) + 1e-4);
#ifdef DEBUG_COMM_STAGGERED
  fprintf(fp, "size histogram in dimension %i\n", n_bins.at(dimension));
  fflush(fp);
#endif

  std::vector<double> work_vec(n_bins.at(dimension), 0.0);

  // compute histogram of work load
  double ** x = atom->x;

  // the work per atom is constant
  // -> set work before iterating over atoms
  double work_atom = 0;
  double *weight = nullptr;
  if (workstyle == NATOMS) {
    work_atom = 1;
  } else if (workstyle == RANK) {
    work_atom = comm->me;
  } else {
    weight = fixstore->vstore;
  }

  for (int i=0; i<atom->nlocal; i++) {

    // calculate bin of atom
    const int idx = std::floor(((x[i][dimension] - domain->boxlo[dimension] - lb) / bin_width));

    // use individual weight or previously defined work
    if (wtflag) work_atom = weight[i];

    // update corresponding bin
    if (idx >= 0 && idx < n_bins.at(dimension)) {
      work_vec.at(idx) += work_atom;
    } else if (idx == -1) {
      overlap += work_atom;
    } else if (idx == n_bins.at(dimension) && fabs(x[i][dimension] - my_vertices[1][dimension])<1e-6) {
      // floating point issue, just use the last bin
      work_vec.at(n_bins.at(dimension) - 1) += work_atom;
    } else {
#ifdef DEBUG_COMM_STAGGERED
      fprintf(fp, "particle: x %15.15g y %15.15g z %15.15g\n", x[i][0], x[i][1], x[i][2]);
      fprintf(fp, "balance/overlay: unexpected histogram bin %i for histogram of size %i x %15.15g lb %15.15g bin_width %15.15g boxlo %15.15g myvert_lo %15.15g myvert_hi %15.15g", idx, n_bins.at(dimension), x[i][dimension], lb, bin_width, domain->boxlo[dimension], my_vertices[0][dimension], my_vertices[1][dimension]);
      fflush(fp);
#endif
      error->one(FLERR, "balance/overlay: unexpected histogram bin {} for histogram of size {} x {} lb {} bin_width {} boxlo {} myvert_lo {} myvert_hi {}", idx, n_bins.at(dimension), x[i][dimension], lb, bin_width, domain->boxlo[dimension], my_vertices[0][dimension], my_vertices[1][dimension]);
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

  if (fh != MPI_FILE_NULL && update->ntimestep >= next_file_write && nevery_file_write > 0) {

    // write z, y, x -> do not change the dimension for z and y
    if (dimension == 0) next_file_write = (update->ntimestep/nevery_file_write)*nevery_file_write + nevery_file_write;

    // print work stats
    // can be used to verify the work model
    // The work of the histogram (+overlap) should be correlated with the scalar work.
    double buffer [23];
    double work_sum_histogram = overlap;
    for (auto w : work_vec) work_sum_histogram += w;

    buffer[0] = comm->me;
    buffer[1] = dimension;
    buffer[2] = work;
    buffer[3] = work_sum_histogram;
    buffer[4] = jull_timer ? jull_timer->get_timer(0) : -1;
    buffer[5] = jull_timer ? jull_timer->get_timer(1) : -1;
    buffer[6] = jull_timer ? jull_timer->get_timer(2) : -1;
    buffer[7] = jull_timer ? jull_timer->get_timer(3) : -1;
    buffer[8] = atom->nlocal;
    buffer[9] = comm->layout == Comm::LAYOUT_STAGGERED ? comm->mysplit[0][0] : -1;
    buffer[10] = comm->layout == Comm::LAYOUT_STAGGERED ? comm->mysplit[0][1] : -1;
    buffer[11] = comm->layout == Comm::LAYOUT_STAGGERED ? comm->mysplit[1][0] : -1;
    buffer[12] = comm->layout == Comm::LAYOUT_STAGGERED ? comm->mysplit[1][1] : -1;
    buffer[13] = comm->layout == Comm::LAYOUT_STAGGERED ? comm->mysplit[2][0] : -1;
    buffer[14] = comm->layout == Comm::LAYOUT_STAGGERED ? comm->mysplit[2][1] : -1;
    buffer[15] = update->ntimestep;
    MPI_File_write_ordered(fh, buffer, 16, MPI_DOUBLE, MPI_STATUS_IGNORE);
  }

  // exchange overlapping workload (histograms might overlap
  // over the domain boundaries

  MPI_Request sreq, rreq;
  MPI_Status ssta, rsta;

  double recv_work = 0;

#ifdef DEBUG_COMM_STAGGERED
  fprintf(fp, "exchange histogram info with neighbours: send to %i and recv from %i\n", rank_left, rank_right);
  fflush(fp);
#endif

  MPI_Isend(&overlap, 1, MPI_DOUBLE, rank_left, 0, world, &sreq);
  MPI_Irecv(&recv_work, 1, MPI_DOUBLE, rank_right, 0, world, &rreq);
  MPI_Wait(&rreq, &rsta);
  MPI_Wait(&sreq, &ssta);

  work_vec.at(n_bins.at(dimension) - 1) += recv_work;

#ifdef ALL_DEBUG_ENABLED
  for (int i=0; i<work_vec.size(); i++) {
    if (! isfinite(work_vec[i])) {
      error->one(FLERR, "[{}]: work_vec[{}]={} work_vec.size()={}", comm->me, i, work_vec[i], work_vec.size());
    }
  }
#endif
#ifdef DEBUG_COMM_STAGGERED
  fprintf(fp, "return work_vec\n");
  fflush(fp);
#endif
  return work_vec;
}

/**
 *  Write the given histogram to standard output.
 *  @param[in] work_vec histogram to be printed
 */

void FixBalanceOverlay::print_histogram(std::vector<double> work_vec)
{
#ifdef DEBUG_COMM_STAGGERED
  fprintf(fp, "Hello from print_histogram\n");
  fflush(fp);
#endif
  printf("[%i]: histogram", comm->me);
  for (double w : work_vec) printf(" %f", w);
  printf("\n");
}

/**
 *  For lammps output only.
 *  @param[in] i index of output vector
 *  @return requested value
 */

double FixBalanceOverlay::compute_vector(int i)
{
#ifdef DEBUG_COMM_STAGGERED
  fprintf(fp, "Hello from compute_vector\n");
  fflush(fp);
#endif
  if (reduce_outvec_flag) {
    double reducebuffer_s, reducebuffer_r;
    reducebuffer_s = timer_lb;

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
  if (i == 1) return 1;
  if (i == 2) return imbalance;
  if (i <= 5) return outvec_timer[i-3];
  return -1;
}

/**
 *  For lammps stats only.
 *  @return # of bytes of allocated memory
 */

double FixBalanceOverlay::memory_usage()
{
#ifdef DEBUG_COMM_STAGGERED
  fprintf(fp, "Hello from memory_usage\n");
  fflush(fp);
#endif
  double bytes = irregular->memory_usage();
  // TODO
  /// @todo
  /// Add ALL memory usage.
  return bytes;
}

/**
 *  Print domain boundaries to standard output.
 *  @note needs to be called by all ranks
 */

void FixBalanceOverlay::print_domains()
{
#ifdef DEBUG_COMM_STAGGERED
  fprintf(fp, "Hello from print_domains\n");
  fflush(fp);
#endif
  if (comm->me == 0) {
    if (comm->layout == Comm::LAYOUT_TILED) {
      printf("LBSTAT LAYOUT_TILED\n");
    } else {
      printf("LBSTAT LAYOUT_TENSOR\n");
      printf("LBSTAT procgrid %i %i %i\n", comm->procgrid[0], comm->procgrid[1], comm->procgrid[1]);
      printf("LBSTAT user_procgrid %i %i %i\n", comm->user_procgrid[0], comm->user_procgrid[1], comm->user_procgrid[1]);
      printf("LBSTAT xsplit %f", comm->xsplit[0]);
      for (int i_proc=1; i_proc<=comm->procgrid[0]; i_proc++)
        printf(" %f", comm->xsplit[i_proc]);
      printf("\nLBSTAT ysplit %f", comm->ysplit[0]);
      for (int i_proc=1; i_proc<=comm->procgrid[1]; i_proc++)
        printf(" %f", comm->ysplit[i_proc]);
      printf("\nLBSTAT zsplit %f", comm->zsplit[0]);
      for (int i_proc=1; i_proc<=comm->procgrid[2]; i_proc++)
        printf(" %f", comm->zsplit[i_proc]);
      printf("\n");
    }
  }

  for (int i_proc=0; i_proc<comm->nprocs; i_proc++) {
    MPI_Barrier(world);
    if (i_proc != comm->me) { continue; }
    printf("LBSTAT processor %i\n", i_proc);

    if (comm->layout == Comm::LAYOUT_TILED) {
    // public settings specific to layout = TILED

      printf("LBSTAT rcbnew %i\n", comm->rcbnew);
      printf("LBSTAT rcbcutdim %i\n", comm->rcbcutdim);
      printf("LBSTAT rcbcutfrac %f\n", comm->rcbcutfrac);
      for (int i_dim=0; i_dim<3; i_dim++)
        printf("LBSTAT mysplit %i %f %f\n", i_dim, comm->mysplit[i_dim][0], comm->mysplit[i_dim][1]);

    } else {
      // public settings specific to layout = UNIFORM, NONUNIFORM

      printf("LBSTAT myloc %i %i %i\n", comm->myloc[0], comm->myloc[1], comm->myloc[2]);
      for (int i_dim=0; i_dim<3; i_dim++)
        printf("LBSTAT procneigh %i %i %i\n", i_dim, comm->procneigh[i_dim][0], comm->procneigh[i_dim][1]);

    }
  }
#ifdef DEBUG_COMM_STAGGERED
  fprintf(fp, "Bye from print_domains\n");
  fflush(fp);
#endif

}

/**
 *  Set everything to zero.
 *  @param[in] lmp lammps to get the Pointers class
 */

JullTimer::JullTimer(LAMMPS *lmp) : Pointers(lmp) {
  for (int i=0; i<4; i++) {
    time[i] = 0;
    time_interval[i] = 0;
  }
}

/**
 *  Reset times to zero.
 */

void JullTimer::init() {
  // do not use set_time();
  // lammps resets the timers to zero, but the timers are not zero at timer initialisation time
  for (int i=0; i<4; i++) {
    time[i] = 0;
    time_interval[i] = 0;
  }
}

/**
 *  Save values of lammps timers.
 *  @note sets time
 */

void JullTimer::set_time() {
  time[0] = timer->get_wall(Timer::PAIR);
  time[1] = timer->get_wall(Timer::NEIGH);
  time[2] = timer->get_wall(Timer::BOND);
  time[3] = timer->get_wall(Timer::KSPACE);
}

/**
 *  Get time work since last call.
 *  @note sets time and time_interval
 *  @return work
 */

double JullTimer::get_work() {
  double work = 0;
  // store old times
  for (int i=0; i<4; i++) time_interval[i] = - time[i];
  set_time();
  // calculate differences to new times
  for (int i=0; i<4; i++) {
    time_interval[i] += time[i];
    work += time_interval[i];
  }
  // add constant time to prevent balancing with numerically zero
  // e.g. (for few atoms per processor and balancing every step)
  return work /*+ 0.1*/;
}

/**
 *  Get requested time_interval.
 *  @note The range of the index is not checked.
 *  @param i requested array index
 *  @return time_interval
 */

double JullTimer::get_timer(int i) {
  if (i<4) return time_interval[i];
  else return -1;
}

/**
 *  Write time array to standard output.
 *  @param[in] rank
 */

void JullTimer::print_times(int rank) {
  printf("proc %i time %f %f %f %f %f\n", rank, time[0], time[1], time[2], time[3], time[4]);
}
