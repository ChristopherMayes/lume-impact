import pathlib
from .parsers import (
    parse_impact_input,
    load_many_fort,
    FORT_STAT_TYPES,
    FORT_DIPOLE_STAT_TYPES,
    FORT_PARTICLE_TYPES,
    header_str,
    header_bookkeeper,
    parse_impact_particles,
    load_stats,
    load_slice_info,
    fort_files,
)
from . import archive, writers, fieldmaps, tools
from .lattice import ele_dict_from, ele_str, get_stop, set_stop, insert_ele_by_s
from .control import ControlGroup

from .fieldmaps import lattice_field
from .plot import plot_stat, plot_layout, plot_stats_with_layout
from .particles import identify_species, track_to_s, track1_to_s
from .fast_autophase import fast_autophase_impact

from .interfaces.bmad import impact_from_tao


from pmd_beamphysics import ParticleGroup
from pmd_beamphysics.units import pmd_unit
from pmd_beamphysics.interfaces.impact import impact_particles_to_particle_data

from scipy.interpolate import interp1d

import h5py
import numpy as np


import functools
from time import time
from copy import deepcopy
import os

from lume.base import CommandWrapper


EXTRA_UNITS = {
    "Bz": pmd_unit("T"),
    "Ez": pmd_unit("V/m"),
}


class Impact(CommandWrapper):
    """

    Files will be written into a temporary directory within workdir.
    If workdir=None, a location will be determined by the system.


    """

    COMMAND = "ImpactTexe"
    COMMAND_MPI = "ImpactTexe-mpi"

    MPI_RUN = tools.find_mpirun()
    WORKDIR = tools.find_workdir()

    # Environmental variables to search for executables
    command_env = "IMPACTT_BIN"
    command_mpi_env = "IMPACTT_MPI_BIN"

    def __init__(self, *args, group=None, always_autophase=False, **kwargs):
        super().__init__(*args, **kwargs)
        # Save init
        self.original_input_file = self.input_file

        self.input = {"header": {}, "lattice": []}
        self.output = {}
        self._units = {}
        self._units.update(EXTRA_UNITS)
        self.group = {}

        # MPI
        self._nnode = 1

        # Convenience lookup of elements in lattice by name
        self.ele = {}

        # Autophase settings to be applied.
        # This will be cleared when actually autophasing
        self._autophase_settings = {}
        self.always_autophase = always_autophase

        # Call configure
        if self.input_file:
            infile = tools.full_path(self.input_file)
            assert os.path.exists(infile), f"Impact input file does not exist: {infile}"
            self.load_input(self.input_file)
            self.configure()

            # Add groups, if any.
            if group:
                for k, v in group.items():
                    self.add_group(k, **v)

        else:
            self.vprint("Using default input: 1 m drift lattice")
            self.input = deepcopy(DEFAULT_INPUT)
            self.configure()

    def add_ele(self, ele):
        """
        Adds an element to .lattice
        """
        name = ele["name"]
        assert name not in self.lattice, "Element already exists"
        insert_ele_by_s(ele, self.lattice, verbose=self.verbose)
        # Add to the ele dict
        self.ele[name] = ele

    def add_group(self, name, **kwargs):
        """
        Add a control group. See control.py
        """
        assert name not in self.ele
        if name in self.group:
            self.vprint(f"Warning: group {name} already exists, overwriting.")

        g = ControlGroup(**kwargs, name=name)
        g.link(self.ele)
        self.group[name] = g

        return self.group[name]

    def configure(self):
        self.configure_impact(workdir=self._workdir)

    def configure_impact(self, input_filepath=None, workdir=None):
        if input_filepath:
            self.load_input(input_filepath)

        # Header Bookkeeper
        self.input["header"] = header_bookkeeper(self.header, verbose=self.verbose)

        if len(self.input["lattice"]) == 0:
            self.vprint("Warning: lattice is empty. Not configured")
            self.configured = False
            return

        # Set ele dict from the lattice
        self.ele_bookkeeper()

        self.setup_workdir(workdir)
        self.vprint("Configured to run in:", self.path)
        self.configured = True

    def input_parser(self, path):
        return parse_impact_input(path, verbose=self.verbose)

    def load_output(self):
        """
        Loads stats, slice_info, and particles.
        """
        self.output["stats"], u = load_stats(
            self.path, species=self.species, types=FORT_STAT_TYPES, verbose=self.verbose
        )
        self._units.update(u)

        # This is not always present
        dipole_stats, u = load_stats(
            self.path,
            species=self.species,
            types=FORT_DIPOLE_STAT_TYPES,
            verbose=self.verbose,
        )
        if dipole_stats:
            self.output["dipole_stats"] = dipole_stats
            self._units.update(u)

        self.output["slice_info"], u = load_slice_info(self.path, self.verbose)
        self._units.update(u)

        self.load_particles()

    def load_particles(self):
        # Standard output
        self.vprint("Loading particles")
        self.output["particles"] = load_many_fort(
            self.path, FORT_PARTICLE_TYPES, verbose=self.verbose
        )

        # Additional particle files:
        for e in self.input["lattice"]:
            if e["type"] == "write_beam":
                name = e["name"]
                fname = e["filename"]
                full_fname = os.path.join(self.path, fname)
                if os.path.exists(full_fname):
                    self.particles[name] = parse_impact_particles(full_fname)
                    self.vprint(f"Loaded write beam particles {name} {fname}")

        # Convert all to ParticleGroup

        # Interpolate stats to get the time.
        time_f = interp1d(
            self.output["stats"]["mean_z"],
            self.output["stats"]["t"],
            assume_sorted=True,
            fill_value="extrapolate",
        )

        for name, pdata in self.particles.items():
            # Initial particles have special z = beta_ref*c. See: impact_particles_to_particle_data
            if name == "initial_particles" and self.header["Flagimg"]:
                cathode_kinetic_energy_ref = self.header["Bkenergy"]
            else:
                cathode_kinetic_energy_ref = None

            time = time_f(pdata["z"].mean())

            pg_data = impact_particles_to_particle_data(
                pdata,
                mc2=self.mc2,
                species=self.species,
                time=time,
                macrocharge=self.macrocharge,
                cathode_kinetic_energy_ref=cathode_kinetic_energy_ref,
                verbose=self.verbose,
            )
            self.particles[name] = ParticleGroup(data=pg_data)
            self.vprint(f"Converted {name} to ParticleGroup")

    def ele_bookkeeper(self):
        """
        Link .ele = dict to the lattice elements by their 'name' field
        """
        self.ele = ele_dict_from(self.input["lattice"])

    @property
    def nnode(self):
        """
        Number of MPI nodes to use
        """
        return self._nnode

    @nnode.setter
    def nnode(self, n):
        self._nnode = n

    # Convenience routines
    @property
    def header(self):
        """Convenience pointer to .input['header']"""
        return self.input["header"]

    @property
    def lattice(self):
        """Convenience pointer to .input['lattice']"""
        return self.input["lattice"]

    @property
    def particles(self):
        """Convenience pointer to .input['lattice']"""
        return self.output["particles"]

    @property
    def fieldmaps(self):
        """Convenience pointer to .input['fieldmaps']"""
        return self.input["fieldmaps"]

    def field(self, z=0, t=0, x=0, y=0, component="Ez"):
        """
        Return the field component at a longitudinal
        position z at time t.

        Warking: This is based on the parsed fieldmaps,
        and not calculated directly from Impact. Not all elements/parameters
        are implemented. Currently x, y must be 0.
        """
        return lattice_field(
            self.lattice, x=x, y=y, z=z, t=t, component=component, fmaps=self.fieldmaps
        )

    def centroid_field(self, component="Ez"):
        zlist = self.stat("mean_z")
        tlist = self.stat("t")
        return np.array(
            [self.field(z=z, t=t, component=component) for z, t in zip(zlist, tlist)]
        )

    def stat(self, key):
        """
        Array from `.output['stats'][key]`

        Additional keys are avalable:
            'mean_energy': mean energy
            'Ez': z component of the electric field at the centroid particle
            'Bz'  z component of the magnetic field at the centroid particle
            'cov_{a}__{b}': any symmetric covariance matrix term

        """

        if key in ("Ez", "Bz"):
            return self.centroid_field(component=key[0:2])

        if key == "mean_energy":
            return self.stat("mean_kinetic_energy") + self.mc2

        # Allow flipping covariance keys
        if key.startswith("cov_") and key not in self.output["stats"]:
            k1, k2 = key[4:].split("__")
            key = f"cov_{k2}__{k1}"

        if key not in self.output["stats"]:
            raise ValueError(f"{key} is not available in the output data")

        return self.output["stats"][key]

    def units(self, key):
        """pmd_unit of a given key"""

        # Allow flipping covariance keys
        if key.startswith("cov_") and key not in self._units:
            k1, k2 = key[4:].split("__")
            key = f"cov_{k2}__{k1}"

        if key not in self._units:
            raise ValueError(f"Unknown unit for {key}")

        return self._units[key]

    # --------------
    # Run
    def run(self):
        if not self.configured:
            self.vprint("not configured to run")
            return
        self.run_impact(verbose=self.verbose, timeout=self.timeout)

    def get_executable(self):
        """
        Gets the full path of the executable from .command, .command_mpi
        Will search environmental variables:
                Impact.command_env='IMPACTT_BIN'
                Impact.command_mpi_env='IMPACTT_MPI_BIN'

        """
        if self.use_mpi:
            exe = tools.find_executable(
                exename=self.command_mpi, envname=self.command_mpi_env
            )
        else:
            exe = tools.find_executable(exename=self.command, envname=self.command_env)
        return exe

    @property
    def numprocs(self):
        """Number of MPI processors = Npcol*Nprow"""
        return self.input["header"]["Npcol"] * self.input["header"]["Nprow"]

    @numprocs.setter
    def numprocs(self, n):
        """Sets the number of processors"""
        if n < 0:
            raise ValueError("numprocs must be >= 0")
        if not n:
            n = tools.get_suggested_nproc()

        Nz = self.header["Nz"]
        Ny = self.header["Ny"]
        Npcol, Nprow = suggested_processor_domain(Nz, Ny, n)

        self.vprint(f"Setting Npcol, Nprow = {Npcol}, {Nprow}")
        self.header["Nprow"] = Nprow
        self.header["Npcol"] = Npcol

        if self.use_mpi and n == 1:
            self.vprint("Disabling MPI")
            self.use_mpi = False

        if n > 1 and not self.use_mpi:
            self.vprint("Enabling MPI")
            self.use_mpi = True

    def get_run_script(self, write_to_path=False, path=None):
        """
        Assembles the run script using self.mpi_run string of the form:
            'mpirun -n {n} {command_mpi}'

        Optionally writes a file 'run' with this line to path.
        """

        n_procs = self.numprocs

        exe = self.get_executable()

        if self.use_mpi:
            # mpi_exe could be a complicated string like:
            # 'srun -N 1 --cpu_bind=cores {n} {command_mpi}'
            # 'mpirun -n {n} {command_mpi}'

            runscript = self.mpi_run.format(
                nnode=self.nnode, nproc=n_procs, command_mpi=exe
            )

        else:
            if n_procs > 1:
                raise ValueError("Error: n_procs > 1 but use_mpi = False")
            runscript = exe

        if write_to_path:
            if path is None:
                path = self.path
            path = os.path.join(path, "run")
            with open(path, "w") as f:
                f.write(runscript)
            tools.make_executable(path)
        return runscript

    def run_impact(self, verbose=False, timeout=None):
        """
        Runs Impact-T

        """

        # Clear output
        self.output = {}

        # Autophase
        autophase_settings = self.autophase_bookkeeper()
        if autophase_settings:
            self.output["autophase_info"] = autophase_settings

        run_info = self.output["run_info"] = {"error": False}

        # Run script, gets executables
        runscript = self.get_run_script()
        run_info["run_script"] = runscript

        t1 = time()
        run_info["start_time"] = t1

        self.vprint("Running Impact-T in " + self.path)
        self.vprint(runscript)
        # Write input
        self.write_input()

        # Remove previous files
        for f in fort_files(self.path):
            os.remove(f)

        if timeout:
            res = tools.execute2(runscript.split(), timeout=timeout, cwd=self.path)
            log = res["log"]
            self.error = res["error"]
            run_info["error"] = self.error
            run_info["why_run_error"] = res["why_error"]
        else:
            # Interactive output, for Jupyter
            log = []
            counter = 0
            for path in tools.execute(runscript.split(), cwd=self.path):
                # Fancy clearing of old lines
                counter += 1
                if verbose:
                    if counter < 15:
                        print(path, end="")
                    else:
                        print(
                            "\r",
                            path.strip() + ", elapsed: " + str(time() - t1),
                            end="",
                        )
                log.append(path)
            self.vprint("Finished.")
        self.log = log

        # Load output
        self.load_output()

        run_info["run_time"] = time() - t1

        self.finished = True

    def write_initial_particles(self, fname=None, update_header=False, path=None):
        if not fname:
            if path is None:
                path = self.path
            fname = os.path.join(path, "partcl.data")

        assert self.initial_particles.species == self.species, "Species mismatch"

        H = self.header
        # check for cathode start
        if self.cathode_start:
            cathode_kinetic_energy_ref = H["Bkenergy"]
            start_str = "Cathode start"

            if not all(self.initial_particles.z == 0):
                self.vprint("Some initial particles z !=0, disabling cathode_start")
                self.cathode_start = False
                cathode_kinetic_energy_ref = None
                start_str = "Normal start"
        else:
            cathode_kinetic_energy_ref = None
            start_str = "Normal start"

        # Call the openPMD-beamphysics writer routine
        res = self.initial_particles.write_impact(
            fname,
            verbose=self.verbose,
            cathode_kinetic_energy_ref=cathode_kinetic_energy_ref,
        )

        if update_header:
            for k, v in res.items():
                if k in H:
                    H[k] = v
                    self.vprint(
                        f"{start_str}: Replaced {k} with {v} according to initial particles"
                    )

            # These need to be set
            H["Flagdist"] = 16
            # Clear out scale factors
            for k in ["xscale", "pxscale", "yscale", "pyscale", "zscale", "pzscale"]:
                if H[k] != 1.0:
                    H[k] = 1.0
                    self.vprint(f"Changing particle scale factor {k} to 1.0")
            # Zero out offsets.
            for k in ["xmu1(m)", "xmu2", "ymu1(m)", "ymu2", "zmu1(m)", "zmu2"]:
                if H[k] != 0:
                    H[k] = 0
                    self.vprint(f"Changing particle offset factor {k} to 0")

            # Single particle must track with no space charge.
            if len(self.initial_particles) == 1:
                self.vprint("Single particle, turning space charge off")
                self.total_charge = 0

            # This will also set the header.
            # total_charge = 0 switches off space charge, so don't update.
            if self.total_charge != 0:
                charge = self.initial_particles.charge
                self.vprint(f"Setting total charge to {charge} C")
                self.total_charge = charge

    def write_input(self, input_filename="ImpactT.in", path=None):
        """
        Write all input.

        If .initial_particles are given,
        """

        if path is None:
            path = self.path

        pathlib.Path(path).mkdir(exist_ok=True, parents=True)

        filePath = os.path.join(path, input_filename)

        # Write fieldmaps
        for name, fieldmap in self.input["fieldmaps"].items():
            file = os.path.join(path, name)
            fieldmaps.write_fieldmap(file, fieldmap)

        # Initial particles (ParticleGroup)
        if self.initial_particles:
            self.write_initial_particles(update_header=True, path=path)

            # Check consistency
            if (
                self.header["Flagimg"] == 1
                and self.header["Nemission"] < 1
                and self.total_charge > 0
            ):
                raise ValueError(
                    f"Cathode start with space charge must "
                    f"set header['Nemission'] > 0. "
                    f"The current value is {self.header['Nemission']}."
                )

        # Symlink
        elif self.header["Flagdist"] == 16:
            src = self.input["input_particle_file"]
            dest = os.path.join(path, "partcl.data")

            # Don't worry about overwriting in temporary directories
            if self._tempdir and os.path.exists(dest):
                os.remove(dest)

            if not os.path.exists(dest):
                writers.write_input_particles_from_file(src, dest, self.header["Np"])
            else:
                self.vprint("partcl.data already exits, will not overwrite.")

        # Check for point-to-point spacechage processor criteria
        if self.numprocs > 1:
            for ele in self.lattice:
                if ele["type"] == "point_to_point_spacecharge":
                    Np = self.header["Np"]
                    numprocs = self.numprocs
                    if Np % numprocs != 0:
                        raise ValueError(
                            f"The number of electrons ({Np}) divided by the number of processors ({numprocs}) must be an integer."
                        )

        # Write main input file. This should come last.
        writers.write_impact_input(filePath, self.header, self.lattice)

        # Write run script
        self.get_run_script(write_to_path=True, path=path)

    @property
    def stop(self):
        return get_stop(self.lattice)

    @stop.setter
    def stop(self, s):
        """
        Sets the stop by inserting a stop element at the end of the lattice.

        Any other stop elements are removed.
        """

        self.input["lattice"], removed_eles = set_stop(self.input["lattice"], s)

        # Bookkeeping
        if self.ele:
            for ele in removed_eles:
                name = ele["name"]
                if name in self.ele:
                    self.ele.pop(name)
                    self.vprint(f"Removed element: {name}")

        self.vprint(f"Set stop to s = {s}")

    def archive(self, h5=None):
        """
        Archive all data to an h5 handle or filename.

        If no file is given, a file based on the fingerprint will be created.

        """
        if not h5:
            h5 = "impact_" + self.fingerprint() + ".h5"

        if isinstance(h5, str):
            new_h5file = True
            fname = os.path.expandvars(h5)
            g = h5py.File(fname, "w")
            self.vprint(f"Archiving to file {fname}")
        else:
            new_h5file = False
            g = h5

        # Write basic attributes
        archive.impact_init(g)

        # Initial particles
        if self.initial_particles:
            self.initial_particles.write(g, name="initial_particles")

        # All input
        archive.write_input_h5(g, self.input, name="input")

        # All output
        archive.write_output_h5(g, self.output, name="output", units=self._units)

        # Control groups
        if self.group:
            archive.write_control_groups_h5(g, self.group, name="control_groups")

        # Close file if created here.
        if new_h5file:
            g.close()

        return h5

    def load_archive(self, h5, configure=True):
        """
        Loads input and output from archived h5 file.

        See: Impact.archive
        """
        if isinstance(h5, str):
            fname = os.path.expandvars(h5)
            g = h5py.File(fname, "r")

            glist = archive.find_impact_archives(g)
            n = len(glist)
            if n == 0:
                # legacy: try top level
                message = "legacy"
            elif n == 1:
                gname = glist[0]
                message = f"group {gname} from"
                g = g[gname]
            else:
                raise ValueError(f"Multiple archives found in file {fname}: {glist}")

            self.vprint(f"Reading {message} archive file {h5}")
        else:
            g = h5

        self.input = archive.read_input_h5(g["input"], verbose=self.verbose)
        self.output, self._units = archive.read_output_h5(
            g["output"], verbose=self.verbose
        )
        self._units.update(EXTRA_UNITS)

        if "initial_particles" in g:
            self.initial_particles = ParticleGroup(h5=g["initial_particles"])

        if "control_groups" in g:
            self.group = archive.read_control_groups_h5(
                g["control_groups"], verbose=self.verbose
            )
        self.vprint("Loaded from archive. Note: Must reconfigure to run again.")
        self.configured = False

        if configure:
            self.configure()

            # Re-link groups
            # TODO: cleaner logic
            for _, cg in self.group.items():
                cg.link(self.ele)

    @property
    def cathode_start(self):
        """Returns a bool if cathode_start is requested. Can also be set as a bool."""
        return self.header["Flagimg"] == 1

    @cathode_start.setter
    def cathode_start(self, val):
        if val:
            self.header["Flagimg"] = 1
        else:
            self.header["Flagimg"] = 0

    @property
    def total_charge(self):
        """Returns the total bunch charge in C. Can be set."""
        return self.header["Bcurr"] / self.header["Bfreq"]

    @total_charge.setter
    def total_charge(self, val):
        self.header["Bcurr"] = val * self.header["Bfreq"]
        # Keep particles up-to-date.
        if self.initial_particles and val > 0:
            self.initial_particles.charge = val

    @property
    def species(self):
        return identify_species(self.header["Bmass"], self.header["Bcharge"])

    @property
    def mc2(self):
        return self.header["Bmass"]

    @property
    def macrocharge(self):
        H = self.header
        Np = H["Np"]
        if Np == 0:
            self.vprint("Error: zero particles. Returning zero macrocharge")
            return 0
        else:
            return H["Bcurr"] / H["Bfreq"] / Np

    # Phasing
    # --------
    def autophase_bookkeeper(self):
        """
        Searches for `'autophase_deg'` attribute in all eles.
        If one is found, autophase is called.

        If .always_autophase == True, calls autophase is called.

        Returns
        -------
        settings: dict
            Autophase settings found
        """
        if self._autophase_settings or self.always_autophase:
            if self.verbose:
                print("Autophase bookkeeper found settings, applying them")

            # Actual found settings
            settings = self.autophase(settings=self._autophase_settings)

            # Clear
            self._autophase_settings = {}

        else:
            settings = {}

        return settings

    def autophase(self, settings=None, full_output=False):
        """
        Calculate the relative phases of each rf element
        by tracking a single particle.
        This uses a fast method that operates outside of Impact

        Parameters
        ----------
        settings: dict, optional=None
            dict of ele_name:rel_phase_deg

        full_output: bool, optional = False
            type of output to return (see Returns)


        Returns
        -------
        if full_output = True retuns a dict of:
                ele_name:info_dict

        Otherwise returns a dict of:
            ele_name:rel_phase_deg
        which is the same format as settings.


        """

        if self.initial_particles:
            t0 = self.initial_particles["mean_t"]
            pz0 = self.initial_particles["mean_pz"]
        else:
            t0 = 0
            pz0 = 0

        return fast_autophase_impact(
            self,
            settings=settings,
            t0=t0,
            pz0=pz0,
            full_output=full_output,
            verbose=self.verbose,
        )

    # Tracking
    # ---------

    def track(self, particles, s=None):
        """
        Track a ParticleGroup. An optional stopping s can be given.
        """
        if not s:
            s = self.stop
        return track_to_s(self, particles, s)

    def track1(
        self,
        x0=0,
        px0=0,
        y0=0,
        py0=0,
        z0=0,
        pz0=1e-15,
        t0=0,
        s=None,  # final s
        species=None,
    ):
        """
        Tracks a single particle with starting coordinates:
        x0, y0, z0 in meters
        px0, py0, pz0 in eV/c
        t0 in seconds

        to a position 's' in meters

        Used for phasing and scaling elements.

        If successful, returns a ParticleGroup with the final particle.

        Otherwise, returns None

        """
        if not s:
            s = self.stop

        if not species:
            species = self.species

        # Change to serial exe just for this
        n_procs_save = self.numprocs
        self.numprocs = 1
        result = track1_to_s(
            self,
            s=s,
            x0=x0,
            px0=px0,
            y0=y0,
            py0=py0,
            z0=z0,
            pz0=pz0,
            t0=t0,
            species=species,
        )
        self.numprocs = n_procs_save
        return result

    def old_plot(self, y="sigma_x", x="mean_z", nice=True, include_layout=True):
        """
        Simple stat plot
        """
        return plot_stat(self, y=y, x=x, nice=nice)

    def plot(
        self,
        y=["sigma_x", "sigma_y"],
        x="mean_z",
        xlim=None,
        ylim=None,
        ylim2=None,
        y2=[],
        nice=True,
        include_layout=True,
        include_labels=False,
        include_markers=True,
        include_particles=True,
        include_field=True,
        field_t=None,
        include_legend=True,
        return_figure=False,
        tex=True,
        **kwargs,
    ):
        """ """

        # Just plot fieldmaps if there are no stats
        if "stats" not in self.output:
            return plot_layout(
                self,
                xlim=xlim,
                include_markers=include_markers,
                include_labels=include_labels,
                include_field=include_field,
                field_t=field_t,
                return_figure=return_figure,
                **kwargs,
            )

        return plot_stats_with_layout(
            self,
            ykeys=y,
            ykeys2=y2,
            xkey=x,
            xlim=xlim,
            ylim=ylim,
            ylim2=ylim2,
            nice=nice,
            tex=tex,
            include_layout=include_layout,
            include_labels=include_labels,
            include_field=include_field,
            field_t=field_t,
            include_markers=include_markers,
            include_particles=include_particles,
            include_legend=include_legend,
            return_figure=return_figure,
            **kwargs,
        )

    def print_lattice(self):
        """
        Pretty printing of the lattice
        """
        for ele in self.input["lattice"]:
            line = ele_str(ele)
            print(line)

    def vprint(self, *args):
        """verbose print"""
        if self.verbose:
            print(*args)

    def reset(self):
        if self.use_temp_dir:
            self.path = None
            self.configured = False

    @classmethod
    @functools.wraps(impact_from_tao)
    def from_tao(cls, tao, fieldmap_style="fourier", n_coef=30, **kwargs):
        return impact_from_tao(
            tao, fieldmap_style=fieldmap_style, n_coef=n_coef, **kwargs
        )

    def __getitem__(self, key):
        """
        Convenience syntax to get a header or element attribute.

        Special syntax:

        end_X
            will return the final item in a stat array X
            Example:
            'end_norm_emit_x'

        particles:X
            will return a ParticleGroup named X
            Example:
                'particles:initial_particles'
                returns the readback of initial particles.
        particles:X:Y
            ParticleGroup named X's property Y
            Example:
                'particles:final_particles:sigma_x'


        See: __setitem__
        """

        # Object attributes
        if hasattr(self, key):
            return getattr(self, key)

        # Send back ele or group object.
        # Do not add these to __setitem__. The user shouldn't be allowed to change them as a whole,
        #   because it will break all the links.
        if key in self.group:
            return self.group[key]
        if key in self.ele:
            return self.ele[key]

        if key.startswith("end_"):
            key2 = key[len("end_") :]
            assert (
                key2 in self.output["stats"]
            ), f"{key} does not have valid output stat: {key2}"
            return self.output["stats"][key2][-1]

        if key.startswith("particles:"):
            key2 = key[len("particles:") :]
            x = key2.split(":")
            if len(x) == 1:
                return self.particles[x[0]]
            else:
                return self.particles[x[0]][x[1]]

        # key isn't an ele or group, should have property s

        x = key.split(":")
        assert len(x) == 2, f"{x} was not found in group or ele dict, so should have : "
        name, attrib = x[0], x[1]

        if name == "header":
            return self.header[attrib]
        elif name in self.ele:
            return self.ele[name][attrib]
        elif name in self.group:
            return self.group[name][attrib]

    def __setitem__(self, key, item):
        """
        Convenience syntax to set a header or element attribute.
        attribute_string should be 'header:key' or 'ele_name:key'

        Examples of attribute_string: 'header:Np', 'SOL1:solenoid_field_scale'

        Settable attributes can also be given:

        ['stop'] = 1.2345 will set Impact.stop = 1.2345

        """

        # Set attributes
        if hasattr(self, key):
            setattr(self, key, item)
            return

        # Must be header:key or elename:attrib
        name, attrib = key.split(":")
        # Try header or lattice
        if name == "header":
            self.header[attrib] = item
        elif attrib == "autophase_deg":
            self._autophase_settings[name] = item
        elif name in self.ele:
            self.ele[name][attrib] = item
        elif name in self.group:
            self.group[name][attrib] = item
        else:
            raise ValueError(
                f"{name} does not exist in eles or groups of the Impact object."
            )

    def __str__(self):
        path = self.path
        s = header_str(self.header)
        if self.finished:
            s += "Impact-T finished in " + path
        elif self.configured:
            s += "Impact-T configured in " + path
        else:
            s += "Impact-T not configured."
        return s

    def __repr__(self):
        """
        Simple repr showing the number of particles and the stop z.
        """
        memloc = hex(id(self))
        np = self.header["Np"]
        z = self.stop
        return f"<Impact with {np} particles, stopping at {z} m, at {memloc}>"


def suggested_processor_domain(nz, ny, nproc):
    """
    Heuristic for the processor layout.

    Note from Ji Qiang:
        Normally, the number of grid points Nz/Nprow and Ny/Npcol are kept about
        the same. If Nz=Ny, normally, I will put Npcol > Nprow, e.g. 16 x 8.


    Returns:
        Npcol, Nprow

    """

    a = nz / ny

    # Try to work with pwers of 2
    pr = int(np.floor(np.log(nproc * a) / np.log(2) / 2))

    nr = 2**pr

    if nr < 1:
        nr = 1
    if nr > nproc:
        nr = nproc

    nc = nproc // nr

    return nc, nr


# Default input
# This should be the same as examples/templates/drift/ImpactT.in
DEFAULT_INPUT = {
    "input_particle_file": None,
    "header": {
        "Npcol": 1,
        "Nprow": 1,
        "Dt": 1e-11,
        "Ntstep": 1000000,
        "Nbunch": 1,
        "Dim": 6,
        "Np": 100000,
        "Flagmap": 1,
        "Flagerr": 0,
        "Flagdiag": 2,
        "Flagimg": 0,
        "Zimage": 0.02,
        "Nx": 32,
        "Ny": 32,
        "Nz": 32,
        "Flagbc": 1,
        "Xrad": 0.015,
        "Yrad": 0.015,
        "Perdlen": 45.0,
        "Flagdist": 2,
        "Rstartflg": 0,
        "Flagsbstp": 0,
        "Nemission": 0,
        "Temission": 0.0,
        "sigx(m)": 0.001,
        "sigpx": 0.0,
        "muxpx": 0.0,
        "xscale": 1.0,
        "pxscale": 1.0,
        "xmu1(m)": 0.0,
        "xmu2": 0.0,
        "sigy(m)": 0.001,
        "sigpy": 0.0,
        "muxpy": 0.0,
        "yscale": 1.0,
        "pyscale": 1.0,
        "ymu1(m)": 0.0,
        "ymu2": 0.0,
        "sigz(m)": 0.0001,
        "sigpz": 0.0,
        "muxpz": 0.0,
        "zscale": 1.0,
        "pzscale": 1.0,
        "zmu1(m)": 0.0,
        "zmu2": 19.569511835591836,  # gammma => 10 MeV energy
        "Bcurr": 1.0,
        "Bkenergy": 1.0,
        "Bmass": 510998.95,  # electrons
        "Bcharge": -1.0,
        "Bfreq": 1000000000.0,
        "Tini": 0.0,
    },
    "lattice": [
        {
            "description": "name:2d_to_3d_spacecharge",
            "original": "0 0 0 -5 0 0 -1000.0 /!name:2d_to_3d_spacecharge",
            "type": "rotationally_symmetric_to_3d",
            "s": -1000.0,
            "name": "2d_to_3d_spacecharge",
        },
        {
            "description": "name:drift_1",
            #  'original': '1.0 0 0 0 1.0 0.15 /!name:drift_1',
            "L": 1.0,
            "type": "drift",
            "zedge": 1.0,
            "radius": 0.15,
            "s": 2.0,
            "name": "drift_1",
        },
        {
            "description": "name:stop_1",
            # 'original': '0 0 0 -99 0 0.0 1 /!name:stop_1',
            "type": "stop",
            "s": 1.0,
            "name": "stop_1",
        },
    ],
    "fieldmaps": {},
}
