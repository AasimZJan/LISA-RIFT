"""RIFT Pipeline specification."""

import configparser
import glob
import os
import re
import subprocess

from ligo.gracedb.rest import HTTPError

from asimov import config, logger
from asimov.utils import set_directory

from asimov.pipeline import Pipeline, PipelineException, PipelineLogger
from asimov.pipeline import PESummaryPipeline


class Rift(Pipeline):
    """
    The RIFT Pipeline.

    Parameters
    ----------
    production : :class:`asimov.Production`
       The production object.
    category : str, optional
        The category of the job.
        Defaults to "C01_offline".
    """

    name = "RIFT"
    STATUS = {"wait", "stuck", "stopped", "running", "finished"}

    def __init__(self, production, category=None):
        super(Rift, self).__init__(production, category)
        self.logger = logger
        self.logger.info("Using the RIFT pipeline (rift.py)")
        if 'RIFT_ASIMOV_INI' in os.environ:
            self.config_template = os.getenv('RIFT_ASIMOV_INI')
        else:
            # FIXME: should use importlib in future!
            import RIFT.lalsimutils
            self.config_template = RIFT.lalsimutils.__file__.replace("lalsimutils.py","asimov/rift.ini")
        if not production.pipeline.lower() == "rift":
            raise PipelineException

        if "bootstrap" in self.production.meta:
            self.bootstrap = self.production.meta["bootstrap"]
        else:
            self.bootstrap = False

        self._create_ledger_entries()

    def _create_ledger_entries(self):
        """Create entries in the ledger which might be required in the templating."""
        if "sampler" not in self.production.meta:
            self.production.meta["sampler"] = {}
        required_args = {
            "sampler": {"ile", "cip"},
            "likelihood": {"marginalization", "assume"},
        }
        for section in required_args.keys():
            section_data = self.production.meta[section]
            for section_arg in required_args[section]:
                if section_arg not in section_data:
                    section_data[section_arg] = {}

    def after_completion(self):

        self.logger.info("Job has completed. Running PE Summary.")
        post_pipeline = PESummaryPipeline(production=self.production)
        cluster = post_pipeline.submit_dag()

        self.production.meta["job id"] = int(cluster)
        self.production.status = "processing"

    def before_submit(self):
        # Check for dependencies (mainly for cache files from gwdata)
        productions = {}
        if self.production.dependencies:
          for production in self.production.event.productions:
                productions[production.name] = production
          # concatenate all cache files into main local.cache directory
          for previous_job in self.production.dependencies:
            print("assets", productions[previous_job].pipeline.collect_assets())

            if "caches" in productions[previous_job].pipeline.collect_assets():
                cache_files = productions[previous_job].pipeline.collect_assets()['caches'].values()
                for cache_file in cache_files:
                    os.system(' cat {}  >> {}/local.cache '.format(cache_file, self.production.rundir))

        pass

    def before_config(self, dryrun=False):
        """
        Convert the text-based PSD to an XML psd if the xml doesn't exist already.
        """
        event = self.production.event
        category = config.get("general", "calibration_directory")
        self.logger.info("Checking for XML format PSDs")
        if len(self.production.get_psds("xml")) == 0 and "psds" in self.production.meta:
            self.logger.info("Did not find XML format PSDs")
            for ifo in self.production.meta["interferometers"]:
                with set_directory(f"{event.work_dir}"):
                    sample = self.production.meta["likelihood"]["sample rate"]
                    self.logger.info(f"Converting {ifo} {sample}-Hz PSD to XML")
                    self._convert_psd(
                        self.production.meta["psds"][sample][ifo], ifo, dryrun=dryrun
                    )
                    asset = f"{ifo.upper()}-psd.xml.gz"
                    self.logger.info(f"Conversion complete as {asset}")
                    git_location = os.path.join(category, "psds")
                    saveloc = os.path.join(
                        git_location, str(sample), f"psd_{ifo}.xml.gz"
                    )
                    self.production.event.repository.add_file(
                        asset,
                        saveloc,
                        commit_message=f"Added the xml format PSD for {ifo}.",
                    )
                    self.logger.info(f"Saved at {saveloc}")

    def build_dag(self, user=None, dryrun=False):
        """
        Construct a DAG file in order to submit a production to the
        condor scheduler using util_RIFT_pseudo_pipe.py

        Parameters
        ----------
        user : str
           The user accounting tag which should be used to run the job.
        dryrun: bool
           If set to true the commands will not be run, but will be printed to standard output. Defaults to False.

        Raises
        ------
        PipelineException
           Raised if the construction of the DAG fails.

        Notes
        -----

        In order to assemble the pipeline the RIFT runner requires additional
        production metadata: at least the l_max value.
        An example RIFT production specification would then look something like:

        ::

           - Prod3:
               rundir: tests/tmp/s000000xx/Prod3
               pipeline: rift
               waveform:
                  approximant: IMRPhenomPv3
               lmax: 2
               cip jobs: 5 # This is optional, and will default to 3
               bootstrap: Prod1
               bootstrap fmin: 20
               needs:
                 - Prod1
               comment: RIFT production run.
               status: ready


        """
        self.before_build()
        cwd = os.getcwd()
        if self.production.event.repository:
            self.logger.info("Checking for existence of coinc file in event repository")
            try:
                coinc_file = self.production.get_coincfile()
                calibration = config.get("general", "calibration_directory")
                coinc_file = os.path.abspath(coinc_file)
                self.logger.info(f"Coinc found at {coinc_file}")
            except HTTPError:
                print(
                    "Unable to download the coinc file because it was not possible to connect to GraceDB"
                )
                self.logger.warning(
                    "Could not download a coinc file for this event; could not connect to GraceDB."
                )
                coinc_file = None
            except ValueError:
                self.logger.warning(
                    "Could not download a coinc file for this event as no GraceDB ID was supplied."
                )
                coinc_file = None
            try:
                ini = self.production.get_configuration().ini_loc
                calibration = config.get("general", "calibration_directory")
                self.logger.info(f"Using the {calibration} calibration")
                ini = os.path.join(
                    self.production.event.repository.directory, calibration, ini
                )
                ini = os.path.abspath(ini)
            except ValueError:
                print(
                    "Unable to find the configuration file. Have you run `$ asimov manage build` yet?"
                )
                ini = "INI MISSING"
        else:
            ini = "INI MISSING"
            coinc_file = os.path.join(cwd, "coinc.xml")

        if self.production.get_meta("user"):
            user = self.production.get_meta("user")
        else:
            user = config.get("condor", "user")
            self.production.set_meta("user", user)

        os.environ["LIGO_USER_NAME"] = f"{user}"
        os.environ[
            "LIGO_ACCOUNTING"
        ] = f"{self.production.meta['scheduler']['accounting group']}"
        if 'avoid hosts' in self.production.meta['scheduler']:
            val = ""
            for name in self.production.meta['scheduler']['avoid hosts']:
                val += "{},".format(name)
            if len(val)>0:
                val = val[:-1] # remove last comma
            os.environ['RIFT_AVOID_HOSTS'] = val
        if 'gpu architectures' in self.production.meta['scheduler']:
            val =""
            for name in self.production.meta['scheduler']['gpu architectures']:
                val += '(DeviceName=!="{}")&&'.format(name)
            if len(val)>0:
                val = val[:-2] # remove last two and
            os.environ['RIFT_REQUIRE_GPUS'] = val

        if "singularity image" in self.production.meta["scheduler"]:
            # Collect the correct information for the singularity image
            os.environ[
                "SINGULARITY_RIFT_IMAGE"
            ] = f"{self.production.meta['scheduler']['singularity image']}"
            os.environ[
                "SINGULARITY_BASE_EXE_DIR"
            ] = f"{self.production.meta['scheduler']['singularity base exe directory']}"

        try:
            calibration = config.get("general", "calibration")
        except configparser.NoOptionError:
            calibration = "C01"

        try:
            approximant = self.production.meta["waveform"]["approximant"]
        except KeyError:
            self.logger.error(
                "Could not find a waveform approximant specified for this job."
            )

        if self.production.rundir:
            rundir = os.path.abspath(self.production.rundir)
        else:
            rundir = os.path.join(
                os.path.expanduser("~"),
                self.production.event.name,
                self.production.name,
            )
            self.production.rundir = rundir

        # lmax = self.production.meta['priors']['amp order']

        command = [
            os.path.join(
                config.get("pipelines", "environment"),
                "bin",
                "util_RIFT_pseudo_pipe.py",
            ),
        ]
        if coinc_file:
            command += ["--use-coinc", coinc_file]

        if "non-spin" in self.production.meta["waveform"]:
            if self.production.meta["waveform"]["non-spin"]:
                command += ["--assume-nospin"]

        command += [
            "--calibration",
            f"{calibration}",
            "--approx",
            f"{approximant}",
            "--use-rundir",
            rundir,
            "--ile-force-gpu",
            "--use-ini",
            ini,
        ]

        if "pipeline" in self.production.meta["scheduler"]:
            # ini file specifications
            for key in self.production.meta["scheduler"]["pipeline"].keys():
                value = self.production.meta["scheduler"]["pipeline"][f"{key}"]
                if value == "True" or value == "true" or value == True:
                    command += [f"--{key}"]
                elif value == "False" or value == "false" or value == False:
                    pass
                else:
                    command += [f"--{key}={value}"]

        # If a starting frequency is specified, add it
        if "start-frequency" in self.production.meta:
            command += ["--fmin-template", self.production.quality["start-frequency"]]

        # Placeholder LI grid bootstrapping; conditional on it existing and location specification

        if self.bootstrap:
            if self.bootstrap == "manual":
                if self.production.event.repository:
                    bootstrap_file = os.path.join(
                        self.production.event.repository.directory,
                        "C01_offline",
                        f"{self.production.name}_bootstrap.xml.gz",
                    )
                else:
                    bootstrap_file = "{self.production.name}_bootstrap.xml.gz"
            else:
                raise PipelineException(
                    f"Unable to find the bootstrapping production for {self.production.name}.",
                    issue=self.production.event.issue_object,
                    production=self.production.name,
                )

            command += ["--manual-initial-grid", bootstrap_file]

        if "scheduler" in self.production.meta:
            if "osg" in self.production.meta["scheduler"]:
                if self.production.meta["scheduler"]["osg"]:
                    command += ["--use-osg-file-transfer"]

#        print(" ".join(command))
        if dryrun:
            print(" ".join(command))

        else:
            self.logger.info(" ".join(command))

            with set_directory(self.production.event.work_dir):
                pipe = subprocess.Popen(
                    command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT
                )
                out, err = pipe.communicate()
                # very large amount of information for asimov log !
                self.logger.info(out)
                if err:
                    self.production.status = "stuck"
                    if hasattr(self.production.event, "issue_object"):
                        self.logger.info(out, production=self.production)
                        self.logger.error(err, production=self.production)
                        raise PipelineException(
                            f"DAG file could not be created.\n{command}\n{out}\n\n{err}",
                            issue=self.production.event.issue_object,
                            production=self.production.name,
                        )
                    else:
                        self.logger.info(out, production=self.production)
                        self.logger.error(err, production=self.production)
                        raise PipelineException(
                            f"DAG file could not be created.\n{command}\n{out}\n\n{err}",
                            production=self.production.name,
                        )
                else:
                    dagfile = os.path.join(rundir,'marginalize_intrinsic_parameters_BasicIterationWorkflow.dag')
                    if not(os.path.exists(dagfile)):
                        self.production.status = "stuck"
                        self.logger.info(out) #, production=self.production)
                        self.logger.error(err) # , production=self.production)
                        raise PipelineException(f"DAG file could not be created.\n{command}\n{out}\n\n{err}",
                            production=self.production.name,
                        )
                    if self.production.event.repository:
                        # with set_directory(os.path.abspath(self.production.rundir)):
                        for psdfile in self.production.get_psds("xml"):
                            ifo = psdfile.split("/")[-1].split("-")[1].split(".")[0]
                            os.system(f"cp {psdfile} {ifo}-psd.xml.gz")

                        # os.system("cat *_local.cache > local.cache")

                        if hasattr(self.production.event, "issue_object"):
                            return PipelineLogger(
                                message=out,
                                issue=self.production.event.issue_object,
                                production=self.production.name,
                            )
                        else:
                            return PipelineLogger(
                                message=out, production=self.production.name
                            )

    def submit_dag(self, dryrun=False):
        """
        Submit a DAG file to the condor cluster (using the RIFT dag name).
        This is an overwrite of the near identical parent function submit_dag()

        Parameters
        ----------
        category : str, optional
           The category of the job.
           Defaults to "C01_offline".
        production : str
           The production name.

        Returns
        -------
        int
           The cluster ID assigned to the running DAG file.
        PipelineLogger
           The pipeline logger message.

        Raises
        ------
        PipelineException
           This will be raised if the pipeline fails to submit the job.
        """
        self.before_submit()
        for psdfile in self.production.get_psds("xml"):
            ifo = psdfile.split("/")[-1].split("-")[1].split(".")[0]
            os.system(f"cp {psdfile} {ifo}-psd.xml.gz")

        command = [
            "condor_submit_dag",
            "-batch-name",
            f"rift/{self.production.event.name}/{self.production.name}",
            "marginalize_intrinsic_parameters_BasicIterationWorkflow.dag",
        ]
        if dryrun:
            for psdfile in self.production.get_psds("xml"):
                print(f"cp {psdfile} {self.production.rundir}/{psdfile.split('/')[-1]}")
            print("")
            print(" ".join(command))
        else:
            for psdfile in self.production.get_psds("xml"):
                os.system(
                    f"cp {psdfile} {self.production.rundir}/{psdfile.split('/')[-1]}"
                )

            try:
                with set_directory(self.production.rundir):

                    dagman = subprocess.Popen(
                        command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT
                    )
                    self.logger.info(command, production=self.production)
            except FileNotFoundError as exception:
                raise PipelineException(
                    "It looks like condor isn't installed on this system.\n"
                    f"""I wanted to run {" ".join(command)}."""
                ) from exception

            stdout, stderr = dagman.communicate()

            if "submitted to cluster" in str(stdout):
                cluster = re.search(
                    r"submitted to cluster ([\d]+)", str(stdout)
                ).groups()[0]
                self.production.status = "running"
                self.production.job_id = int(cluster)
                return cluster, PipelineLogger(stdout)
            else:
                raise PipelineException(
                    f"The DAG file could not be submitted.\n\n{stdout}\n\n{stderr}",
                    issue=self.production.event.issue_object,
                    production=self.production.name,
                )

    def html(self):
        """Return the HTML representation of this pipeline."""
        pages_dir = os.path.join(
            self.production.event.name, self.production.name, "pesummary"
        )
        out = ""
        if self.production.status in {"uploaded"}:
            out += """<div class="asimov-pipeline">"""
            out += f"""<p><a href="{pages_dir}/home.html">Summary Pages</a></p>"""
            out += f"""<img height=200 src="{pages_dir}/plots/{self.production.name}_psd_plot.png"</src>"""
            out += f"""<img height=200 src="{pages_dir}/plots/{self.production.name}_waveform_time_domain.png"</src>"""

            out += """</div>"""

        return out

    def resurrect(self):
        """
        Attempt to ressurrect a failed job.
        """
        try:
            count = self.production.meta["resurrections"]
        except KeyError:
            count = 0
        count = len(
            glob.glob(
                os.path.join(
                    self.production.rundir,
                    "marginalize_intrinsic_parameters_BasicIterationWorkflow.dag.rescue*",
                )
            )
        )
        # check if dagman.out  newer than rescue - job has not really died? Prevent superfluous restarts!
        time_mod_out = os.path.getmtime(                os.path.join(
                    self.production.rundir,
                    "marginalize_intrinsic_parameters_BasicIterationWorkflow.dag.dagman.out",
                ))
        last_rescue =        glob.glob(
                os.path.join(
                    self.production.rundir,
                    "marginalize_intrinsic_parameters_BasicIterationWorkflow.dag.rescue*",
                )
            ).sort()[-1]
        time_mod_rescue = os.path.getmtime(last_rescue)
        if count < 100 and (time_mod_out > time_mod_rescue+30): # some buffer in seconds for file i/o
            print("   ... still going, leaving it alone ")
            return None
        if "allow ressurect" in self.production.meta:
            count = 0
        if (count < 90) and (
            len(
                glob.glob(
                    os.path.join(
                        self.production.rundir,
                        "marginalize_intrinsic_parameters_BasicIterationWorkflow.dag.rescue*",
                    )
                )
            )
            > 0
        ):
            count += 1
            # os.system("cat *_local.cache > local.cache")
            self.submit_dag()
        elif count ==0:
            # user has most likely intervened, deleted al the rescue files, and we are about to start new, OR we have 'allow resurrect'
            print( "    - starting rift from scratch ? ")
            self.submit_dag()
        else:
            raise PipelineException(
                "This job was resurrected too many times.",
                issue=self.production.event.issue_object,
                production=self.production.name,
            )

    def collect_logs(self):
        """
        Collect all of the log files which have been produced by this production and
        return their contents as a dictionary.
        """
        logs = glob.glob(
            f"{self.production.rundir}/*.err"
        )  # + glob.glob(f"{self.production.rundir}/*/logs/*")
        logs += glob.glob(f"{self.production.rundir}/*.out")
        messages = {}
        for log in logs:
            with open(log, "r") as log_f:
                message = log_f.read()
                messages[log.split("/")[-1]] = message
        return messages

    def detect_completion(self):
        """
        Check for the production of the posterior file to signal that the job has completed.
        """
        # check if calmarg requested -- different filename
        calmarg_requested = False
        if 'likelihood' in self.production.meta['sampler']:
            if 'calibration' in self.production.meta['sampler']['likelihood']:
                calmarg_requested = self.production.meta['sampler']['likelihood']['calibration']['sample']
        fname_last = "extrinsic_posterior_samples.dat"
        if calmarg_requested:
            fname_last = 'reweighted_posterior_samples.dat'

        results_dir = glob.glob(f"{self.production.rundir}")
        if len(results_dir) > 0:  # dynesty_merge_result.json
            if (
                len(
                    glob.glob(
                        os.path.join(results_dir[0], fname_last)
                    )
                )
                > 0
            ):
                return True
            else:
                return False
        else:
            return False

    def collect_assets(self,absolute=False):
        """
        Gather all of the results assets for this job.
        """
        if absolute:
            rundir = os.path.abspath(self.production.rundir)
        else:
            rundir = self.production.rundir
        rift_all_lnL = os.path.join(rundir, 'all.net')
        samples_raw = os.path.join(rundir,'extrinsic_posterior_samples.dat')
        dict_out = {"samples":self.samples(), "lnL_marg":rift_all_lnL, "samples_raw":samples_raw}
        rewt_file_name  = os.path.join(rundir,'reweighted_posterior_samples.dat')
        if os.path.exists(rewt_file_name):
            dict_out['samples_calmarg'] = rewt_file_name
            dict_out['samples'] = rewt_file_name

        return dict_out


    def samples(self, absolute=False):
        """
        Collect the combined samples file for PESummary.
        """

        if absolute:
            rundir = os.path.abspath(self.production.rundir)
        else:
            rundir = self.production.rundir
        return glob.glob(os.path.join(rundir, "extrinsic_posterior_samples.dat"))
