import numpy as np
import pandas as pd

from classes.patient_array import Patient_array

class SSNAP_Pathway:
    """
    Model of stroke pathway.

    Each scenario mimics 100 years of a stroke pathway. Patient times through
    the pathway are sampled from distributions passed to the model using NumPy.

    Array columns:
     0: Patient aged 80+
     1: Allowable onset to needle time (may depend on age)
     2: Onset time known (boolean)
     3: Onset to arrival is less than 4 hours (boolean)
     4: Onset known and onset to arrival is less than 4 hours (boolean)
     5: Onset to arrival minutes
     6: Arrival to scan is less than 4 hours
     7: Arrival to scan minutes
     8: Minutes left to thrombolyse
     9: Onset time known and time left to thrombolyse
    10: Proportion ischaemic stroke (if they are filtered at this stage)
    11: Assign eligible for thrombolysis (for those scanned within 4 hrs of onset)
    12: Thrombolysis planned (scanned within time and eligible)
    13: Scan to needle time
    14: Clip onset to thrombolysis time to maximum allowed onset-to-thrombolysis



    WRITE A NEW HUGE DOCSTRING PLEASE WITH EVERYTHING THIS NOW DOES #############################################################


    IVT = intra-veneous thrombolysis
    MT = mechanical thrombectomy
    """
    # Settings that are common across all stroke teams and all trials:

    # Assume these patient proportions:
    proportion_lvo = 0.35
    proportion_nlvo = 0.65
    # If these do not sum to 1, the remainder will be assigned to
    # all other stroke types combined (e.g. haemorrhage).
    # They're not subdivided more finely because the outcome model
    # can only use nLVO and LVO at present (May 2023).

    # Assume these time limits for the checks at each point
    # (e.g. is onset to arrival time below 4 hours?)
    limit_ivt_mins = 4*60
    limit_mt_mins = 8*60

    # Set up allowed time and over-run for thrombolysis...
    allowed_onset_to_needle_time_mins = 270  # 4h 30m
    allowed_overrun_for_slow_scan_to_needle_mins = 15
    # ... and for thrombectomy
    allowed_onset_to_puncture_time_mins = 8*60  # --------------------------------- need to check for a reaonsable number here
    allowed_overrun_for_slow_scan_to_puncture_mins = 15

    def __init__(
            self,
            hospital_name: str,
            hospital_data: pd.DataFrame
            ):
        """
        Sets up the data required to calculate the patient pathways.

        Inputs:
        -------
        hospital_name - str. Label for this hospital.
        hospital_data - pd.DataFrame. See the function
                        self._run_sanity_check_on_hospital_data() to
                        see the expected contents of this dataframe.

        Initialises:
        ------------
        Generic time limits in minutes:
        - allowed onset to needle time (thrombolysis)
        - allowed overrun for slow scan to needle (thrombolysis)
        - allowed onset to puncture time (thrombectomy)
        - allowed overrun for slow scan to puncture (thrombectomy)

        Hospital-specific data:
        - hospital name
        - patients per run

        Proportions of patients in the following categories:
        - onset time known
        - known arrival within 4 hours
        - scan within 4 hours
        - chosen for thrombolysis
        - chosen for thrombectomy

        Log-normal number generation parameters mu and sigma for
        each of the following:
        - onset to arrival time
        - arrival to scan time
        - scan to needle time (thrombolysis)
        - scan to puncture time (thrombectomy)
        mu is the mean and sigma the standard deviation of the
        hospital's performance in these times in log-normal space.
        """

        try:
            self.hospital_name = str(hospital_name)
        except TypeError:
            print('Hospital name should be a string. Name not set.')
            self.hospital_name = ''

        # Run sanity checks.
        # If the data has problems, these raise an exception.
        self._run_sanity_check_on_hospital_data(hospital_data)

        # From hospital data:
        self.patients_per_run = int(hospital_data['admissions'])

        # Patient population:
        self.real_data_dict = dict(
            proportion_onset_known = \
                hospital_data['onset_known'],
            # IVT patient proportions
            proportion_known_arrival_on_time_ivt = \
                hospital_data['known_arrival_within_4hrs'],
            proportion_arrival_to_scan_on_time_ivt = \
                hospital_data['scan_within_4_hrs'],
            proportion_onset_to_scan_on_time_ivt = \
                hospital_data['onset_scan_4_hrs'],
            proportion_scan_to_needle_on_time = \
                hospital_data['scan_needle_4_hrs'],
            proportion_chosen_for_ivt = \
                hospital_data['proportion_chosen_for_thrombolysis'],
            # MT patient proportions
            proportion_known_arrival_on_time_mt = \
                hospital_data['known_arrival_within_8hrs'],
            proportion_arrival_to_scan_on_time_mt = \
                hospital_data['scan_within_8_hrs'],
            proportion_onset_to_scan_on_time_mt = \
                hospital_data['onset_scan_8_hrs'],
            proportion_scan_to_puncture_on_time = \
                hospital_data['scan_puncture_8_hrs'],
            proportion_chosen_for_mt = \
                hospital_data['proportion_chosen_for_thrombectomy'],
            # GET THI SOUT OF THE HOSPITAL PERFORAMCNE DATA INSTEAD #######################################
            proportion_of_mt_also_receiving_ivt = 1.0,
            # IVT log-normal number generation:
            lognorm_mu_onset_arrival_ivt_mins = \
                hospital_data['onset_arrival_mins_mu'],
            lognorm_sigma_onset_arrival_ivt_mins = \
                hospital_data['onset_arrival_mins_sigma'],
            lognorm_mu_arrival_scan_arrival_ivt_mins = \
                hospital_data['arrival_scan_arrival_mins_mu'],
            lognorm_sigma_arrival_scan_arrival_ivt_mins = \
                hospital_data['arrival_scan_arrival_mins_sigma'],
            lognorm_mu_scan_needle_mins = \
                hospital_data['scan_needle_mins_mu'],
            lognorm_sigma_scan_needle_mins = \
                hospital_data['scan_needle_mins_sigma'],
            # MT log-normal number generation:
            lognorm_mu_onset_arrival_mt_mins = \
                hospital_data['onset_arrival_thrombectomy_mins_mu'],
            lognorm_sigma_onset_arrival_mt_mins = \
                hospital_data['onset_arrival_thrombectomy_mins_sigma'],
            lognorm_mu_arrival_scan_arrival_mt_mins = \
                hospital_data['arrival_scan_arrival_thrombectomy_mins_mu'],
            lognorm_sigma_arrival_scan_arrival_mt_mins = \
                hospital_data['arrival_scan_arrival_thrombectomy_mins_sigma'],
            lognorm_mu_scan_puncture_mins = \
                hospital_data['scan_puncture_mins_mu'],
            lognorm_sigma_scan_puncture_mins = \
                hospital_data['scan_puncture_mins_sigma'],
        )
        
        # The outputs will go in this dictionary:
        # Patient_array(
        #    number_of_patients, valid_dtypes_list, valid_min, valid_max)
        n = self.patients_per_run  # Defined to shorten the following.
        self.trial = dict(
            #                                                                #
            # Initial steps                                                  #
            onset_to_arrival_mins = (
                Patient_array(n, ['float'], 0.0, np.inf)),
            onset_to_arrival_within_limit_ivt_bool = (
                Patient_array(n, ['int', 'bool'], 0, 1)),
            onset_to_arrival_within_limit_mt_bool = (
                Patient_array(n, ['int', 'bool'], 0, 1)),
            onset_time_known_bool = (
                Patient_array(n, ['int', 'bool'], 0, 1)),
            arrival_to_scan_mins = (
                Patient_array(n, ['float'], 0.0, np.inf)),
            arrival_to_scan_within_limit_ivt_bool = (
                Patient_array(n, ['int', 'bool'], 0, 1)),
            arrival_to_scan_within_limit_mt_bool = (
                Patient_array(n, ['int', 'bool'], 0, 1)),
            onset_to_scan_mins = (
                Patient_array(n, ['float'], 0.0, np.inf)),
            #                                                                #
            # IVT (thrombolysis)
            onset_to_scan_within_limit_ivt_bool = (
                Patient_array(n, ['int', 'bool'], 0, 1)),
            time_left_for_ivt_after_scan_mins = (
                Patient_array(n, ['float'], 0.0, np.inf)),
            enough_time_for_ivt_bool = (
                Patient_array(n, ['int', 'bool'], 0, 1)),
            scan_to_needle_mins = (
                Patient_array(n, ['float'], 0.0, np.inf)),
            scan_to_needle_within_limit_ivt_bool = (
                Patient_array(n, ['int', 'bool'], 0, 1)),
            scan_to_needle_within_limit_mt_bool = (
                Patient_array(n, ['int', 'bool'], 0, 1)), #################### ??
            onset_to_needle_mins = (
                Patient_array(n, ['float'], 0.0, np.inf)),
            ivt_chosen_bool = (
                Patient_array(n, ['int', 'bool'], 0, 1)),
            #                                                                #
            # Masks of IVT pathway to match input hospital performance       #
            ivt_mask1_onset_known = (
                Patient_array(n, ['int', 'bool'], 0, 1)),
            ivt_mask2_mask1_and_onset_to_arrival_on_time = (
                Patient_array(n, ['int', 'bool'], 0, 1)),
            ivt_mask3_mask2_and_arrival_to_scan_on_time = (
                Patient_array(n, ['int', 'bool'], 0, 1)),
            ivt_mask4_mask3_and_onset_to_scan_on_time = (
                Patient_array(n, ['int', 'bool'], 0, 1)),
            ivt_mask5_mask4_and_enough_time_to_treat = (
                Patient_array(n, ['int', 'bool'], 0, 1)), 
            ivt_mask6_mask5_and_treated = (
                Patient_array(n, ['int', 'bool'], 0, 1)),
            #                                                                #
            # MT (thrombectomy)
            onset_to_scan_within_limit_mt_bool = (
                Patient_array(n, ['int', 'bool'], 0, 1)),
            time_left_for_mt_after_scan_mins = (
                Patient_array(n, ['float'], 0.0, np.inf)),
            enough_time_for_mt_bool = (
                Patient_array(n, ['int', 'bool'], 0, 1)),
            scan_to_puncture_mins = (
                Patient_array(n, ['float'], 0.0, np.inf)),
            scan_to_puncture_within_limit_ivt_bool = (
                Patient_array(n, ['int', 'bool'], 0, 1)), ################# ?
            scan_to_puncture_within_limit_mt_bool = (
                Patient_array(n, ['int', 'bool'], 0, 1)),
            onset_to_puncture_mins = (
                Patient_array(n, ['float'], 0.0, np.inf)),
            mt_chosen_bool = (
                Patient_array(n, ['int', 'bool'], 0, 1)),
            #                                                                #
            # Masks of MT pathway to match input hospital performance        #
            mt_mask1_onset_known = (
                Patient_array(n, ['int', 'bool'], 0, 1)),
            mt_mask2_mask1_and_onset_to_arrival_on_time = (
                Patient_array(n, ['int', 'bool'], 0, 1)),
            mt_mask3_mask2_and_arrival_to_scan_on_time = (
                Patient_array(n, ['int', 'bool'], 0, 1)),
            mt_mask4_mask3_and_onset_to_scan_on_time = (
                Patient_array(n, ['int', 'bool'], 0, 1)),
            mt_mask5_mask4_and_enough_time_to_treat = (
                Patient_array(n, ['int', 'bool'], 0, 1)),
            mt_mask6_mask5_and_treated = (
                Patient_array(n, ['int', 'bool'], 0, 1)),
            #                                                                #
            # Use the treatment decisions to assign stroke type              #
            stroke_type_code = (
                Patient_array(n, ['int'], 0, 2)),
        )

    #
    def run_trial(self, patients_per_run=0):
        """
        The big loopy.

        Create a new attribute for each new "column" of data.
        Store the new info as many 1D arrays instead of the
        usual single 2D grid.

        Creates the following attributes:
        - trial_onset_time_known_bool
        - trial_onset_to_arrival_within_limit_ivt_bool
        - trial_onset_to_arrival_mins
        - trial_arrival_to_scan_within_limit_ivt_bool
        - trial_arrival_to_scan_mins
        - trial_time_left_for_ivt_after_scan_mins
        - trial_enough_time_for_ivt_bool
        - trial_ischaemic_bool
        - trial_ivt_chosen_bool
        - trial_ivt_conditions_met_bool
        - trial_scan_to_needle_mins
        - trial_onset_to_needle_mins
        - trial_ivt_rate_percent

        """
        if patients_per_run > 0:
            # Overwrite the default value from the hospital data.
            self.patients_per_run = patients_per_run
        else:
            # Don't update anything.
            pass

        # Generate pathway times for all patients.
        # These sets of times are all independent of each other.
        self._generate_onset_to_arrival_time_lognorm()
        # Assign randomly whether the onset time is known
        # in the same proportion as the real performance data.
        self._generate_onset_time_known_binomial()
        # Where onset time is not known, the arrays relating to onset
        # to arrival times are updated. Times are set to Not A Number
        # and boolean "is time below x hours" are set to 0 (=False).
        self._create_masks_onset_to_arrival_on_time()
        # Other generated times:
        self._generate_arrival_to_scan_time_lognorm()
        self._generate_scan_to_needle_time_lognorm()
        self._generate_scan_to_puncture_time_lognorm()


        # Calculate more times from the previously-generated times.
        self._calculate_onset_to_scan_time()
        # Thrombolysis:
        self._calculate_time_left_for_ivt_after_scan()
        self._calculate_and_clip_onset_to_needle_time(clip_limit_mins=(
            self.allowed_onset_to_needle_time_mins +
            self.allowed_overrun_for_slow_scan_to_needle_mins
            ))
        # Thrombectomy:
        self._calculate_time_left_for_mt_after_scan()
        self._calculate_and_clip_onset_to_puncture_time(clip_limit_mins=(
            self.allowed_onset_to_puncture_time_mins +
            self.allowed_overrun_for_slow_scan_to_puncture_mins
            ))
        self._create_masks_enough_time_to_treat()

        # Check that the generated times are reasonable:
        self._sanity_check_generated_times_patient_proportions()

        # Treatment decision
        self._generate_whether_ivt_chosen_binomial()
        self._generate_whether_mt_chosen_binomial()
        self._create_masks_treatment_given()

        # Assign which type of stroke it is *after* choosing which
        # treatments are given.
        self._assign_stroke_type_code()

        # Place all useful outputs into a pandas Dataframe:
        df = self._gather_results_in_dataframe()
        return df


    # ###################################
    # ##### PATHWAY TIME GENERATION #####
    # ###################################

    def _generate_lognorm_times(
            self,
            mu_mt,
            sigma_mt,
            mu_ivt,
            sigma_ivt,
            proportion_within_mt_limit,
            number_of_patients,
            label_for_printing=''
            ):
        """ MEND ME """

        times_mins = np.random.lognormal(
            mu_mt,     # Mean
            sigma_mt,  # Standard dev.
            number_of_patients   # n of samples
            )
        
        # Round times to nearest minute:
        times_mins = np.round(times_mins, 0)
        # Set times below zero to zero:
        times_mins = np.maximum(times_mins, 0)

        times_mins = self._fudge_patients_after_time_limit_mt(
            times_mins,
            proportion_within_mt_limit,
            number_of_patients
            )

        # Sanity checks:
        self._check_distribution_statistics(
            times_mins[times_mins <= self.limit_mt_mins],
            mu_mt,
            sigma_mt,
            label=label_for_printing + ' on time for thrombectomy'
            )
        if mu_mt != mu_ivt and sigma_mt != sigma_ivt:
            self._check_distribution_statistics(
                times_mins[times_mins <= self.limit_ivt_mins],
                mu_ivt,
                sigma_ivt,
                label=label_for_printing + ' on time for thrombolysis'
                )
        
        return times_mins


    def _generate_onset_to_arrival_time_lognorm(self):
        """
        Assign onset to arrival time (natural log normal distribution).

        Creates:
        --------
        trial_onset_to_arrival_mins -
            Onset to arrival times in minutes from the log-normal
            distribution. One time per patient.
        trial_onset_to_arrival_within_limit_ivt_bool -
            True or False for each patient arriving under
            the time limit for thrombolysis.
        trial_onset_to_arrival_within_limit_mt_bool -
            True or False for each patient arriving under
            the time limit for thrombectomy.
        """
        # Invent new times for the patient subgroup:
        trial_onset_to_arrival_mins = self._generate_lognorm_times(
            self.real_data_dict['lognorm_mu_onset_arrival_mt_mins'],
            self.real_data_dict['lognorm_sigma_onset_arrival_mt_mins'],
            self.real_data_dict['lognorm_mu_onset_arrival_ivt_mins'],
            self.real_data_dict['lognorm_sigma_onset_arrival_ivt_mins'],
            self.real_data_dict['proportion_known_arrival_on_time_mt'],
            self.patients_per_run,
            'onset to arrival'
            )

        # Store the generated times:
        self.trial['onset_to_arrival_mins'].data = \
            trial_onset_to_arrival_mins

        # Also generate and store boolean arrays of
        # whether the times are below given limits.
        self.trial['onset_to_arrival_within_limit_ivt_bool'].data = \
            (trial_onset_to_arrival_mins <= self.limit_ivt_mins)
        self.trial['onset_to_arrival_within_limit_mt_bool'].data = \
            (trial_onset_to_arrival_mins <= self.limit_mt_mins)


    def _generate_arrival_to_scan_time_lognorm(self):
        """
        Assign arrival to scan time (natural log normal distribution).
        """
        # Invent new times for the patient subgroup:
        trial_arrival_to_scan_mins = self._generate_lognorm_times(
            self.real_data_dict['lognorm_mu_arrival_scan_arrival_mt_mins'],
            self.real_data_dict['lognorm_sigma_arrival_scan_arrival_mt_mins'],
            self.real_data_dict['lognorm_mu_arrival_scan_arrival_ivt_mins'],
            self.real_data_dict['lognorm_sigma_arrival_scan_arrival_ivt_mins'],
            self.real_data_dict['proportion_arrival_to_scan_on_time_mt'],
            self.patients_per_run,
            'arrival to scan'
            )

        # Store the generated times:
        self.trial['arrival_to_scan_mins'].data = \
            trial_arrival_to_scan_mins

        # Also generate and store boolean arrays of
        # whether the times are below given limits.
        self.trial['arrival_to_scan_within_limit_ivt_bool'].data = \
            (trial_arrival_to_scan_mins <= self.limit_ivt_mins)
        self.trial['arrival_to_scan_within_limit_mt_bool'].data = \
            (trial_arrival_to_scan_mins <= self.limit_mt_mins)

        self._create_masks_arrival_to_scan_on_time()

    def _generate_scan_to_needle_time_lognorm(self):
        """h"""
        # Invent new times for the patient subgroup:
        trial_scan_to_needle_mins = self._generate_lognorm_times(
            self.real_data_dict['lognorm_mu_scan_needle_mins'],
            self.real_data_dict['lognorm_sigma_scan_needle_mins'],
            self.real_data_dict['lognorm_mu_scan_needle_mins'],
            self.real_data_dict['lognorm_sigma_scan_needle_mins'],
            self.real_data_dict['proportion_scan_to_needle_on_time'],
            self.patients_per_run,
            'scan to needle'
            )
        # Store the generated times:
        self.trial['scan_to_needle_mins'].data = \
            trial_scan_to_needle_mins

        # Also generate and store boolean arrays of
        # whether the times are below given limits.
        self.trial['scan_to_needle_within_limit_ivt_bool'].data = \
            (trial_scan_to_needle_mins <= self.limit_ivt_mins)
        self.trial['scan_to_needle_within_limit_mt_bool'].data = \
            (trial_scan_to_needle_mins <= self.limit_mt_mins)


    def _generate_scan_to_puncture_time_lognorm(self):
        """h"""
        # Invent new times for the patient subgroup:
        trial_scan_to_puncture_mins = self._generate_lognorm_times(
            self.real_data_dict['lognorm_mu_scan_puncture_mins'],
            self.real_data_dict['lognorm_sigma_scan_puncture_mins'],
            self.real_data_dict['lognorm_mu_scan_puncture_mins'],
            self.real_data_dict['lognorm_sigma_scan_puncture_mins'],
            self.real_data_dict['proportion_scan_to_puncture_on_time'],
            self.patients_per_run,
            'scan to puncture'
            )
        # Store the generated times:
        self.trial['scan_to_puncture_mins'].data = trial_scan_to_puncture_mins

        # Also generate and store boolean arrays of
        # whether the times are below given limits.
        self.trial['scan_to_puncture_within_limit_ivt_bool'].data = \
            (trial_scan_to_puncture_mins <= self.limit_ivt_mins)
        self.trial['scan_to_puncture_within_limit_mt_bool'].data = \
            (trial_scan_to_puncture_mins <= self.limit_mt_mins)


    # #######################################
    # ##### PATHWAY BINOMIAL GENERATION #####
    # #######################################
    def _generate_onset_time_known_binomial(self):
        """Assign onset time known.

        Run this after:
        ...


        things
        """
        self.trial['onset_time_known_bool'] = \
            np.random.binomial(
                1,                            # Number of trials
                self.real_data_dict['proportion_onset_known'],  # Probability of success
                self.patients_per_run         # Number of samples drawn
                ) == 1

        # Update onset-to-arrival times and relevant boolean
        # arrays so that when onset time is unknown, the values
        # are set to NaN (time) or 0 (boolean).
        inds = (self.trial['onset_time_known_bool'].data == 0)
        self.trial['onset_to_arrival_mins'].data[inds] = np.NaN
        self.trial['onset_to_arrival_within_limit_ivt_bool'].data[inds] = 0
        self.trial['onset_to_arrival_within_limit_mt_bool'].data[inds] = 0

        self._create_masks_onset_time_known()

    def _generate_whether_ivt_chosen_binomial(self):
        """
        need mask run already
        """
        trial_ivt_chosen_bool = np.zeros(self.patients_per_run, dtype=int)

        inds_treated = np.where(
            self.trial['ivt_mask5_mask4_and_enough_time_to_treat'].data == 1)[0]
        n_treated = len(inds_treated)

        # Eligable for thrombolysis (proportion of ischaemic patients
        # eligable for thrombolysis when scanned within 4 hrs )
        ivt_chosen_bool = \
            np.random.binomial(
                1,                     # Number of trials
                self.real_data_dict['proportion_chosen_for_ivt'],
                                       # ^ Probability of success
                n_treated              # Number of samples drawn
                )
        trial_ivt_chosen_bool[inds_treated] = ivt_chosen_bool
        self.trial['ivt_chosen_bool'].data = trial_ivt_chosen_bool == 1


    def _generate_whether_mt_chosen_binomial(self):
        """
        eh?
        """

        # Find how many patients could receive thrombectomy and
        # where they are in the patient array:
        inds_eligible_for_mt = np.where(
            self.trial['mt_mask5_mask4_and_enough_time_to_treat'].data == 1)[0]
        n_eligible_for_mt = len(inds_eligible_for_mt)
        # Use the known proportion chosen to find the number of
        # patients who will receive thrombectomy:
        n_mt = int(n_eligible_for_mt *
                   self.real_data_dict['proportion_chosen_for_mt'])

        # Use the proportion of patients who receive thrombolysis and
        # thrombectomy to create two groups now.
        # Number of patients receiving both:
        n_mt_and_ivt = int(
            n_mt * self.real_data_dict['proportion_of_mt_also_receiving_ivt'])
        # Number of patients receiving thrombectomy only:
        n_mt_not_ivt = n_mt - n_mt_and_ivt

        # Find which patients in the array may be used for each group:
        inds_eligible_for_mt_and_ivt = np.where(
            (self.trial['mt_mask5_mask4_and_enough_time_to_treat'].data == 1) &
            (self.trial['ivt_chosen_bool'].data == 1)
            )[0]
        inds_eligible_for_mt_not_ivt = np.where(
            (self.trial['mt_mask5_mask4_and_enough_time_to_treat'].data == 1) &
            (self.trial['ivt_chosen_bool'].data == 0)
            )[0]

        # Randomly select patients from these subgroups to be given
        # thrombectomy.
        inds_mt_and_ivt = np.random.choice(
            inds_eligible_for_mt_and_ivt,
            size=n_mt_and_ivt,
            replace=False
            )
        inds_mt_not_ivt = np.random.choice(
            inds_eligible_for_mt_not_ivt,
            size=n_mt_not_ivt,
            replace=False
            )

        # Initially create the array with nobody receiving treatment...
        trial_mt_chosen_bool = np.zeros(self.patients_per_run, dtype=int)
        # ... then update the patients that we've just picked out.
        trial_mt_chosen_bool[inds_mt_and_ivt] = 1
        trial_mt_chosen_bool[inds_mt_not_ivt] = 1
        # Store in self:
        self.trial['mt_chosen_bool'].data = trial_mt_chosen_bool == 1


    # ##############################
    # ##### COMBINE CONDITIONS #####
    # ##############################
    def _calculate_onset_to_scan_time(self):
        """
        Recalculate existing times to find time since onset.

        Store the calculated times and boolean arrays of
        whether the times are below given limits.
        """

        # Onset to scan:
        self.trial['onset_to_scan_mins'].data = np.sum([
            self.trial['onset_to_arrival_mins'].data,
            self.trial['arrival_to_scan_mins'].data,
            ], axis=0)

        self.trial['onset_to_scan_within_limit_ivt_bool'].data = \
            (self.trial['onset_to_scan_mins'].data <= self.limit_ivt_mins)
        self.trial['onset_to_scan_within_limit_mt_bool'].data = \
            (self.trial['onset_to_scan_mins'].data <= self.limit_mt_mins)

        self._create_masks_onset_to_scan_on_time()

    def _calculate_time_left_for_ivt_after_scan(
            self,
            minutes_left=15
            ):
        """Minutes left to thrombolyse after scan"""
        self.trial['time_left_for_ivt_after_scan_mins'].data = np.maximum((
            self.allowed_onset_to_needle_time_mins -
            (self.trial['onset_to_arrival_mins'].data +
             self.trial['arrival_to_scan_mins'].data)
            ), -0.0)
        # Onset time known,
        # scan in 4 hours and
        # 15 min time left to thrombolyse
        # (1 to proceed, 0 not to proceed)
        self.trial['enough_time_for_ivt_bool'].data = (
            self.trial['ivt_mask4_mask3_and_onset_to_scan_on_time'].data *
            (self.trial['time_left_for_ivt_after_scan_mins'].data
                >= minutes_left)
            )


    def _calculate_and_clip_onset_to_needle_time(
            self, clip_limit_mins=-100.0):
        """Onset to needle"""
        onset_to_needle_mins = (
            self.trial['onset_to_arrival_mins'].data +
            self.trial['arrival_to_scan_mins'].data +
            self.trial['scan_to_needle_mins'].data
            )
        if clip_limit_mins > 0.0:
            # Clip to 4.5 hrs + given allowance max
            onset_to_needle_mins = np.clip(
                onset_to_needle_mins,  # Values to clip
                0,                     # Minimum allowed value
                clip_limit_mins        # Maximum allowed value
                )
        self.trial['onset_to_needle_mins'].data = onset_to_needle_mins


    def _calculate_ivt_rate(self):
        """Calculate thrombolysis rate across the whole cohort."""
        self.trial['ivt_rate_percent'].data = \
            self.trial['ivt_chosen_bool'].data.mean() * 100

    # ########################
    # ##### Thrombectomy #####
    # ########################
    def _calculate_time_left_for_mt_after_scan(
            self,
            minutes_left=15
            ):
        """Minutes left to thrombolyse after scan"""
        self.trial['time_left_for_mt_after_scan_mins'].data = np.maximum((
            self.allowed_onset_to_puncture_time_mins -
            (self.trial['onset_to_arrival_mins'].data +
             self.trial['arrival_to_scan_mins'].data)
            ), -0.0)

        # Onset time known,
        # scan in 4 hours and
        # 15 min time left to thrombectomise
        # (1 to proceed, 0 not to proceed)
        self.trial['enough_time_for_mt_bool'].data = (
            self.trial['mt_mask4_mask3_and_onset_to_scan_on_time'].data *
            (self.trial['time_left_for_mt_after_scan_mins'].data
                >= minutes_left)
            )

    def _calculate_and_clip_onset_to_puncture_time(
            self, clip_limit_mins=-100.0):
        """Onset to puncture"""
        onset_to_puncture_mins = (
            self.trial['onset_to_arrival_mins'].data +
            self.trial['arrival_to_scan_mins'].data +
            self.trial['scan_to_puncture_mins'].data
            )
        if clip_limit_mins > 0.0:
            # Clip to HOWEVER MANY hrs + given allowance max
            onset_to_puncture_mins = np.clip(
                onset_to_puncture_mins,  # Values to clip
                0,                       # Minimum allowed value
                clip_limit_mins          # Maximum allowed value
                )
        self.trial['onset_to_puncture_mins'].data = onset_to_puncture_mins

    def _calculate_mt_rate(self):
        """Calculate thrombectomy rate across the whole cohort."""
        self.trial['mt_rate_percent'].data = \
            self.trial['mt_chosen_bool'].data.mean() * 100




    def _assign_stroke_type_code(self):
        """
        Assign stroke type based partly on treatment decision.

        --- Requirements ---
        The patient cohort can be split into the following four groups:

        ▓A▓ - patients receiving thrombectomy only.
        ░B░ - patients receiving thrombolysis and thrombectomy.
        ▒C▒ - patients receiving thrombolysis only.
        █D█ - patients receiving neither thrombolysis nor thrombectomy.

        For example, the groups might have these proportions:
         A   B       C                             D
        ▓▓▓░░░░░▒▒▒▒▒▒▒▒▒▒▒████████████████████████████████████████████

        The rules:
        + Groups ▓A▓ and ░B░ must contain only LVOs.
        + Group ▒C▒ must contain only nLVOs or LVOs.
        + Group █D█ may contain any stroke type.

        --- Method ---
        1. Set everyone in Groups ▓A▓ and ░B░ to have LVOs.
        2. Decide Groups ░B░ and ▒C▒ combined should contain LVO and
           nLVO patients in the same proportion as the whole cohort.
           For example:   B       C          Some LVO patients
                        ░░░░░▒▒▒▒▒▒▒▒▒▒▒     have already been placed
                        <-LVO-><--nLVO->     into Group ░B░.
           To find how many LVO patients go into Group ▒C▒: ########################################## update this method.
           35% (e.g.) = (
               (LVO patients in Group ░B░ +
               LVO patients in Group ▒C▒) /
               (All patients in Groups ░B░ and ▒C▒)
               )
           -> LVO patients in Group ▒C▒ = (
               35% (e.g.) * (All patients in Groups ░B░ and ▒C▒) -
               LVO patients in Group ░B░
               )
           The rest of the patients in Group ▒C▒ are assigned as nLVO.
        3. Randomly pick patients in Group █D█ to be each stroke type
           so that the numbers add up as expected.

        --- Result ---
        Creation of self.trial['stroke_type_code'] array.
        
        """
        # Initially set all patients to "other stroke type":
        trial_stroke_type_code = np.full(self.patients_per_run, 0)
        # Keep track of which patients we've assigned a value to:
        trial_type_assigned_bool = np.zeros(self.patients_per_run, dtype=int)

        # Target numbers of patients with each stroke type:
        total = dict()
        total['lvo'] = int(round(self.patients_per_run * self.proportion_lvo, 0))
        total['nlvo'] = int(round(self.patients_per_run * self.proportion_nlvo, 0))
        total['else'] = self.patients_per_run - (total['lvo'] + total['nlvo'])

        # Available combinations:
        # | Type | Code | Thrombolysis | Thrombectomy |
        # +------+------+--------------+--------------|
        # | nLVO | 1    | Yes or no    | No           |
        # | LVO  | 2    | Yes or no    | Yes or no    |
        # | Else | 0    | No           | No           |
        n_lvo_left_to_assign = total['lvo']
        n_nlvo_left_to_assign = total['nlvo']
        n_else_left_to_assign = total['else']

        # ### STEP 1 ###
        # Firstly set everyone chosen for thrombectomy to LVO.
        # (Groups A and B)
        inds_chosen_for_mt = np.where(
            self.trial['mt_chosen_bool'].data == 1)[0]
        trial_stroke_type_code[inds_chosen_for_mt] = 2
        # Bookkeeping:
        groupAB = dict()
        groupAB['all'] = len(inds_chosen_for_mt)
        groupAB['lvo'] = groupAB['all']
        groupAB['nlvo'] = 0
        groupAB['else'] = 0

        n_lvo_left_to_assign -= groupAB['lvo']
        trial_type_assigned_bool[inds_chosen_for_mt] += 1

        # ### STEP 2 ###
        # Set everyone chosen for thrombolysis to either nLVO or LVO
        # in the same ratio as the patient proportions if possible.
        # (Groups B and C)
        inds_chosen_for_ivt = np.where(
            self.trial['ivt_chosen_bool'].data == 1)[0]
        # Group B:
        inds_chosen_for_ivt_and_mt = np.where(
            (self.trial['ivt_chosen_bool'].data == 1) &
            (self.trial['mt_chosen_bool'].data == 1))[0]
        # Group C:
        inds_chosen_for_ivt_and_not_mt = np.where(
            (self.trial['ivt_chosen_bool'].data == 1) &
            (self.trial['mt_chosen_bool'].data == 0))[0]

        groupBC = dict()
        groupBC['all'] = len(inds_chosen_for_ivt)
        # groupBC['lvo'] = int(round(groupBC['all'] * self.proportion_lvo, 0))
        # groupBC['nlvo'] = groupBC['all'] - groupBC['lvo']
        groupBC['nlvo'] = np.minimum(
            int(round(groupBC['all'] * (1.0 - self.proportion_lvo), 0)), 
            len(inds_chosen_for_ivt_and_not_mt))
        groupBC['lvo'] = groupBC['all'] - groupBC['nlvo']
        groupBC['else'] = 0

        # So for this subset of patients...
        # But some LVO have already been assigned because of the
        # overlap between patients receiving thrombolysis and
        # thrombectomy.
        # Group A:
        groupA = dict()
        groupA['all'] = int(round(
            groupAB['all'] * 
            (1.0 - self.real_data_dict['proportion_of_mt_also_receiving_ivt']), 0))
        groupA['lvo'] = groupA['all']
        groupA['nlvo'] = 0
        groupA['else'] = 0
        # Group B:
        groupB = dict()
        groupB['all'] = len(inds_chosen_for_ivt_and_mt)
        groupB['lvo'] = groupB['all']
        groupB['nlvo'] = 0
        groupB['else'] = 0
        # Group C:
        groupC = dict()
        groupC['all'] = len(inds_chosen_for_ivt_and_not_mt)
        groupC['nlvo'] = groupBC['nlvo']
        groupC['lvo'] = groupC['all'] - groupC['nlvo']
        groupC['else'] = 0

        if groupBC['lvo'] != np.sum((groupC['lvo'], groupB['lvo'])):
            print('??', groupBC['lvo'], groupB['lvo'], groupC['lvo'])


        # For each of LVO and nLVO, randomly select some indices out
        # of Group C (IVT and not MT). 
        inds_lvo_chosen_for_ivt_and_not_mt = np.random.choice(
            inds_chosen_for_ivt_and_not_mt,
            size=groupC['lvo'],
            replace=False
            )
        inds_nlvo_chosen_for_ivt_and_not_mt = np.array(list(
            set(inds_chosen_for_ivt_and_not_mt) -
            set(inds_lvo_chosen_for_ivt_and_not_mt)
            ))
        trial_type_assigned_bool[inds_chosen_for_ivt_and_not_mt] += 1
        trial_stroke_type_code[inds_lvo_chosen_for_ivt_and_not_mt] = 2
        trial_stroke_type_code[inds_nlvo_chosen_for_ivt_and_not_mt] = 1

        # Bookkeeping:
        n_lvo_left_to_assign -= groupC['lvo']
        n_nlvo_left_to_assign -= groupC['nlvo']

        # ### STEP 3 ###
        # Sanity check that the number of patients we're about to
        # assign matches the number remaining in the array:
        n_left_list = [
            n_lvo_left_to_assign, n_nlvo_left_to_assign, n_else_left_to_assign
            ]
        n_not_yet_assigned = len(
            np.where(trial_type_assigned_bool == 0)[0])
        if np.sum(n_left_list) != n_not_yet_assigned:
            print('Something has gone wrong in assigning stroke types. ')

        # All other patients received neither thrombolysis nor
        # thrombectomy. They may have any stroke type.
        # For each stroke type, randomly select some indices out
        # of those that have not yet been assigned a stroke type:
        # Group D.
        groupD = dict()
        groupD['all'] = n_not_yet_assigned
        inds_no_treatment_lvo = np.random.choice(
            np.where(trial_type_assigned_bool == 0)[0],
            size=n_lvo_left_to_assign,
            replace=False
            )
        trial_type_assigned_bool[inds_no_treatment_lvo] += 1
        trial_stroke_type_code[inds_no_treatment_lvo] = 2
        groupD['lvo'] = n_lvo_left_to_assign

        inds_no_treatment_nlvo = np.random.choice(
            np.where(trial_type_assigned_bool == 0)[0],
            size=n_nlvo_left_to_assign,
            replace=False
            )
        trial_type_assigned_bool[inds_no_treatment_nlvo] += 1
        trial_stroke_type_code[inds_no_treatment_nlvo] = 1
        groupD['nlvo'] = n_nlvo_left_to_assign

        inds_no_treatment_else = np.random.choice(
            np.where(trial_type_assigned_bool == 0)[0],
            size=n_else_left_to_assign,
            replace=False
            )
        trial_type_assigned_bool[inds_no_treatment_else] += 1
        trial_stroke_type_code[inds_no_treatment_else] = 0
        groupD['else'] = n_else_left_to_assign


        # ### Final check ###
        # Sanity check that no patient was assigned multiple stroke types:
        if np.any(trial_type_assigned_bool > 1):
            print(''.join([
                'Warning: check the stroke type code assignment. ',
                'Some patients were assigned multiple stroke types. ']))
        if np.any(trial_type_assigned_bool == 0):
            print(''.join([
                'Warning: check the stroke type code assignment. ',
                'Some patients were not assigned a stroke type. ']))

        # Now save this final array to self:
        self.trial['stroke_type_code'].data = trial_stroke_type_code


    # ##############################
    # ######## CREATE MASKS ########
    # ##############################
    """
    Make masks of patients meeting the conditions.

    Create masks for subgroups of patients as in the
    hospital performance data extraction.

    Key:
    ░ - patients still in the subgroup
    ▒ - patients rejected from the subgroup at this step
    █ - patients rejected from the subgroup in previous steps


    ▏Start: Full group                                                ▕
    ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░
    ▏-------------------------All patients----------------------------▕
    ▏                                                                 ▕
    ▏Mask 1: Is onset time known?                                     ▕
    ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒
    ▏--------------------Yes----------------------▏---------No--------▕
    ▏                                             ▏                   ▕
    ▏Mask 2: Is arrival within x hours?           ▏                   ▕
    ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░▒▒▒▒▒▒▒▒▒▒▒█████████████████████
    ▏---------------Yes----------------▏----No----▏------Rejected-----▕
    ▏                                  ▏          ▏                   ▕
    ▏Mask 3: Is scan within x hours of arrival?   ▏                   ▕
    ░░░░░░░░░░░░░░░░░░░░░░░░░░░░▒▒▒▒▒▒▒████████████████████████████████
    ▏------------Yes------------▏--No--▏-----------Rejected-----------▕
    ▏                           ▏      ▏                              ▕
    ▏Mask 4: Is scan within x hours of onset?                         ▕
    ░░░░░░░░░░░░░░░░░░░░░░░▒▒▒▒▒███████████████████████████████████████
    ▏----------Yes---------▏-No-▏---------------Rejected--------------▕
    ▏                      ▏    ▏                                     ▕
    ▏Mask 5: Is there enough time left for thrombolysis/thrombectomy? ▕
    ░░░░░░░░░░░░░░░░▒▒▒▒▒▒▒████████████████████████████████████████████
    ▏------Yes------▏--No--▏------------------Rejected----------------▕
    ▏               ▏      ▏                                          ▕
    ▏Mask 6: Did the patient receive thrombolysis/thrombectomy?       ▕
    ░░░░░░░░░░░▒▒▒▒▒███████████████████████████████████████████████████
    ▏----Yes---▏-No-▏---------------------Rejected--------------------▕

    """
    def _create_masks_onset_time_known(self):
        """Mask 1: Is onset time known?"""
        # Same mask for thrombolysis and thrombolysis.
        mask = np.copy(self.trial['onset_time_known_bool'].data)

        self.trial['ivt_mask1_onset_known'].data = mask
        self.trial['mt_mask1_onset_known'].data = mask

    def _create_masks_onset_to_arrival_on_time(self):
        """Mask 2: Is arrival within x hours?"""
        mask_ivt = (
            (self.trial['ivt_mask1_onset_known'].data == 1) &
            (self.trial['onset_to_arrival_within_limit_ivt_bool'].data == 1)
            )
        mask_mt = (
            (self.trial['mt_mask1_onset_known'].data == 1) &
            (self.trial['onset_to_arrival_within_limit_mt_bool'].data == 1)
            )

        self.trial['ivt_mask2_mask1_and_onset_to_arrival_on_time'].data = mask_ivt
        self.trial['mt_mask2_mask1_and_onset_to_arrival_on_time'].data = mask_mt

    def _create_masks_arrival_to_scan_on_time(self):
        """Mask 3: Is scan within x hours of arrival?"""
        mask_ivt = (
            (self.trial['ivt_mask2_mask1_and_onset_to_arrival_on_time'].data == 1) &
            (self.trial['arrival_to_scan_within_limit_ivt_bool'].data == 1)
            )
        mask_mt = (
            (self.trial['mt_mask2_mask1_and_onset_to_arrival_on_time'].data == 1) &
            (self.trial['arrival_to_scan_within_limit_mt_bool'].data == 1)
            )

        self.trial['ivt_mask3_mask2_and_arrival_to_scan_on_time'].data = mask_ivt
        self.trial['mt_mask3_mask2_and_arrival_to_scan_on_time'].data = mask_mt

    def _create_masks_onset_to_scan_on_time(self):
        """Step 4: Is scan within x hours of onset?"""
        mask_ivt = (
            (self.trial['ivt_mask3_mask2_and_arrival_to_scan_on_time'].data == 1) &
            (self.trial['onset_to_arrival_within_limit_ivt_bool'].data == 1)
            )
        mask_mt = (
            (self.trial['mt_mask3_mask2_and_arrival_to_scan_on_time'].data == 1) &
            (self.trial['onset_to_arrival_within_limit_mt_bool'].data == 1)
            )

        self.trial['ivt_mask4_mask3_and_onset_to_scan_on_time'].data = mask_ivt
        self.trial['mt_mask4_mask3_and_onset_to_scan_on_time'].data = mask_mt

    def _create_masks_enough_time_to_treat(self):
        """Mask 5: Is there enough time left for threatment?"""
        mask_ivt = (
            (self.trial['ivt_mask4_mask3_and_onset_to_scan_on_time'].data == 1) &
            (self.trial['enough_time_for_ivt_bool'].data == 1)
            )
        mask_mt = (
            (self.trial['mt_mask4_mask3_and_onset_to_scan_on_time'].data == 1) &
            (self.trial['enough_time_for_mt_bool'].data == 1)
            )

        self.trial['ivt_mask5_mask4_and_enough_time_to_treat'].data = mask_ivt
        self.trial['mt_mask5_mask4_and_enough_time_to_treat'].data = mask_mt

    def _create_masks_treatment_given(self):
        """Mask 6: Was treatment given?"""
        mask_ivt = (
            (self.trial['ivt_mask5_mask4_and_enough_time_to_treat'].data == 1) &
            (self.trial['ivt_chosen_bool'].data == 1)
            )
        mask_mt = (
            (self.trial['mt_mask5_mask4_and_enough_time_to_treat'].data == 1) &
            (self.trial['mt_chosen_bool'].data == 1)
            )

        self.trial['ivt_mask6_mask5_and_treated'].data = mask_ivt
        self.trial['mt_mask6_mask5_and_treated'].data = mask_mt


    def _gather_results_in_dataframe(self):
        
        # Combine entries in 
        
        
        # The following construction for "data" will work as long as
        # all arrays in trial have the same length.
        df = pd.DataFrame(
            data=np.array([v.data for v in self.trial.values()], dtype=object).T,
            columns=list(self.trial.keys())
            )
        self.results_dataframe = df
        


    # #########################
    # ##### SANITY CHECKS #####
    # #########################
    def _run_sanity_check_on_hospital_data(self, hospital_data):
        """
        Sanity check the hospital data.

        Check all of the relevant keys exist and that the data is
        of the right dtype.
        """
        keys = [
            'admissions',
            'onset_known',
            'known_arrival_within_4hrs',
            'scan_within_4_hrs',
            'proportion_chosen_for_thrombolysis',
            'proportion_chosen_for_thrombectomy',
            'onset_arrival_mins_mu',
            'onset_arrival_mins_sigma',
            'arrival_scan_arrival_mins_mu',
            'arrival_scan_arrival_mins_sigma',
            'scan_needle_mins_mu',
            'scan_needle_mins_sigma',
            'scan_puncture_mins_mu',
            'scan_puncture_mins_sigma',
            'onset_scan_4_hrs',
            'scan_needle_4_hrs',
            'known_arrival_within_8hrs',
            'scan_within_8_hrs',
            'onset_scan_8_hrs',
            'onset_arrival_thrombectomy_mins_mu',
            'onset_arrival_thrombectomy_mins_sigma',
            'arrival_scan_arrival_thrombectomy_mins_mu',
            'arrival_scan_arrival_thrombectomy_mins_sigma',
            ]
        expected_dtypes = [['float']] * len(keys)

        success = True
        for k, key in enumerate(keys):
            try:
                value_here = hospital_data[key]
                if type(value_here) in [np.ndarray]:
                    dtype_here = value_here.dtype
                    expected_dtypes_here = [
                        np.dtype(d) for d in expected_dtypes[k]
                        ]
                    if dtype_here not in expected_dtypes_here:
                        print(''.join([
                            f'{key} is type {dtype_here} instead of ',
                            f'expected type {expected_dtypes_here}.'
                            ]))
                        success = False
            except KeyError:
                print(f'{key} is missing from the hospital data.')
                success = False
        if success is False:
            error_str = 'The input hospital data needs fixing.'
            raise ValueError(error_str) from None





    def _fudge_patients_after_time_limit_mt(
            self,
            onset_to_arrival_mins,
            proportion_within_8hr,
            number_of_patients
            ):
        """
        Make sure the proportion of patients with arrival time
        below X hours matches the hospital performance data.
        Set a few patients now to have arrival times above
        X hours.
        """
        inds_patients_arriving_after_X_hours = np.where(
            onset_to_arrival_mins > self.limit_mt_mins)
        n_patients_arriving_after_X_hours = len(
            inds_patients_arriving_after_X_hours[0])

        expected_patients_arriving_after_X_hours = (
            number_of_patients -
            int(proportion_within_8hr *
                number_of_patients
                )
            )

        n_patients_to_fudge = (
            expected_patients_arriving_after_X_hours -
            n_patients_arriving_after_X_hours
            )

        if n_patients_to_fudge > 0:
            inds_patients_to_fudge = np.random.randint(
                0,                         # minimum
                number_of_patients,        # maximum
                size=n_patients_to_fudge,
                dtype=int
            )
            # Indices that are already above the 8 hour limit:
            inds_already_above_8hr = np.intersect1d(
                inds_patients_arriving_after_X_hours, inds_patients_to_fudge)
            if len(inds_already_above_8hr) > 0:
                # Change repeat indices to non-repeated values.
                for ind in inds_already_above_8hr:
                    new_ind = ind
                    breaker = 0  # For stopping infinite loops
                    while new_ind not in inds_patients_to_fudge:
                        new_ind += 1
                        if new_ind >= number_of_patients:
                            new_ind = 0
                            breaker += 1
                            if breaker > 1:
                                raise RecursionError(''.join([
                                    'Something has gone wrong with ',
                                    'reassigning onset-to-arrival times.'
                                    ])) from None
                    # Add the new index to the fudge list and
                    # remove the duplicated index.
                    inds_patients_to_fudge = np.append(
                        inds_patients_to_fudge, new_ind)
                    inds_patients_to_fudge = np.delete(
                        inds_patients_to_fudge,
                        np.where(inds_patients_to_fudge == ind)
                        )
            # Add eight hours to the affected patients.
            onset_to_arrival_mins[inds_patients_to_fudge] += \
                self.limit_mt_mins
        return onset_to_arrival_mins



    def _sanity_check_generated_times_patient_proportions(self):
        """
        eh?
        """
        for i, treatment in enumerate(['ivt', 'mt']):
            # Create patient proportions from generated (g) data:

            # Known arrival within x hours:
            # Yes to Step 2 / Yes to Step 1
            g_proportion_known_arrival_on_time = (
                np.sum(self.trial[treatment + '_mask2_mask1_and_onset_to_arrival_on_time'].data) /
                np.sum(self.trial[treatment + '_mask1_onset_known'].data)
                if np.sum(self.trial[treatment + '_mask1_onset_known'].data) > 0
                else np.NaN)

            # Scan within x hours of arrival:
            # Yes to Step 3 / Yes to Step 2
            g_proportion_arrival_to_scan_on_time = (
                np.sum(self.trial[treatment + '_mask3_mask2_and_arrival_to_scan_on_time'].data) /
                np.sum(self.trial[treatment + '_mask2_mask1_and_onset_to_arrival_on_time'].data)
                if np.sum(self.trial[treatment + '_mask2_mask1_and_onset_to_arrival_on_time'].data) > 0
                else np.NaN)

            # Scan within x hours of onset:
            # Yes to Step 4 / Yes to Step 3
            g_proportion_onset_to_scan_on_time = (
                np.sum(self.trial[treatment + '_mask4_mask3_and_onset_to_scan_on_time'].data) /
                np.sum(self.trial[treatment + '_mask3_mask2_and_arrival_to_scan_on_time'].data)
                if np.sum(self.trial[treatment + '_mask3_mask2_and_arrival_to_scan_on_time'].data) > 0
                else np.NaN)

            # Compare with patient proportions from real hospital data:
            self._check_proportion(
                g_proportion_known_arrival_on_time,
                self.real_data_dict['proportion_known_arrival_on_time_' + treatment],
                label=f'known arrival on time for {treatment}',
                leeway=0.25
                )
            self._check_proportion(
                g_proportion_arrival_to_scan_on_time,
                self.real_data_dict['proportion_arrival_to_scan_on_time_' + treatment],
                label=f'arrival to scan on time for {treatment}',
                leeway=0.25
                )
            self._check_proportion(
                g_proportion_onset_to_scan_on_time,
                self.real_data_dict['proportion_onset_to_scan_on_time_' + treatment],
                label=f'onset to scan on time for {treatment}',
                leeway=0.25
                )


    def _check_proportion(
            self,
            prop_current,
            prop_target,
            label='',
            leeway=0.1
            ):
        """
        If the proportion is more than (leeway*100)% off the target,
        raise a warning message.
        """
        if ((prop_current > prop_target + leeway) or
            (prop_current < prop_target - leeway)):
            print(''.join([
                f'The proportion of "{label}" is ',
                f'over {leeway*100}% out from the target value. '
                f'Target: {prop_target}, ',
                f'current: {prop_current}.'
                ]))
        else:
            pass


    def _check_distribution_statistics(
            self,
            patient_times,
            mu_target,
            sigma_target,
            label=''
            ):
        """
        Raise warning if:
        - the new mu is outside the old mu +/- old sigma, or
        - the new sigma is considerably larger than old sigma.
        """
        # Set all zero or negative values to something tiny here
        # to prevent RuntimeWarning about division by zero
        # encountered in log.
        patient_times = np.clip(patient_times, a_min=1e-7, a_max=None)
        
        # Actual statistics.
        mu_actual = np.mean(np.log(patient_times))
        sigma_actual = np.std(np.log(patient_times))

        # Check new mu:
        if abs(mu_target - mu_actual) > sigma_target:
            print(''.join([
                f'Warning: the log-normal "{label}" distribution ',
                'has a mean outside the target mean plus or minus ',
                'one standard deviation.'
            ]))
        else:
            pass

        # Check new sigma:
        if sigma_target > 5*sigma_actual:
            print(''.join([
                f'Warning: the log-normal "{label}" distribution ',
                'has a standard deviation at least five times as large ',
                'as the target standard deviation.'
            ]))
        else:
            pass
