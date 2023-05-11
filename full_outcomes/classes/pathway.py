import numpy as np


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

    """

    def __init__(self, hospital_name, hospital_data):  # , calibration=1.0):
        """
        USeful
        """
        # Set up allowed time for thrombolysis (for under 80 and 80+)
        self.allowed_onset_to_needle_times = (270, 270)
        # Add allowed over-run
        self.allowed_overrun_for_slow_scan_to_needle = 15

        # Anything specific to this instance of the class
        # should go in here.
        try:
            self.hospital_name = str(hospital_name)
        except:
            print('Hospital name should be a string. Name not set.')
            self.hospital_name = ''
        # self.calibration = calibration

        
        # INCLUDE SANITY CHECKS FOR THIS STUFF
        # won't the class end up huge? Is there a better way?
        # Just check the dtype of hospital_data, check the size, check for missing values?
        # This is the only place in this class that we need sanity checks - 
        # everything else uses self. or is generated from previously-sanity-checked things.
        
        # From hospital data:
        self.patients_per_run = \
            int(hospital_data['admissions'])

        # Patient population:
        self.proportion_80plus = \
            hospital_data['80_plus']
        self.proportion_onset_known = \
            hospital_data['onset_known']
        self.proportion_known_arrival_within_4hr = \
            hospital_data['known_arrival_within_4hrs']
        self.proportion_scan_within_4hr = \
            hospital_data['scan_within_4_hrs']
        self.proportion_chosen_for_thrombolysis = \
            hospital_data['eligable']

        # For log-normal number generation:
        self.lognorm_mu_onset_arrival_mins = \
            hospital_data['onset_arrival_mins_mu']
        self.lognorm_sigma_onset_arrival_mins = \
            hospital_data['onset_arrival_mins_sigma']
        self.lognorm_mu_arrival_scan_arrival_mins = \
            hospital_data['arrival_scan_arrival_mins_mu']
        self.lognorm_sigma_arrival_scan_arrival_mins = \
            hospital_data['arrival_scan_arrival_mins_sigma']
        self.lognorm_mu_scan_needle_mins = \
            hospital_data['scan_needle_mins_mu']
        self.lognorm_sigma_scan_needle_mins = \
            hospital_data['scan_needle_mins_sigma']

    #
    def run_trial(self, patients_per_run=0):
        """
        The big loopy.

        Creates the following attributes:
        - patient_array_80plus_bool
        - patient_array_onset_to_needle_limit_mins
        - patient_array_onset_time_known_bool
        - patient_array_onset_to_arrival_under_4hr_bool
        - patient_array_onset_to_arrival_mins
        - patient_array_arrival_to_scan_below_4hr_bool
        - patient_array_arrival_to_scan_mins
        - patient_array_time_left_to_thrombolyse_after_scan_mins
        - patient_array_enough_time_for_thrombolysis_bool
        - patient_array_ischaemic_bool
        - patient_array_thrombolysis_chosen_bool
        - patient_array_thrombolysis_conditions_met_bool
        - patient_array_scan_to_needle_mins
        - patient_array_onset_to_needle_mins
        - patient_array_thrombolysis_rate_percent
        """
        if patients_per_run > 0:
            # Overwrite the default value from the hospital data.
            self.patients_per_run = patients_per_run
        else:
            # Don't update anything.
            pass

        # Create a new attribute for each new "column" of data.
        # Store the new info as many 1D arrays instead of the
        # usual single 2D grid.
        self._generate_80plus_binomial()
        self._allocate_limit_onset_to_needle_based_on_80plus()

        self._generate_onset_time_known_binomial()
        self._generate_onset_to_arrival_under_4hr_binomial()
        self._combine_conditions_onset_known_and_arrival_under_4hr()
        self._generate_onset_to_arrival_time_lognorm()

        self._generate_arrival_to_scan_below_4hr_binomial()
        self._generate_arrival_to_scan_time_lognorm()
        self._calculate_time_left_to_thrombolyse_after_scan()
        self._combine_conditions_enough_time_for_thrombolysis()
        self._generate_whether_ischaemic_binomial()
        self._generate_whether_thrombolysis_chosen_binomial()
        self._combine_conditions_for_thrombolysis()
        self._generate_scan_to_needle_time_lognorm()
        self._calculate_and_clip_onset_to_needle_time(
            clip_limit_mins=(
                270 + self.allowed_overrun_for_slow_scan_to_needle)
            )

        self._calculate_thrombolysis_rate()

    #
    # Methods for the pathway:
    def _generate_80plus_binomial(self):
        """h"""
        self.patient_array_80plus_bool = \
            np.random.binomial(
                1,                       # Number of trials
                self.proportion_80plus,  # Probability of success
                self.patients_per_run    # Number of samples drawn
                )

    def _allocate_limit_onset_to_needle_based_on_80plus(self):
        """
        Assign allowable onset to needle (for under 80 and 80+).

        Prerequisites:
        - patient_array_80plus_bool
        """
        # New empty list:
        limit_onset_to_needle = np.full(
            self.patient_array_80plus_bool.shape, 0.0)
        # Assign time for under 80s:
        limit_onset_to_needle[self.patient_array_80plus_bool == 0] = \
            self.allowed_onset_to_needle_times[0]
        # Assign time for over 80s:
        limit_onset_to_needle[self.patient_array_80plus_bool == 1] = \
            self.allowed_onset_to_needle_times[1]

        self.patient_array_onset_to_needle_limit_mins = \
            limit_onset_to_needle

    def _generate_onset_time_known_binomial(self):
        """Assign onset time known"""
        self.patient_array_onset_time_known_bool = \
            np.random.binomial(
                1,                            # Number of trials
                self.proportion_onset_known,  # Probability of success
                self.patients_per_run         # Number of samples drawn
                ) == 1

    def _generate_onset_to_arrival_under_4hr_binomial(self):
        """Assign onset to arrival is less than 4 hours."""
        self.patient_array_onset_to_arrival_under_4hr_bool = \
            np.random.binomial(
                1,                      # Number of trials
                self.proportion_known_arrival_within_4hr,
                                        # ^ Probability of success
                self.patients_per_run   # Number of samples drawn
                )

    def _combine_conditions_onset_known_and_arrival_under_4hr(self):
        """Onset known and is within 4 hours."""
        # If both conditions are met (values = 1), the product = 1.
        # Otherwise the product is 0.
        self.patient_array_onset_known_and_arrival_under_4hr_bool = (
            self.patient_array_onset_time_known_bool *
            self.patient_array_onset_to_arrival_under_4hr_bool
            )

    def _generate_onset_to_arrival_time_lognorm(self):
        """Assign onset to arrival time (natural log normal distribution)"""
        self.patient_array_onset_to_arrival_mins = \
            np.random.lognormal(
                self.lognorm_mu_onset_arrival_mins,     # Mean
                self.lognorm_sigma_onset_arrival_mins,  # Standard dev.
                self.patients_per_run                   # n of samples
                )

    def _generate_arrival_to_scan_below_4hr_binomial(self):
        """Assign arrival to scan is less than 4 hours"""
        self.patient_array_arrival_to_scan_below_4hr_bool = \
            np.random.binomial(
                1,                                # Number of trials
                self.proportion_scan_within_4hr,  # Success probability
                self.patients_per_run             # Number of samples
                )

    def _generate_arrival_to_scan_time_lognorm(self):
        """Assign arrival to scan time (natural log normal distribution)"""
        self.patient_array_arrival_to_scan_mins = \
            np.random.lognormal(
                self.lognorm_mu_arrival_scan_arrival_mins,     # Mean
                self.lognorm_sigma_arrival_scan_arrival_mins,  # Std
                self.patients_per_run                          # Number
                )

    def _calculate_time_left_to_thrombolyse_after_scan(self):
        """Minutes left to thrombolyse after scan"""
        self.patient_array_time_left_to_thrombolyse_after_scan_mins = (
            self.patient_array_onset_to_needle_limit_mins -
            (self.patient_array_onset_to_arrival_mins +
             self.patient_array_arrival_to_scan_mins)
            )

    def _combine_conditions_enough_time_for_thrombolysis(
            self, minutes_left=15
            ):
        """Check conditions"""
        # Onset time known,
        # scan in 4 hours and
        # 15 min time left to thrombolyse
        # (1 to proceed, 0 not to proceed)
        self.patient_array_enough_time_for_thrombolysis_bool = (
            self.patient_array_onset_known_and_arrival_under_4hr_bool *
            self.patient_array_arrival_to_scan_below_4hr_bool *
            (self.patient_array_time_left_to_thrombolyse_after_scan_mins
                >= minutes_left)
            )

    def _generate_whether_ischaemic_binomial(
            self, proportion_ischaemic=1):
        """Ischaemic_stroke"""
        self.patient_array_ischaemic_bool = \
            np.random.binomial(
                1,                     # Number of trials
                proportion_ischaemic,  # Probability of success
                self.patients_per_run  # Number of samples drawn
                )

    def _generate_whether_thrombolysis_chosen_binomial(self):
        """h"""
        # Eligable for thrombolysis (proportion of ischaemic patients
        # eligable for thrombolysis when scanned within 4 hrs )
        self.patient_array_thrombolysis_chosen_bool = \
            np.random.binomial(
                1,                     # Number of trials
                self.proportion_chosen_for_thrombolysis,
                                       # ^ Probability of success
                self.patients_per_run  # Number of samples drawn
                )

    def _combine_conditions_for_thrombolysis(self):
        """h"""
        # Thrombolysis planned (checks this is within thrombolysys time, &
        # patient considerd eligable for thrombolysis if scanned in time
        self.patient_array_thrombolysis_conditions_met_bool = (
            self.patient_array_enough_time_for_thrombolysis_bool *
            self.patient_array_ischaemic_bool *
            self.patient_array_thrombolysis_chosen_bool
            )

    def _generate_scan_to_needle_time_lognorm(self):
        """h"""
        # scan to needle
        self.patient_array_scan_to_needle_mins = \
            np.random.lognormal(
                self.lognorm_mu_scan_needle_mins,     # Mean
                self.lognorm_sigma_scan_needle_mins,  # Standard dev.
                self.patients_per_run                 # Num. of samples
                )

    def _calculate_and_clip_onset_to_needle_time(
            self, clip_limit_mins=-100.0):
        """Onset to needle"""
        onset_to_needle_mins = (
            self.patient_array_onset_to_arrival_mins +
            self.patient_array_arrival_to_scan_mins +
            self.patient_array_scan_to_needle_mins
            )
        if clip_limit_mins > 0.0:
            # Clip to 4.5 hrs + given allowance max
            onset_to_needle_mins = np.clip(
                onset_to_needle_mins,  # Values to clip
                0,                     # Minimum allowed value
                clip_limit_mins        # Maximum allowed value
                )
        self.patient_array_onset_to_needle_mins = onset_to_needle_mins

    def _calculate_thrombolysis_rate(self):
        # Calculate overall thrombolysis rate
        self.patient_array_thrombolysis_rate_percent = \
            self.patient_array_thrombolysis_conditions_met_bool.mean() * 100
