import numpy as np
import pandas as pd

from classes.patient_array import Patient_array

class SSNAP_Pathway:
    """
    Model of stroke pathway.

    This model simulates the passage through the emergency stroke 
    pathway for a cohort of patients. Each patient spends different
    amounts of time in each step of the pathway and may or may not
    meet various conditions for treatment. The calculations
    for all patients are performed simultaneously.
    
    ----- Method summary -----
    Patient times through the pathway are sampled from distributions 
    passed to the model using NumPy. Then any Yes or No choice is 
    guided by target hospital performance data, so for example if the
    target proportion of known onset times is 40%, we randomly pick 
    40% of these patients to have known onset times.
    
    The goal is to find out whether each patient passed through the
    pathway on time for treatment and then whether they are selected
    for treatment. There are separate time limits for thrombolysis
    and thrombectomy.
    
    The resulting arrays are then sanity checked against more 
    proportions from the target hospital performance data.
    A series of masks are created here with conditions matching those
    used to extract the hospital performance data, so that these masks
    can be used to calculate the equivalent metric for comparison of 
    the generated and target data.
    
    ----- Results -----
    The main results are, for each patient:
    + Arrival time
      + Is onset time known?                               (True/False)
      + Onset to arrival time                                 (minutes)
      + Is time below the thrombolysis limit?              (True/False)
      + Is time below the thrombectomy limit?              (True/False)
    + Scan time
      + Arrival to scan time                                  (minutes)
      + Is time below the thrombolysis limit?              (True/False)
      + Is time below the thrombectomy limit?              (True/False)
      + Onset to scan time                                    (minutes)
      + Is time below the thrombolysis limit?              (True/False)
      + Is time below the thrombectomy limit?              (True/False)
    + Thrombolysis decision
      + How much time is left for thrombolysis after scan?    (minutes)
      + Is there enough time left for thrombolysis?        (True/False)
      + Scan to needle time                                   (minutes)
      + Is time below the thrombolysis limit?              (True/False)
      + Is time below the thrombectomy limit?              (True/False)
      + Onset to needle time                                  (minutes)
      + Is thrombolysis given?                             (True/False)
    + Thrombolysis masks
      1. Onset time is known.                              (True/False)
      2. Mask 1 and onset to arrival time below limit.     (True/False)
      3. Mask 2 and arrival to scan time below limit.      (True/False)
      4. Mask 3 and onset to scan time below limit.        (True/False)
      5. Mask 4 and enough time left for thrombolysis.     (True/False)
      6. Mask 5 and the patient received thrombolysis.     (True/False)
    + Thrombectomy decision
      + How much time is left for thrombectomy after scan?    (minutes)
      + Is there enough time left for thrombectomy?        (True/False)
      + Scan to puncture time                                 (minutes)
      + Is time below the thrombolysis limit?              (True/False)
      + Is time below the thrombectomy limit?              (True/False)
      + Onset to puncture time                                (minutes)
      + Is thrombectomy given?                             (True/False)
    + Thrombectomy masks
      1. Onset time is known.                              (True/False)
      2. Mask 1 and onset to arrival time below limit.     (True/False)
      3. Mask 2 and arrival to scan time below limit.      (True/False)
      4. Mask 3 and onset to scan time below limit.        (True/False)
      5. Mask 4 and enough time left for thrombectomy.     (True/False)
      6. Mask 5 and the patient received thrombectomy.     (True/False)
    + Stroke type code                         (0=Other, 1=nLVO, 2=LVO)
    
    ----- Code layout -----
    Each of the above results is stored as a numpy array containing one
    value for each patient. Any Patient X appears at the same position 
    in all arrays, so the arrays can be seen as columns of a 2D table 
    of patient data. The main run_trial() function outputs all of the
    arrays as a single pandas DataFrame that can be saved to csv. The
    individual arrays are stored in the self.trial dictionary 
    attribute.
    
    These acronyms are used to prevent enormous variable names:
    + IVT = intra-veneous thrombolysis
    + MT = mechanical thrombectomy
    + LVO = large-vessel occlusion
    + nLVO = non-large-vessel occlusion
    
    "Needle" refers to thrombolysis and "puncture" to thrombectomy.
    "On time" means within the time limit, e.g. 4hr for thrombolysis.
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
            target_data_dict: pd.DataFrame
            ):
        """
        Sets up the data required to calculate the patient pathways. # ####################### rewrite this

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
        
        Access the data in the trial attribute with this syntax:
          patient_array.trial['onset_to_arrival_mins'].data
        """

        try:
            self.hospital_name = str(hospital_name)
        except TypeError:
            print('Hospital name should be a string. Name not set.')
            self.hospital_name = ''

        # Run sanity checks.
        # If the data has problems, these raise an exception.
        self._run_sanity_check_on_hospital_data(target_data_dict)

        # From hospital data:
        self.patients_per_run = int(target_data_dict['admissions'])

        # add comments here to explain that this is setting up just the sanity check bits of the thing and not actually creating the data at this stage mmkay
        # The outputs will go in this dictionary:
        # Patient_array(
        #    number_of_patients, valid_dtypes_list, valid_min, valid_max)
        n = self.patients_per_run  # Defined to shorten the following.
        self.trial = dict(
            #
            # Initial steps
            onset_to_arrival_mins = Patient_array(n, ['float'], 0.0, np.inf),
            onset_to_arrival_on_time_ivt_bool = (
                Patient_array(n, ['int', 'bool'], 0, 1)),
            onset_to_arrival_on_time_mt_bool = (
                Patient_array(n, ['int', 'bool'], 0, 1)),
            onset_time_known_bool = Patient_array(n, ['int', 'bool'], 0, 1),
            arrival_to_scan_mins = Patient_array(n, ['float'], 0.0, np.inf),
            arrival_to_scan_on_time_ivt_bool = (
                Patient_array(n, ['int', 'bool'], 0, 1)),
            arrival_to_scan_on_time_mt_bool = (
                Patient_array(n, ['int', 'bool'], 0, 1)),
            onset_to_scan_mins = Patient_array(n, ['float'], 0.0, np.inf),
            #
            # IVT (thrombolysis)
            onset_to_scan_on_time_ivt_bool = (
                Patient_array(n, ['int', 'bool'], 0, 1)),
            time_left_for_ivt_after_scan_mins = (
                Patient_array(n, ['float'], 0.0, np.inf)),
            enough_time_for_ivt_bool = (
                Patient_array(n, ['int', 'bool'], 0, 1)),
            scan_to_needle_mins = (
                Patient_array(n, ['float'], 0.0, np.inf)),
            scan_to_needle_on_time_ivt_bool = (
                Patient_array(n, ['int', 'bool'], 0, 1)),
            scan_to_needle_on_time_mt_bool = (
                Patient_array(n, ['int', 'bool'], 0, 1)), #################### ??
            onset_to_needle_mins = (
                Patient_array(n, ['float'], 0.0, np.inf)),
            onset_to_needle_on_time_bool = (
                Patient_array(n, ['int', 'bool'], 0, 1)),
            ivt_chosen_bool = (
                Patient_array(n, ['int', 'bool'], 0, 1)),
            #
            # Masks of IVT pathway to match input hospital performance
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
            #
            # MT (thrombectomy)
            onset_to_scan_on_time_mt_bool = (
                Patient_array(n, ['int', 'bool'], 0, 1)),
            time_left_for_mt_after_scan_mins = (
                Patient_array(n, ['float'], 0.0, np.inf)),
            enough_time_for_mt_bool = (
                Patient_array(n, ['int', 'bool'], 0, 1)),
            scan_to_puncture_mins = (
                Patient_array(n, ['float'], 0.0, np.inf)),
            scan_to_puncture_on_time_ivt_bool = (
                Patient_array(n, ['int', 'bool'], 0, 1)), ################# ?
            scan_to_puncture_on_time_mt_bool = (
                Patient_array(n, ['int', 'bool'], 0, 1)),
            onset_to_puncture_mins = (
                Patient_array(n, ['float'], 0.0, np.inf)),
            onset_to_puncture_on_time_bool = (
                Patient_array(n, ['int', 'bool'], 0, 1)),
            mt_chosen_bool = (
                Patient_array(n, ['int', 'bool'], 0, 1)),
            #
            # Masks of MT pathway to match input hospital performance
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
            #
            # Use the treatment decisions to assign stroke type
            stroke_type_code = (
                Patient_array(n, ['int'], 0, 2)),
        )

    def __str__(self):
        """Prints info when print(Instance) is called."""
        print_str = ''.join([
            f'For hospital {self.hospital_name}, the target data is: '
        ])
        for (key, val) in zip(
                self.target_data_dict.keys(),
                self.target_data_dict.values()
                ):
            print_str += '\n'
            print_str += f'  {key:50s} '
            print_str += f'{repr(val)}'
        
        print_str += '\n\n'
        print_str += ''.join([
            'The main useful attribute is self.trial, ',
            'a dictionary of the results of the trial.'
            ])
        print_str += ''.join([
            '\n',
            'The easiest way to create the results is:\n',
            '  patient_array.run_trial()\n',    
            'Access the data in the trial dictionary with this syntax:\n'
            '  patient_array.trial[\'onset_to_arrival_mins\'].data'
            ])
        return print_str
    

    def __repr__(self):
        """Prints how to reproduce this instance of the class."""
        return ''.join([
            'SSNAP_Pathway(',
            f'hospital_name={self.hospital_name}, '
            f'hospital_data={self.target_data_dict})'
            ])    
    #
    def run_trial(self, patients_per_run: int=-1):
        """
        Create the pathway details for each patient in the trial.
        
        The pathway timings and proportions of patients meeting various
        criteria are chosen to match the target hospital data,
        for example the distributions of onset to arrival times
        and the proportion of patients with known onset time.
        
        Each of the created patient arrays contains n values, one for
        each of the n patients in the trial. The xth value in all
        lists refers to the same patient. The data arrays are stored
        in the dictionary self.trial and are outputted here as a single
        pandas DataFrame.
        
        Onset   Arrival   Scan         Treatment
        o--------o-------------o----------o
            o----o-------o---------o
          o------o----------o---------o
         ? ? ? ? o-------o
         
         # ############ if patient not treated, set scan to needle time to NaN.
         # ############ then how do we ensure that the dist is right? currently treatment is decided based on scan to needle time.

        ----- Method: ----- # ############################################################ update me.
        1. Generate pathway times. 
                  <-σ--->                   
                 ^  ▃                    Use the target mu and sigma to
                 |  █▄                   make a lognorm distribution.
                 | ▆██▄                  
         Number  | ████                  Pick a time for each patient
           of    | ████▆                 from this distribution.
        patients | ██████▅▂              
                 | ████████▇▆▄▂          Check that the proportions of
                 |▆█████████████▇▆▅▄▃▂▁  patients with times under the
                 +--|----------------->  limit for IVT and MT match the
                    μ    Time            target proportions.
           This is used to create:
           + onset to arrival time
           + arrival to scan time
           + scan to needle time (thrombolysis)
           + scan to puncture time (thrombectomy)
           
        2. Generate whether onset time is known.
           Use the target proportion of known onset time.
           Randomly select patients to have known onset time so that
           the proportion known is the same.
        
        3. Generate whether patients receive treatments.
           Use the target proportions of patients receiving
           thrombolysis and thrombectomy given that they meet all of:
           + known onset time
           + arrival within x hours
           + scan within x hours of arrival
           + scan within x hours of onset
           + enough time left for treatments
           
           Randomly select patients that meet these conditions to
           receive thrombolysis so that the proportion matches the 
           target. Then more carefully randomly select patients to
           receive thrombectomy so that both the proportion receiving
           thrombectomy and the proportion receiving both treatments
           match the targets.
           
        4. Assign stroke types.
           n.b. this is not used elsewhere in this class but is useful
                for future modelling, e.g. in the outcome modelling.
           Use target proportions of patients with each stroke type.
           Assign nLVO, LVO, and "else" stroke types to the patients
           such that the treatments given make sense. Only patients
           with LVOs may receive thrombectomy, and only patients with
           nLVO or LVO may receive thrombolysis.
        
        ----- Outputs: -----
        results_dataframe - pandas DataFrame. Contains all of the
                            useful patient array data that was created
                            during the trial run.
        
        The useful patient array data is also available in the 
        self.trial attribute, which is a dictionary.
        
        """
        if patients_per_run > 0:
            # Overwrite the default value from the hospital data.
            self.patients_per_run = patients_per_run
        else:
            # Don't update anything.
            pass
        
        # Assign randomly whether the onset time is known
        # in the same proportion as the real performance data.
        self._generate_onset_time_known_binomial()
        self._create_masks_onset_time_known()
        # Generate pathway times for all patients.
        # These sets of times are all independent of each other.
        self._sample_onset_to_arrival_time_lognorm()
        self._create_masks_onset_to_arrival_on_time()
        
        self._sample_arrival_to_scan_time_lognorm()
        self._create_masks_arrival_to_scan_on_time()
        
        # Combine these generated times into other measures:
        self._calculate_onset_to_scan_time()
        self._create_masks_onset_to_scan_on_time()
        
        # Is there enough time left for treatment?
        self._calculate_time_left_for_ivt_after_scan()
        self._calculate_time_left_for_mt_after_scan()
        self._create_masks_enough_time_to_treat()
        
        # Treatment decision
        self._generate_whether_ivt_chosen_binomial()
        self._generate_whether_mt_chosen_binomial()
        self._create_masks_treatment_given()
        
        # 
        self._sample_scan_to_needle_time_lognorm()
        self._sample_scan_to_puncture_time_lognorm()
        self._calculate_onset_to_needle_time()
        self._calculate_onset_to_puncture_time()

        # Check that proportion of patients answering "yes" to each
        # mask matches the target proportions.        
        self._sanity_check_masked_patient_proportions()
        
        # Assign which type of stroke it is *after* choosing which
        # treatments are given.
        self._assign_stroke_type_code()

        # Place all useful outputs into a pandas Dataframe:
        results_dataframe = self._gather_results_in_dataframe()
        return results_dataframe


    # ###################################
    # ##### PATHWAY TIME GENERATION #####
    # ###################################

    def _generate_lognorm_times(
            self,
            proportion_on_time: float=None,
            number_of_patients: int=self.patients_per_run,
            mu_mt: float=None,
            sigma_mt: float=None,
            mu_ivt: float=None,
            sigma_ivt: float=None,
            label_for_printing: str=''
            ):
        """
        Generate times from a lognorm distribution and sanity check.
        
                  <-σ--->                   
                 ^  ▃                    Use the target mu and sigma to
                 |  █▄                   make a lognorm distribution.
                 | ▆██▄                  
         Number  | ████                  Pick a time for each patient
           of    | ████▆                 from this distribution.
        patients | ██████▅▂              
                 | ████████▇▆▄▂          Check that the proportions of
                 |▆█████████████▇▆▅▄▃▂▁  patients with times under the
                 +--|----------------->  limit for IVT and MT match the
                    μ    Time            target proportions.
        
        If mu and sigma for thrombectomy are provided, the generated
        times may be used for two different checks:
        - cut off at the thrombectomy limit,
          ensure proportion under thrombectomy limit matches target,
          compare cut-off distribution's mu and sigma with targets.
        - cut off at the thrombolysis limit,
          compare cut-off distribution's mu and sigma with targets.
          
        If the distribution for thrombectomy matches the target,
        then expect cutting off this distribution at the thrombolysis
        limit to match that target too. In the real data, the two are
        identical distributions just with different cut-off points.
        
        If only mu and sigma for thombolysis are provided, the checks
        against the thrombectomy limit are not done.
                    
        ----- Inputs: -----
        proportion_on_time - float or None. The target proportion of 
                             patients with times below the limit. If
                             this is None, the checks are not made.
        number_of_patients - int. Number of times to generate.
        mu_mt              - float or None. Lognorm mu for times 
                             below the thrombectomy limit.
        sigma_mt           - float or None. Lognorm sigma for 
                             times below the thrombectomy limit.
        mu_ivt             - float or None. Lognorm mu for times 
                             below the thrombolysis limit.
        sigma_ivt          - float or None. Lognorm sigma for 
                             times below the thrombolysis limit.
        label_for_printing - str. Identifier for the warning
                             string printed if sanity checks fail.
                                  
        ----- Returns: -----
        times_mins - np.array. The sanity-checked generated times.
        """
        # Select which mu and sigma to use:
        mu = mu_mt if mu_mt is not None else mu_ivt
        sigma = sigma_mt if sigma_mt is not None else sigma_ivt
        time_limit_mins = (self.limit_mt_mins if mu_mt is not None 
                           else self.limit_ivt_mins)
        
        # Generate times:
        times_mins = np.random.lognormal(
            mu,                  # mean
            sigma,               # standard deviation
            number_of_patients   # number of samples
            )
        # Round times to nearest minute:
        times_mins = np.round(times_mins, 0)
        # Set times below zero to zero:
        times_mins = np.maximum(times_mins, 0)

        if proportion_on_time is not None:
            times_mins = self._fudge_patients_after_time_limit(
                times_mins,
                proportion_on_time,
                time_limit_mins
                )

        # Sanity checks:
        if mu_mt is not None and sigma_mt is not None:
            self._check_distribution_statistics(
                times_mins[times_mins <= self.limit_mt_mins],
                mu_mt,
                sigma_mt,
                label=label_for_printing + ' on time for thrombectomy'
                )
        if mu_ivt is not None and sigma_ivt is not None:
            self._check_distribution_statistics(
                times_mins[times_mins <= self.limit_ivt_mins],
                mu_ivt,
                sigma_ivt,
                label=label_for_printing + ' on time for thrombolysis'
                )
        
        return times_mins


    def _sample_onset_to_arrival_time_lognorm(self):
        """
        Assign onset to arrival time (natural log normal distribution).

        Creates:
        --------
        onset_to_arrival_mins -
            Onset to arrival times in minutes from the log-normal
            distribution. One time per patient.
        onset_to_arrival_on_time_ivt_bool -
            True or False for each patient arriving under
            the time limit for thrombolysis.
        onset_to_arrival_on_time_mt_bool -
            True or False for each patient arriving under
            the time limit for thrombectomy.
            
        Uses:
        -----
        onset_time_known_bool -
            Whether each patient has a known onset time. Created in
            _generate_onset_time_known_binomial().
        """
        # Initial array with all zero times:
        trial_onset_to_arrival_mins = np.zeros(self.patients_per_run)
        
        # Update onset-to-arrival times so that when onset time is 
        # unknown, the values are set to NaN (time).
        inds = (self.trial['onset_time_known_bool'].data == 0)
        trial_onset_to_arrival_mins[inds] = np.NaN
        
        # Find which patients have known onset times:
        inds_valid_times = np.where(trial_onset_to_arrival_mins == 0)[0]
        
        # Invent new times for this known-onset-time subgroup:
        valid_onset_to_arrival_mins = self._generate_lognorm_times(
            self.target_data_dict[
                'proportion2_of_mask1_with_onset_to_arrival_on_time_mt'],
            len(inds_valid_times),
            self.target_data_dict['lognorm_mu_onset_arrival_mt_mins'],
            self.target_data_dict['lognorm_sigma_onset_arrival_mt_mins'],
            self.target_data_dict['lognorm_mu_onset_arrival_ivt_mins'],
            self.target_data_dict['lognorm_sigma_onset_arrival_ivt_mins'],
            'onset to arrival'
            )
        # Place these times into the full patient list:
        trial_onset_to_arrival_mins[inds_valid_times] = \
            valid_onset_to_arrival_mins
        
        # Store the generated times:
        self.trial['onset_to_arrival_mins'].data = \
            trial_onset_to_arrival_mins

        # Also generate and store boolean arrays of
        # whether the times are below given limits.
        # (NaN times return False in these bool lists.)
        self.trial['onset_to_arrival_on_time_ivt_bool'].data = \
            (trial_onset_to_arrival_mins <= self.limit_ivt_mins)
        self.trial['onset_to_arrival_on_time_mt_bool'].data = \
            (trial_onset_to_arrival_mins <= self.limit_mt_mins)


    def _sample_arrival_to_scan_time_lognorm(self):
        """
        Assign arrival to scan time (natural log normal distribution).
        
        Creates:
        --------
        arrival_to_scan_mins -
            Arrival to scan times in minutes from the log-normal
            distribution. One time per patient.
        arrival_to_scan_on_time_ivt_bool -
            True or False for each patient arriving under
            the time limit for thrombolysis.
        arrival_to_scan_on_time_mt_bool -
            True or False for each patient arriving under
            the time limit for thrombectomy.
        """
        # need to check these proportions - comparing the generated times across th efull cohort with proportions from the real hospital data that only appl yto a subset.
        #does this actully mametter? or is it likely ot matter enough that erw want to ensuer the proportion matches exactly? caould jusst chace that the proportion masked ends up matching by luck.
        # CHECK ME AND DECIDE 
        # there's no guarantee tha tthis proportion under the limit will be anything like th etarget by the time we've masked this proportion after the fact.
        # so probably need to chang etihs?
        # Generat ein two batches, those within the mask and those without? Ensures proportion but sitll otheriwese random times.
        
        # if mask provided, do it in two batches, otherwise don't bother? default mask is 1 to everything?
        # Store the generated times in here:
        trial_arrival_to_scan_mins = np.zeros(self.patients_per_run, 
                                              dtype=float)
        # Generate the times in two batches, once for all of the values
        # where mask is True and once for all of the values where mask
        # is False. This ensures that the distribution statistics 
        # match the target values in the masked data.
        try:
            mask1 = self.trial['ivt_mask2_mask1_and_onset_to_arrival_on_time'].data
            mask2 = self.trial['mt_mask2_mask1_and_onset_to_arrival_on_time'].data
            mask = np.sum([mask1, mask2])
        except KeyError:
            mask = np.zeros(self.patients_per_run, dtype=float)
        # but have separate IVT and MT masks...! What to do now? Similar if ok for 8hr then cut down for 4hr, still holds?
        # I don't think that will hold because the masks are so different. Would have the same problem.
        # So do batches of on time for IVT, then on time for MT but not IVT, then on time for neither...
        for b in [0, 1, 2]:
            inds = np.where(mask == b)[0]
            if b == 2:
                # If on time for IVT, don't check the times against MT:
                mu_mt = None
                sigma_mt = None
            else:
                mu_mt = self.target_data_dict[
                    'lognorm_mu_arrival_scan_arrival_mt_mins']
                sigma_mt = self.target_data_dict[
                    'lognorm_sigma_arrival_scan_arrival_mt_mins']
            # Invent new times for the patient subgroup:
            masked_arrival_to_scan_mins = self._generate_lognorm_times(
                self.target_data_dict[
                    'proportion3_of_mask2_with_arrival_to_scan_on_time_mt'],
                len(inds),
                mu_mt,
                sigma_mt,
                self.target_data_dict[
                    'lognorm_mu_arrival_scan_arrival_ivt_mins'],
                self.target_data_dict[
                    'lognorm_sigma_arrival_scan_arrival_ivt_mins'],
                'arrival to scan'
                )
            trial_arrival_to_scan_mins[inds] = masked_arrival_to_scan_mins

        # Store the generated times:
        self.trial['arrival_to_scan_mins'].data = \
            trial_arrival_to_scan_mins

        # Also generate and store boolean arrays of
        # whether the times are below given limits.
        self.trial['arrival_to_scan_on_time_ivt_bool'].data = \
            (trial_arrival_to_scan_mins <= self.limit_ivt_mins)
        self.trial['arrival_to_scan_on_time_mt_bool'].data = \
            (trial_arrival_to_scan_mins <= self.limit_mt_mins)


    def _sample_scan_to_needle_time_lognorm(self):
        """
        Assign scan to needle time (natural log normal distribution).
        
        Creates:
        --------
        scan_to_needle_mins -
            Scan to needle times in minutes from the log-normal
            distribution. One time per patient.
        scan_to_needle_on_time_ivt_bool -
            True or False for each patient arriving under
            the time limit for thrombolysis.
        scan_to_needle_on_time_mt_bool -
            True or False for each patient arriving under
            the time limit for thrombectomy.
        """
        # Store the generated times in here:
        trial_arrival_to_scan_mins = np.zeros(self.patients_per_run, 
                                              dtype=float)
        # Generate the times in two batches, once for all of the values
        # where mask is True and once for all of the values where mask
        # is False. This ensures that the distribution statistics 
        # match the target values in the masked data.
        try:
            mask = self.trial['ivt_mask5_mask4_and_enough_time_to_treat'].data
        except KeyError:
            mask = np.zeros(self.patients_per_run, dtype=float)
        for b in [0, 1]:
            inds = np.where(mask == b)[0]

            # Invent new times for the patient subgroup:
            masked_scan_to_needle_mins = self._generate_lognorm_times(
                self.target_data_dict['proportion_scan_to_needle_on_time'],
                len(inds),
                mu_ivt=self.target_data_dict['lognorm_mu_scan_needle_mins'],
                sigma_ivt=self.target_data_dict['lognorm_sigma_scan_needle_mins'],
                label_for_printing='scan to needle'
                )
            
            trial_scan_to_needle_mins[inds] = masked_scan_to_needle_mins

        # Store the generated times:
        self.trial['scan_to_needle_mins'].data = \
            trial_scan_to_needle_mins

        # Also generate and store boolean arrays of
        # whether the times are below given limits.
        self.trial['scan_to_needle_on_time_ivt_bool'].data = \
            (trial_scan_to_needle_mins <= self.limit_ivt_mins)
        self.trial['scan_to_needle_on_time_mt_bool'].data = \
            (trial_scan_to_needle_mins <= self.limit_mt_mins)


    def _sample_scan_to_puncture_time_lognorm(self):
        """
        Assign scan to puncture time (natural log normal distribution).
        
        Creates:
        --------
        scan_to_puncture_mins -
            Scan to puncture times in minutes from the log-normal
            distribution. One time per patient.
        scan_to_puncture_on_time_ivt_bool -
            True or False for each patient arriving under
            the time limit for thrombolysis.
        scan_to_puncture_on_time_mt_bool -
            True or False for each patient arriving under
            the time limit for thrombectomy.
        """
        # Store the generated times in here:
        trial_scan_to_puncture_mins = np.zeros(self.patients_per_run, 
                                              dtype=float)
        # Generate the times in two batches, once for all of the values
        # where mask is True and once for all of the values where mask
        # is False. This ensures that the distribution statistics 
        # match the target values in the masked data.
        try:
            mask = self.trial['mt_mask5_mask4_and_enough_time_to_treat'].data
        except KeyError:
            mask = np.zeros(self.patients_per_run, dtype=float)
        for b in [0, 1]:
            inds = np.where(mask == b)[0]

            # Invent new times for the patient subgroup:
            masked_scan_to_puncture_mins = self._generate_lognorm_times(
                self.target_data_dict['proportion_scan_to_puncture_on_time'],
                len(inds),
                mu_mt=self.target_data_dict['lognorm_mu_scan_puncture_mins'],
                sigma_mt=self.target_data_dict[
                    'lognorm_sigma_scan_puncture_mins'],
                label_for_printing='scan to puncture'
                )
            trial_scan_to_puncture_mins[inds] = masked_scan_to_puncture_mins
            
        # Store the generated times:
        self.trial['scan_to_puncture_mins'].data = trial_scan_to_puncture_mins

        # Also generate and store boolean arrays of
        # whether the times are below given limits.
        self.trial['scan_to_puncture_on_time_ivt_bool'].data = \
            (trial_scan_to_puncture_mins <= self.limit_ivt_mins)
        self.trial['scan_to_puncture_on_time_mt_bool'].data = \
            (trial_scan_to_puncture_mins <= self.limit_mt_mins)


    # #######################################
    # ##### PATHWAY BINOMIAL GENERATION #####
    # #######################################
    def _generate_onset_time_known_binomial(self):
        """
        Assign whether onset time is known for each patient.

        Creates:
        --------
        onset_time_known_bool -
            True or False for each patient having a known onset time.
        """
        self.trial['onset_time_known_bool'].data = \
            np.random.binomial(
                1,                          # Number of trials
                self.target_data_dict['proportion1_of_all_with_onset_known_ivt'],  
                                            # ^ Probability of success
                self.patients_per_run       # Number of samples drawn
                ) == 1                      # Convert int to bool



    def _generate_whether_ivt_chosen_binomial(self):
        """
        Generate whether patients receive thrombolysis (IVT).
        
        Use the target proportion of patients receiving
        thrombolysis given that they meet all of:
        + known onset time
        + arrival within x hours
        + scan within x hours of arrival
        + scan within x hours of onset
        + enough time left for treatments

        Randomly select patients that meet these conditions to
        receive thrombolysis so that the proportion matches the    # ################ change this to not depend on scan to needle time
        target.

        Creates:
        --------
        ivt_chosen_bool - 
            True or False for each patient receiving thrombolysis.
            
        Uses:
        -----
        ivt_mask5_mask4_and_enough_time_to_treat -
            True or False for each patient being eligible for 
            thrombolysis. Created in 
            _create_masks_enough_time_to_treat().
        """
        # Find the indices of patients that meet thrombolysis criteria:
        inds_scan_on_time = np.where(
            self.trial[
                'ivt_mask5_mask4_and_enough_time_to_treat'].data == 1)[0]
        n_scan_on_time = len(inds_scan_on_time)

        # How many of these patients do we expect to receive 
        # thrombolysis?
        n_thrombolysed = int(round(
            n_scan_on_time * self.target_data_dict[
                'proportion6_of_mask5_with_treated_ivt'], 0))
        
        # Find the patients that have onset to needle on time:
        inds_needle_on_time = np.where(
            (self.trial['onset_to_needle_on_time_bool'].data == 1) & 
            (inds_scan_on_time == 1))[0]
        n_needle_on_time = len(inds_needle_on_time)
        # Find the patients that have late onset to needle:
        inds_needle_late = np.where(
            (self.trial['onset_to_needle_on_time_bool'].data == 0) & 
            (inds_scan_on_time == 1))[0]
        
        # Initially select the subgroup on time to receive 
        # thrombolysis, and if necessary pick patients who have late
        # onset-to-needle times.
        trial_ivt_chosen_bool = self._select_inds_prioritising_on_time(
            n_thrombolysed,
            inds_needle_on_time,
            inds_needle_late,
            )
                
        # Store this in self (==1 to convert to boolean).
        self.trial['ivt_chosen_bool'].data = trial_ivt_chosen_bool == 1


    def _generate_whether_mt_chosen_binomial(self):
        """
        Generate whether patients receive thrombectomy (MT).
        
        Use the target proportion of patients receiving
        thrombectomy given that they meet all of:
        + known onset time
        + arrival within x hours
        + scan within x hours of arrival
        + scan within x hours of onset
        + enough time left for treatments

        Randomly select patients that meet these conditions to
        receive thrombectomy so that the proportion matches the 
        target. The selection is done in two steps to account for
        some patients also receiving IVT and some not, in a known   # ################ change this to not depend on scan to needle time
        target proportion.

        Creates:
        --------
        mt_chosen_bool - 
            True or False for each patient receiving thrombectomy.
        
        Uses:
        -----
        mt_mask5_mask4_and_enough_time_to_treat -
            True or False for each patient being eligible for 
            thrombectomy. Created in 
            _create_masks_enough_time_to_treat().
        ivt_chosen_bool -
            True of False for each patient receiving thrombolysis.
            Created in _generate_whether_ivt_chosen_binomial().
        """
        # Find how many patients could receive thrombectomy and
        # where they are in the patient array:
        inds_eligible_for_mt = np.where(
            self.trial[
                'mt_mask5_mask4_and_enough_time_to_treat'].data == 1)[0]
        n_eligible_for_mt = len(inds_eligible_for_mt)
        # Use the known proportion chosen to find the number of
        # patients who will receive thrombectomy:
        n_mt = int(n_eligible_for_mt *
                   self.target_data_dict['proportion6_of_mask5_with_treated_mt'])

        # Use the proportion of patients who receive thrombolysis and
        # thrombectomy to create two groups now.
        # Number of patients receiving both:
        n_mt_and_ivt = int(n_mt * self.target_data_dict[
            'proportion_of_mt_also_receiving_ivt'])
        # Number of patients receiving thrombectomy only:
        n_mt_not_ivt = n_mt - n_mt_and_ivt

        # Find which patients in the array may be used for each group:
        inds_eligible_for_mt_and_ivt_on_time = np.where(
            (self.trial[
                'mt_mask5_mask4_and_enough_time_to_treat'].data == 1) &
            (self.trial['ivt_chosen_bool'].data == 1) &
            (self.trial['onset_to_puncture_on_time_bool'].data == 1)
            )[0]
        
        inds_eligible_for_mt_and_ivt_late = np.where(
            (self.trial[
                'mt_mask5_mask4_and_enough_time_to_treat'].data == 1) &
            (self.trial['ivt_chosen_bool'].data == 1) &
            (self.trial['onset_to_puncture_on_time_bool'].data == 0)
            )[0]
        
        inds_eligible_for_mt_not_ivt_on_time = np.where(
            (self.trial[
                'mt_mask5_mask4_and_enough_time_to_treat'].data == 1) &
            (self.trial['ivt_chosen_bool'].data == 0) &
            (self.trial['onset_to_puncture_on_time_bool'].data == 1)
            )[0]
        
        inds_eligible_for_mt_not_ivt_late = np.where(
            (self.trial[
                'mt_mask5_mask4_and_enough_time_to_treat'].data == 1) &
            (self.trial['ivt_chosen_bool'].data == 0) &
            (self.trial['onset_to_puncture_on_time_bool'].data == 0)
            )[0]
        
        # Randomly select patients from these subgroups to be given
        # thrombectomy, prioritising those with onset to puncture on time.
        trial_mt_and_ivt_chosen_bool = self._select_inds_prioritising_on_time(
            n_mt_and_ivt,
            inds_eligible_for_mt_and_ivt_on_time,
            inds_eligible_for_mt_and_ivt_late,
            )
        trial_mt_not_ivt_chosen_bool = self._select_inds_prioritising_on_time(
            n_mt_not_ivt,
            inds_eligible_for_mt_not_ivt_on_time,
            inds_eligible_for_mt_not_ivt_late,
            )
        trial_mt_chosen = np.sum(trial_mt_and_ivt_chosen_bool, 
                                 trial_mt_not_ivt_chosen_bool)
        
        # Store in self (==1 to convert to boolean):
        self.trial['mt_chosen_bool'].data = trial_mt_chosen_bool == 1

        
    def _select_inds_prioritising_on_time(
            n_to_pick: int,
            inds_on_time: np.ndarray,
            inds_late: np.ndarray,
            n_in_full_list: int=self.patients_per_run
            ):
        """
        Select values from two lists, prioritising one of them.

        Pick the indices first out of the inds_on_time list,
        and if there are any left over then pick them from inds_late.
        Then set those indices in the output picked_bool list to be 1
        and keep the not-picked values at 0.

        Inputs:
        -------
        n_to_pick      - int. Number of values to pick from the lists.
        inds_on_time   - np.ndarray. First list of values.
        inds_late      - np.ndarray. Bonus list of values.
        n_in_full_list - int. How many values are in the full lists,
                         for example the total number of patients.

        Returns:
        --------
        picked_bool - np.ndarray. Array of n_in_full_list values where
                      the indices picked from the lists are set to 1
                      and those not picked are set to 0.
        """
        # Initially create the array with nobody picked.
        picked_bool = np.zeros(n_in_full_list, dtype=int)
        
        if n_to_pick <= 0:
            pass  # Don't do anything.
        else:
            # Numbers of patients in the lists to choose from:
            n_on_time = len(inds_on_time)
            n_late = len(inds_late)

            # If there are enough people in the first list...
            if n_on_time >= n_to_pick:
                # Pick people from the subgroup with treatment on time.
                inds_picked_on_time = np.random.choice(
                    inds_on_time,
                    size=n_to_pick,
                    replace=False
                    )
                # Don't pick anyone who was treated after the limit.
                inds_picked_late = np.array([], dtype=int)
            # ... otherwise use the second list too:
            else:
                # Pick everyone treated on time.
                inds_picked_on_time = inds_on_time
                # Pick people from the subgroup with late treatment.
                inds_picked_late = np.random.choice(
                    inds_late,
                    size=n_to_pick - n_on_time,
                    replace=False
                    )

            # Update the patients that we've just picked out.
            picked_bool[inds_picked_on_time] = 1
            picked_bool[inds_picked_late] = 1

        return picked_bool
    
    
    # ##############################
    # ##### COMBINE CONDITIONS #####
    # ##############################
    def _calculate_onset_to_scan_time(self):
        """
        Find onset to scan time and boolean arrays from existing times.

        Creates:
        --------
        onset_to_scan_mins -
            Onset to scan times in minutes from the log-normal
            distribution. One time per patient.
        onset_to_scan_on_time_ivt_bool -
            True or False for each patient being scanned under
            the time limit for thrombolysis.
        onset_to_scan_on_time_mt_bool -
            True or False for each patient being scanned under
            the time limit for thrombectomy.
        
        Uses:
        -----
        onset_to_arrival_mins -
            Onset to arrival times in minutes from the log-normal
            distribution. One time per patient. Created in
            _sample_onset_to_arrival_time_lognorm().
        arrival_to_scan_mins -
            Arrival to scan times in minutes from the log-normal
            distribution. One time per patient. Created in
            _sample_arrival_to_scan_time_lognorm().
        """
        # Calculate onset to scan by summing onset to arrival and
        # arrival to scan:
        self.trial['onset_to_scan_mins'].data = np.sum([
            self.trial['onset_to_arrival_mins'].data,
            self.trial['arrival_to_scan_mins'].data,
            ], axis=0)

        # Create boolean arrays for whether each patient's time is
        # below the limit.
        self.trial['onset_to_scan_on_time_ivt_bool'].data = \
            (self.trial['onset_to_scan_mins'].data <= self.limit_ivt_mins)
        self.trial['onset_to_scan_on_time_mt_bool'].data = \
            (self.trial['onset_to_scan_mins'].data <= self.limit_mt_mins)


    def _calculate_time_left_for_ivt_after_scan(
            self,
            minutes_left: float=15.0
            ):
        """
        Calculate the minutes left to thrombolyse after scan.
        
        Creates:
        --------
        time_left_for_ivt_after_scan_mins -
            Time left before the allowed onset to needle time in 
            minutes. If the allowed time has passed, the time left
            is set to zero. One time per patient.
        enough_time_for_ivt_bool -
            True or False for each patient having enough time left
            for thrombolysis.
        
        Uses:
        -----
        onset_to_scan_mins -
            Onset to scan times in minutes from the log-normal
            distribution. One time per patient. Created in
            _calculate_onset_to_scan_time().
        ivt_mask4_mask3_and_onset_to_scan_on_time -
            True or False for each patient being eligible for 
            thrombolysis. Created in 
            _create_masks_enough_time_to_treat().
        """
        self.trial['time_left_for_ivt_after_scan_mins'].data = np.maximum(( # ################################### update comments here
            self.allowed_onset_to_needle_time_mins -
            self.trial['onset_to_scan_mins'].data
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

    def _calculate_time_left_for_mt_after_scan(
            self,
            minutes_left: float=15.0
            ):
        """
        Calculate the minutes left for thrombectomy after scan.
        
        Creates:
        --------
        time_left_for_mt_after_scan_mins -
            Time left before the allowed onset to puncture time in 
            minutes. If the allowed time has passed, the time left
            is set to zero. One time per patient.
        enough_time_for_mt_bool -
            True or False for each patient having enough time left
            for thrombectomy.
        
        Uses:
        -----
        onset_to_scan_mins -
            Onset to scan times in minutes from the log-normal
            distribution. One time per patient. Created in
            _calculate_onset_to_scan_time().
        mt_mask4_mask3_and_onset_to_scan_on_time -
            True or False for each patient being eligible for 
            thrombectomy. Created in 
            _create_masks_enough_time_to_treat().
        """
        self.trial['time_left_for_mt_after_scan_mins'].data = np.maximum((
            self.allowed_onset_to_puncture_time_mins -
            self.trial['onset_to_scan_mins'].data 
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

    def _calculate_onset_to_needle_time(self):):
        """
        Calculate onset to needle times from existing data.
        
        Creates:
        --------
        onset_to_needle_mins -
            Onset to needle times in minutes from summing the onset to
            scan and scan to needle times and setting a maximum value
            of clip_limit_mins. One time per patient.
        
        Uses:
        -----
        onset_to_scan_mins -
            Onset to scan times in minutes from the log-normal
            distribution. One time per patient. Created in
            _calculate_onset_to_scan_time().
        scan_to_needle_mins -
            Scan to needle times in minutes from the log-normal
            distribution. One time per patient. Created in
            _sample_scan_to_needle_time_lognorm().
        """
        onset_to_needle_mins = (
            self.trial['onset_to_scan_mins'].data +
            self.trial['scan_to_needle_mins'].data
            )
        self.trial['onset_to_needle_mins'].data = onset_to_needle_mins
        
        # Which patients have onset to needle time below the limit?
        onset_to_needle_limit_mins=(
                self.allowed_onset_to_needle_time_mins +
                self.allowed_overrun_for_slow_scan_to_needle_mins
                )
        self.trial['onset_to_needle_on_time_bool'].data = (
            self.trial['onset_to_needle_mins'].data <= 
            onset_to_needle_limit_mins)


    def _calculate_onset_to_puncture_time(self):
        """
        Calculate onset to puncture times from existing data.
        
        Creates:
        --------
        onset_to_puncture_mins -
            Onset to puncture times in minutes from summing the onset
            to scan and scan to puncture times and setting a maximum 
            value of clip_limit_mins. One time per patient.
        
        Uses:
        -----
        onset_to_scan_mins -
            Onset to scan times in minutes from the log-normal
            distribution. One time per patient. Created in
            _calculate_onset_to_scan_time().
        scan_to_puncture_mins -
            Scan to puncture times in minutes from the log-normal
            distribution. One time per patient. Created in
            _sample_scan_to_puncture_time_lognorm().
        """
        onset_to_puncture_mins = (
            self.trial['onset_to_scan_mins'].data +
            self.trial['scan_to_puncture_mins'].data
            )
        self.trial['onset_to_puncture_mins'].data = onset_to_puncture_mins
        
        # Which patients have onset to needle time below the limit?
        onset_to_puncture_limit_mins=(
            self.allowed_onset_to_puncture_time_mins +
            self.allowed_overrun_for_slow_scan_to_puncture_mins
            )
        self.trial['onset_to_puncture_on_time_bool'].data = (
            self.trial['onset_to_puncture_mins'].data <= 
            onset_to_puncture_limit_mins)


    def _calculate_ivt_rate(self):
        """
        Calculate thrombolysis rate across the whole cohort.
        
        Creates:
        --------
        ivt_rate_percent - 
            Float. The proportion of the whole cohort that received
            thrombolysis, as a percent.
        
        Uses:
        -----
        ivt_chosen_bool -
            True of False for each patient receiving thrombolysis.
            Created in _generate_whether_ivt_chosen_binomial().
        """
        self.trial['ivt_rate_percent'].data = \
            self.trial['ivt_chosen_bool'].data.mean() * 100


    def _calculate_mt_rate(self):
        """
        Calculate thrombectomy rate across the whole cohort.
        
        Creates:
        --------
        mt_rate_percent - 
            Float. The proportion of the whole cohort that received
            thrombectomy, as a percent.
        
        Uses:
        -----
        mt_chosen_bool -
            True of False for each patient receiving thrombectomy.
            Created in _generate_whether_mt_chosen_binomial().
        """
        self.trial['mt_rate_percent'].data = \
            self.trial['mt_chosen_bool'].data.mean() * 100


    def _gather_results_in_dataframe(self):
        """
        Combine all results arrays into a single dataframe.
        
        The gathered arrays are all contained in the trial dictionary.
        """        
        # The following construction for "data" will work as long as
        # all arrays in trial have the same length.
        df = pd.DataFrame(
            data=np.array(
                [v.data for v in self.trial.values()], dtype=object).T,
            columns=list(self.trial.keys())
            )
        self.results_dataframe = df
        

    # ##################################
    # ######## STROKE TYPE CODE ########
    # ##################################

    def _assign_stroke_type_code(self):
        """
        Assign stroke type based partly on treatment decision.

        Available combinations:
        +------+------+--------------+--------------|
        | Type | Code | Thrombolysis | Thrombectomy |
        +------+------+--------------+--------------|
        | nLVO | 1    | Yes or no    | No           |
        | LVO  | 2    | Yes or no    | Yes or no    |
        | Else | 0    | No           | No           |
        +------+------+--------------+--------------|
        
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
                        <-LVO--><-nLVO->     into Group ░B░.

           Calculate how many patients should have nLVO according to
           the target proportion. Set the number of nLVO patients in
           Group ▒C▒ to either this number or the total number of 
           patients in Group ▒C▒, whichever is smaller. Then
           the rest of the patients in Group ▒C▒ are assigned as LVO.
           The specific patients chosen for each are picked at random.
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
        total = dict(
            total = self.patients_per_run,
            lvo = int(
                round(self.patients_per_run * self.proportion_lvo, 0)),
            nlvo = int(
                round(self.patients_per_run * self.proportion_nlvo, 0))
        )
        total['other'] = (
            self.patients_per_run - (total['lvo'] + total['nlvo']))
        
        
        # Find which patients are in each group.
        inds_groupA = np.where(
            (self.trial['mt_chosen_bool'].data == 1) & 
            (self.trial['ivt_chosen_bool'].data == 0))[0]
        inds_groupB = np.where(
            (self.trial['ivt_chosen_bool'].data == 1) &
            (self.trial['mt_chosen_bool'].data == 1))[0]
        inds_groupBC = np.where(
            self.trial['ivt_chosen_bool'].data == 1)[0]
        inds_groupC = np.where(
            (self.trial['ivt_chosen_bool'].data == 1) &
            (self.trial['mt_chosen_bool'].data == 0))[0]
        
        # Find how many patients in each group have each stroke type.
        # Step 1: all thrombectomy patients are LVO.
        groupA = dict(
            total = len(inds_groupA),
            lvo = len(inds_groupA),
            nlvo = 0,
            other = 0,
            )
        groupB = dict(
            total = len(inds_groupB),
            lvo = len(inds_groupB),
            nlvo = 0,
            other = 0
            )
        
        # Step 2: all thrombolysis patients have nLVO or LVO
        # in the same ratio as the patient proportions if possible.
        # Work out how many people have nLVO and were thrombolysed.
        # This is the smaller of: 
        # the total number of people in group C...
        n1 = len(inds_groupC)
        # ... and the number of people in groups B and C that 
        # should have nLVO according to the target proportion...
        n2 = int(
            round(len(inds_groupBC) * (1.0 - self.proportion_lvo), 0)
            )
        # ... to give this value:
        n_nlvo_and_thrombolysis = np.minimum(n1, n2)
        
        groupC = dict(
            total = len(inds_groupC),
            nlvo = n_nlvo_and_thrombolysis,
            lvo = len(inds_groupC) - n_nlvo_and_thrombolysis,
            other = 0
            )
                
        # Randomly select which patients in Group C have each type.
        # For LVO, select these people... 
        inds_lvo_groupC = np.random.choice(
            inds_groupC,
            size=groupC['lvo'],
            replace=False
            )
        # ... and for nLVO, select everyone else.
        inds_nlvo_groupC = np.array(list(
            set(inds_groupC) -
            set(inds_lvo_groupC)
            ))
        
        # Bookkeeping:
        # Set the chosen patients to their stroke types:
        trial_stroke_type_code[inds_groupA] = 2
        trial_stroke_type_code[inds_groupB] = 2
        trial_stroke_type_code[inds_lvo_groupC] = 2
        trial_stroke_type_code[inds_nlvo_groupC] = 1
        
        # Keep track of which patients have been assigned a type:
        trial_type_assigned_bool[inds_groupA] += 1
        trial_type_assigned_bool[inds_groupB] += 1
        trial_type_assigned_bool[inds_groupC] += 1


        # Step 3: everyone else is in Group D.
        groupD = dict(
            total = (total['total'] - 
                    (groupA['total'] + groupB['total'] + groupC['total'])),
            lvo = (total['lvo'] - 
                  (groupA['lvo'] + groupB['lvo'] + groupC['lvo'])),
            nlvo = (total['nlvo'] - 
                   (groupA['nlvo'] + groupB['nlvo'] + groupC['nlvo'])),
            other = (total['other'] - 
                    (groupA['other'] + groupB['other'] + groupC['other']))
            )
        # For each stroke type, randomly select some indices out
        # of those that have not yet been assigned a stroke type.
        # LVO selects from everything in Group D:
        inds_groupD_lvo = np.random.choice(
            np.where(trial_type_assigned_bool == 0)[0],
            size=groupD['lvo'],
            replace=False
            )
        trial_type_assigned_bool[inds_groupD_lvo] += 1
        trial_stroke_type_code[inds_groupD_lvo] = 2
        
        # nLVO selects from everything in Group D that hasn't
        # already been assigned to LVO:
        inds_groupD_nlvo = np.random.choice(
            np.where(trial_type_assigned_bool == 0)[0],
            size=groupD['nlvo'],
            replace=False
            )
        trial_type_assigned_bool[inds_groupD_nlvo] += 1
        trial_stroke_type_code[inds_groupD_nlvo] = 1

        # Other types select from everything in Group D that hasn't
        # already been assigned to LVO or nLVO:
        inds_groupD_other = np.random.choice(
            np.where(trial_type_assigned_bool == 0)[0],
            size=groupD['other'],
            replace=False
            )
        trial_type_assigned_bool[inds_groupD_other] += 1
        trial_stroke_type_code[inds_groupD_other] = 0


        # ### Final check ###
        # Sanity check that each patient was assigned exactly 
        # one stroke type:
        if np.any(trial_type_assigned_bool > 1):
            print(''.join([
                'Warning: check the stroke type code assignment. ',
                'Some patients were assigned multiple stroke types. ']))
        if np.any(trial_type_assigned_bool == 0):
            print(''.join([
                'Warning: check the stroke type code assignment. ',
                'Some patients were not assigned a stroke type. ']))

        # Now store this final array in self:
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
    ▏Mask 5: Is there enough time left for thrombolysis/thrombectomy? ▕  # rem ove thus
    ░░░░░░░░░░░░░░░░▒▒▒▒▒▒▒████████████████████████████████████████████
    ▏------Yes------▏--No--▏------------------Rejected----------------▕
    ▏               ▏      ▏                                          ▕
    ▏Mask 6: Did the patient receive thrombolysis/thrombectomy?       ▕
    ░░░░░░░░░░░▒▒▒▒▒███████████████████████████████████████████████████
    ▏----Yes---▏-No-▏---------------------Rejected--------------------▕
    
    
    don't need the enough time left mask, just make sure taht the people given treatment are the ones where there is enough time left. The "yes to mask 4" group is the one that should be getting the thrombolysis rate from the hospital target data.
    Currently kept that mask in the hospital performacne though... ??
    can't have scan to needle on time in the real data extraction because the valu eis either >=0 or NaN.
    # ##################################################################################### decide  on this ok

    """
    def _create_masks_onset_time_known(self):
        """
        Mask 1: Is onset time known?
        
        Although this mask looks redundant, it is provided for easier
        direct comparison with the masks creating during the hospital
        performance data extraction. The IVT mask is identical to the
        MT mask and both are an exact copy of the onset known boolean
        array.
        
        Creates:
        --------
        ivt_mask1_onset_known -
            Mask of whether onset time is known for each patient.
        mt_mask1_onset_known -
            Mask of whether onset time is known for each patient.
        
        Uses:
        -----
        onset_time_known_bool - 
            Whether onset time is known for each patient. Created in
            _generate_onset_time_known_binomial().
        """
        # Same mask for thrombolysis and thrombolysis.
        mask = np.copy(self.trial['onset_time_known_bool'].data)

        self.trial['ivt_mask1_onset_known'].data = mask
        self.trial['mt_mask1_onset_known'].data = mask


    def _create_masks_onset_to_arrival_on_time(self):
        """
        Mask 2: Is arrival within x hours?
        
        Creates:
        --------
        ivt_mask2_mask1_and_onset_to_arrival_on_time -
            Mask of whether the onset to arrival time is below the
            thrombolysis limit and whether mask 1 is True 
            for each patient.
        mt_mask2_mask1_and_onset_to_arrival_on_time -
            Mask of whether the onset to arrival time is below the
            thrombectomy limit and whether mask 1 is True 
            for each patient.
        
        Uses:
        -----
        ivt_mask1_onset_known - 
            Whether onset time is known for each patient. Created in
            _create_masks_onset_time_known().
        onset_to_arrival_on_time_ivt_bool -
            Whether onset to arrival time for each patient is under 
            the thrombolysis limit. Created in 
            _sample_onset_to_arrival_time_lognorm().
        mt_mask1_onset_known - 
            Whether onset time is known for each patient. Created in
            _create_masks_onset_time_known(). 
        onset_to_arrival_on_time_mt_bool -
            Whether onset to arrival time for each patient is under 
            the thrombectomy limit. Created in 
            _sample_onset_to_arrival_time_lognorm().
        """
        mask_ivt = (
            (self.trial['ivt_mask1_onset_known'].data == 1) &
            (self.trial['onset_to_arrival_on_time_ivt_bool'].data == 1)
            )
        mask_mt = (
            (self.trial['mt_mask1_onset_known'].data == 1) &
            (self.trial['onset_to_arrival_on_time_mt_bool'].data == 1)
            )

        self.trial[
            'ivt_mask2_mask1_and_onset_to_arrival_on_time'].data = mask_ivt
        self.trial[
            'mt_mask2_mask1_and_onset_to_arrival_on_time'].data = mask_mt

        
    def _create_masks_arrival_to_scan_on_time(self):
        """
        Mask 3: Is scan within x hours of arrival?
                
        Creates:
        --------
        ivt_mask3_mask2_and_arrival_to_scan_on_time -
            Mask of whether the arrival to scan time is below the
            thrombolysis limit and whether mask 2 is True 
            for each patient.
        mt_mask3_mask2_and_arrival_to_scan_on_time -
            Mask of whether the arrival to scan time is below the
            thrombectomy limit and whether mask 2 is True 
            for each patient.
        
        Uses:
        -----
        ivt_mask2_mask1_and_onset_to_arrival_on_time - 
            IVT mask 2. Created in
            _create_masks_onset_to_arrival_on_time().
        arrival_to_scan_on_time_ivt_bool -
            Whether arrival to scan time for each patient is under 
            the thrombolysis limit. Created in 
            _sample_arrival_to_scan_time_lognorm().
        mt_mask2_mask1_and_onset_to_arrival_on_time - 
            MT mask 2. Created in
            _create_masks_onset_to_arrival_on_time(). 
        arrival_to_scan_on_time_mt_bool -
            Whether arrival to scan time for each patient is under 
            the thrombectomy limit. Created in 
            _sample_arrival_to_scan_time_lognorm().
        """
        mask_ivt = (
            (self.trial[
                'ivt_mask2_mask1_and_onset_to_arrival_on_time'].data == 1) &
            (self.trial['arrival_to_scan_on_time_ivt_bool'].data == 1)
            )
        mask_mt = (
            (self.trial[
                'mt_mask2_mask1_and_onset_to_arrival_on_time'].data == 1) &
            (self.trial['arrival_to_scan_on_time_mt_bool'].data == 1)
            )

        self.trial[
            'ivt_mask3_mask2_and_arrival_to_scan_on_time'].data = mask_ivt
        self.trial[
            'mt_mask3_mask2_and_arrival_to_scan_on_time'].data = mask_mt

        
    def _create_masks_onset_to_scan_on_time(self):
        """
        Mask 4: Is scan within x hours of onset?
                
        Creates:
        --------
        ivt_mask4_mask3_and_onset_to_scan_on_time -
            Mask of whether the onset to scan time is below the
            thrombolysis limit and whether mask 3 is True 
            for each patient.
        mt_mask4_mask3_and_onset_to_scan_on_time -
            Mask of whether the onset to scan time is below the
            thrombectomy limit and whether mask 3 is True 
            for each patient.
        
        Uses:
        -----
        ivt_mask3_mask2_and_arrival_to_scan_on_time - 
            IVT mask 3. Created in
            _create_masks_arrival_to_scan_on_time().
        onset_to_scan_on_time_ivt_bool -
            Whether arrival to scan time for each patient is under 
            the thrombolysis limit. Created in 
            _calculate_onset_to_scan_time().
        mt_mask3_mask2_and_arrival_to_scan_on_time - 
            MT mask 3. Created in
            _create_masks_arrival_to_scan_on_time(). 
        onset_to_scan_on_time_mt_bool -
            Whether arrival to scan time for each patient is under 
            the thrombectomy limit. Created in 
            _calculate_onset_to_scan_time().        
        """
        mask_ivt = (
            (self.trial[
                'ivt_mask3_mask2_and_arrival_to_scan_on_time'].data == 1) &
            (self.trial['onset_to_scan_on_time_ivt_bool'].data == 1)
            )
        mask_mt = (
            (self.trial[
                'mt_mask3_mask2_and_arrival_to_scan_on_time'].data == 1) &
            (self.trial['onset_to_scan_on_time_mt_bool'].data == 1)
            )

        self.trial[
            'ivt_mask4_mask3_and_onset_to_scan_on_time'].data = mask_ivt
        self.trial['mt_mask4_mask3_and_onset_to_scan_on_time'].data = mask_mt

        
    def _create_masks_enough_time_to_treat(self):
        """
        Mask 5: Is there enough time left for threatment?
        
        Creates:
        --------
        ivt_mask5_mask4_and_enough_time_to_treat -
            Mask of whether there is enough time before the 
            thrombolysis limit and whether mask 4 is True 
            for each patient.
        mt_mask5_mask4_and_enough_time_to_treat -
            Mask of whether there is enough time before the 
            thrombectomy limit and whether mask 4 is True 
            for each patient.
        
        Uses:
        -----
        ivt_mask4_mask3_and_onset_to_scan_on_time - 
            IVT mask 4. Created in
            _create_masks_onset_to_scan_on_time().
        enough_time_for_ivt_bool -
            Whether there is enough time left before the thrombolysis
            limit. Created in 
            _calculate_time_left_for_ivt_after_scan().
        mt_mask4_mask3_and_onset_to_scan_on_time - 
            MT mask 4. Created in
            _create_masks_onset_to_scan_on_time(). 
        enough_time_for_mt_bool -
            Whether arrival to scan time for each patient is under 
            the thrombectomy limit. Created in 
            _calculate_time_left_for_mt_after_scan().
        """
        mask_ivt = (
            (self.trial[
                'ivt_mask4_mask3_and_onset_to_scan_on_time'].data == 1) &
            (self.trial['enough_time_for_ivt_bool'].data == 1)
            )
        mask_mt = (
            (self.trial[
                'mt_mask4_mask3_and_onset_to_scan_on_time'].data == 1) &
            (self.trial['enough_time_for_mt_bool'].data == 1)
            )

        self.trial['ivt_mask5_mask4_and_enough_time_to_treat'].data = mask_ivt
        self.trial['mt_mask5_mask4_and_enough_time_to_treat'].data = mask_mt

        
    def _create_masks_treatment_given(self):
        """
        Mask 6: Was treatment given?
                
        Creates:
        --------
        ivt_mask6_mask5_and_treated -
            Mask of whether each patient received thrombolysis
            and whether mask 5 is True for each patient.
        mt_mask6_mask5_and_treated -
            Mask of whether each patient received thrombectomy
            and whether mask 5 is True for each patient.
        
        Uses:
        -----
        ivt_mask5_mask4_and_enough_time_to_treat - 
            IVT mask 5. Created in
            _create_masks_enough_time_to_treat().
        ivt_chosen_bool -
            Whether there is enough time left before the thrombolysis
            limit. Created in 
            _generate_whether_ivt_chosen_binomial().
        mt_mask5_mask4_and_enough_time_to_treat - 
            MT mask 5. Created in _create_masks_enough_time_to_treat().
        mt_chosen_bool -
            Whether arrival to scan time for each patient is under 
            the thrombectomy limit. Created in 
            _generate_whether_mt_chosen_binomial().
        """
        mask_ivt = (
            (self.trial[
                'ivt_mask5_mask4_and_enough_time_to_treat'].data == 1) &
            (self.trial['ivt_chosen_bool'].data == 1)
            )
        mask_mt = (
            (self.trial[
                'mt_mask5_mask4_and_enough_time_to_treat'].data == 1) &
            (self.trial['mt_chosen_bool'].data == 1)
            )

        self.trial['ivt_mask6_mask5_and_treated'].data = mask_ivt
        self.trial['mt_mask6_mask5_and_treated'].data = mask_mt



    # #########################
    # ##### SANITY CHECKS #####
    # #########################
    def _run_sanity_check_on_hospital_data(self, hospital_data: dict):
        """
        Sanity check the hospital data.

        Check all of the relevant keys exist and that the data is
        of the right dtype.
        
        Inputs:
        -------
        hospital_data - a dictionary containing the keywords in the 
                        following "keys" list. Each of the keys points
                        to an array because the expected hospital_data
                        format is a single labelled row of a pandas 
                        DataFrame.
        """
        keys = [
            'admissions',
            'proportion1_of_all_with_onset_known_ivt',
            'proportion_scan_to_needle_on_time',
            'proportion6_of_mask5_with_treated_ivt',
            'proportion2_of_mask1_with_onset_to_arrival_on_time_mt',
            'proportion3_of_mask2_with_arrival_to_scan_on_time_mt',
            'proportion_scan_to_puncture_on_time',
            'proportion6_of_mask5_with_treated_mt',
            'proportion_of_mt_with_ivt',
            'lognorm_mu_onset_arrival_mins_ivt',
            'lognorm_sigma_onset_arrival_mins_ivt', 
            'lognorm_mu_arrival_scan_arrival_mins_ivt', 
            'lognorm_sigma_arrival_scan_arrival_mins_ivt', 
            'lognorm_mu_scan_needle_mins',
            'lognorm_sigma_scan_needle_mins', 
            'lognorm_mu_onset_arrival_mins_mt', 
            'lognorm_sigma_onset_arrival_mins_mt', 
            'lognorm_mu_arrival_scan_arrival_mins_mt', 
            'lognorm_sigma_arrival_scan_arrival_mins_mt',
            'lognorm_mu_scan_puncture_mins',
            'lognorm_sigma_scan_puncture_mins', 
            ]
        expected_dtypes = [['float']] * len(keys)

        success = True
        for k, key in enumerate(keys):
            # Does this key exist?
            try:
                value_here = hospital_data[key]
                # Is this value an array?
                if type(value_here) in [np.ndarray]:
                    # Are those values of the expected data type?
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
                    else:
                        pass  # The data is OK.
                else:
                    pass  # Skip the data type check.
            except KeyError:
                print(f'{key} is missing from the hospital data.')
                success = False
        if success is False:
            error_str = 'The input hospital data needs fixing.'
            raise ValueError(error_str) from None
        else:
            pass  # All of the data is ok.


    def _fudge_patients_after_time_limit(
            self,
            patient_times_mins: float,
            proportion_within_limit: float,
            time_limit_mins: float
            ):
        """
        Make sure the proportion of patients with arrival time
        below X hours matches the hospital performance data.
        Set a few patients now to have arrival times above
        X hours.
        
        Inputs:
        -------
        patient_times_mins      - array. One time per patient.
        proportion_within_limit - float. How many of those times should
                                  be under the time limit?
        time_limit_mins         - float. The time limit we're comparing
                                  these patient times with.
                                
        Returns:
        --------
        patient_times_mins - array. The input time array but with some
                             times changed to be past the limit so that
                             the proportion within the limit is met.
        """
        # How many patients are currently arriving after the limit?
        inds_times_after_limit = np.where(
            patient_times_mins > time_limit_mins)
        n_times_after_limit = len(inds_times_after_limit[0])
        # How many patients should arrive after the limit?
        expected_times_after_limit = int(
            len(patient_times_mins) * (1.0 - proportion_within_limit)
            )
        # How many extra patients should we alter the values for?
        n_times_to_fudge = expected_times_after_limit - n_times_after_limit
        
        if n_times_to_fudge > 0:
            # Randomly pick this many patients out of those who are
            # currently arriving within the limit.
            inds_times_to_fudge = np.random.choice(
                np.where(patient_times_mins <= time_limit_mins)[0],
                size=n_times_to_fudge,
                replace=False
                )
            # Set these patients to be beyond the time limit.
            patient_times_mins[inds_times_to_fudge] = time_limit_mins + 1
        return patient_times_mins



    def _sanity_check_masked_patient_proportions(
            self, leeway: float=0.25):
        """
        Check if generated proportions match the targets.
        
        Compare proportions of patients passing each mask with
        the target proportions (e.g. from real hospital performance 
        data).
        
        Inputs:
        -------
        leeway - float. How far away the generated proportion can be
                 from the target proportion without raising a warning.
                 Set this to 0.0 for no difference allowed (not 
                 recommended!) or >1.0 to guarantee no warnings.
        """
        target_proportions = [
            'proportion_known_arrival_on_time_',
            'proportion_arrival_to_scan_on_time_',
            'proportion_onset_to_scan_on_time_',
            'proportion_enough_time_to_treat_',
            'proportion_treated_'
        ]
        mask_names = [
            '_mask1_onset_known',
            '_mask2_mask1_and_onset_to_arrival_on_time',
            '_mask3_mask2_and_arrival_to_scan_on_time',
            '_mask4_mask3_and_onset_to_scan_on_time',
            '_mask5_mask4_and_enough_time_to_treat',
            '_mask6_mask5_and_treated'
        ]
        for i, treatment in enumerate(['ivt', 'mt']):
            for j, proportion_name in enumerate(target_proportions):
                try:
                    target_proportion = self.target_data_dict[
                        proportion_name + treatment]
                    success = True
                except KeyError:
                    # This proportion hasn't been given by the user.
                    success = False
                    
                if success is False:
                    pass  # Don't perform any checks.
                else:
                    mask_before = self.trial[treatment + mask_names[j]].data
                    mask_now = self.trial[treatment + mask_names[j+1]].data

                    # Create patient proportions from generated (g) data.
                    # Proportion is Yes to Mask now / Yes to Mask before.
                    g_proportion = (np.sum(mask_now) / np.sum(mask_before)
                                    if np.sum(mask_before) > 0 else np.NaN)

                    # If there's a problem, print this label:
                    label_to_print = proportion_name.replace('_', ' ')
                    label_to_print = label_to_print.split('proportion ')[1] 
                    label_to_print += treatment

                    # Compare with target proportion:
                    self._check_proportion(
                        g_proportion,
                        target_proportion,
                        label=label_to_print,
                        leeway=leeway
                        )


    def _check_proportion(
            self,
            prop_current: float,
            prop_target: float,
            label: str='',
            leeway: float=0.1
            ):
        """
        Check whether generated proportion is close to the target.
        
        If the proportion is more than (leeway*100)% off the target,
        raise a warning message.
        
        Inputs:
        prop_current - float. Calculated proportion between 0.0 and 1.0.
        prop_target  - float. Target proportion between 0.0 and 1.0.
        label        - str. Label to print if there's a problem.
        leeway       - float. How far away the calculated proportion is
                       allowed to be from the target.
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
            patient_times: np.ndarray,
            mu_target: float,
            sigma_target: float,
            label: str=''
            ):
        """
        Check whether generated times follow target distribution.
        
        Raise warning if:
        - the new mu is outside the old mu +/- old sigma, or
        - the new sigma is considerably larger than old sigma.
        
        Inputs:
        -------
        patient_times - np.ndarray. The distribution to check.
        mu_target     - float. mu for the target distribution.
        sigma_target  - float. sigma for the target distribution.
        label         - str. Label to print if there's a problem.
        """
        # Set all zero or negative values to something tiny here
        # to prevent RuntimeWarning about division by zero
        # encountered in log.
        patient_times = np.clip(patient_times, a_min=1e-7, a_max=None)
        
        # Generated distribution statistics.
        mu_generated = np.mean(np.log(patient_times))
        sigma_generated = np.std(np.log(patient_times))

        # Check new mu:
        if abs(mu_target - mu_generated) > sigma_target:
            print(''.join([
                f'Warning: the log-normal "{label}" distribution ',
                'has a mean outside the target mean plus or minus ',
                'one standard deviation.'
            ]))
        else:
            pass

        # Check new sigma:
        if sigma_target > 5*sigma_generated:
            print(''.join([
                f'Warning: the log-normal "{label}" distribution ',
                'has a standard deviation at least five times as large ',
                'as the target standard deviation.'
            ]))
        else:
            pass
