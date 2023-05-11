import numpy as np

class Basic_Outcome:
    """
    What dis?
    
    15: Set baseline probability of good outcome based on age group
    16: Convert baseline probability good outcome to odds
    17: Calculate odds ratio of good outcome based on time to thrombolysis
    18: Patient odds of good outcome if given thrombolysis
    19: Patient probability of good outcome if given thrombolysis
    20: Clip patient probability of good outcome to minimum of zero
    21: Individual patient good outcome if given thrombolysis (boolean)*
    21: Individual patient good outcome if not given thrombolysis (boolean)*

    *Net population outcome is calculated here by summing probabilities of good
    outcome for all patients, rather than using individual outcomes. These columns
    are added for potential future use.

    """

    def __init__(self):
        # Set proportion of good outcomes for under 80 and 80+)
        self.good_outcome_base = (0.3499, 0.1318)


    def calculate_outcomes(
            self,
            onset_to_needle_mins,
            thrombolysis_conditions_met_bool,
            age_80plus_bool
            ):
        """
        Calculate tings
        """
        patients_per_run = len(age_80plus_bool)

        # Set baseline probability good outcome (based on age group)
        patient_array_good_outcome_base_prob = np.full(patients_per_run, 0.0)
        patient_array_good_outcome_base_prob[age_80plus_bool == 0] = \
            self.good_outcome_base[0]
        patient_array_good_outcome_base_prob[age_80plus_bool == 1] = \
            self.good_outcome_base[1]

        # Convert baseline probability to baseline odds
        patient_array_good_outcome_base_odds = (
            patient_array_good_outcome_base_prob /
            (1 - patient_array_good_outcome_base_prob)
            )

        # Calculate odds ratio based on time to treatment
        patient_array_odds_ratio = (
            10 ** (0.326956 + (-0.00086211 * onset_to_needle_mins))
            )

        # Adjust odds of good outcome
        patient_array_good_outcome_adjusted_odds = (
            patient_array_good_outcome_base_odds * patient_array_odds_ratio
            )

        # Convert odds back to probability
        patient_array_good_outcome_adjusted_prob = (
            patient_array_good_outcome_adjusted_odds /
            (1 + patient_array_good_outcome_adjusted_odds)
            )

        # Improved probability of good outcome (calc changed probability
        # then multiply by whether thrombolysis given)
        patient_array_good_outcome_improved_prob_actual = (
            (patient_array_good_outcome_adjusted_prob -
              patient_array_good_outcome_base_prob) *
              thrombolysis_conditions_met_bool
            )

        # remove any negative probabilities calculated
        # (can occur if long treatment windows set)
        patient_array_good_outcome_improved_prob = np.amax(
            [patient_array_good_outcome_improved_prob_actual,
             np.zeros(patients_per_run)],
            axis=0)

        # # Individual good ouctome due to thrombolysis
        # # This is not currently used in the analysis
        # patient_array[:, 21] = np.random.binomial(
        #     1, patient_array_good_outcome_improved_prob, patients_per_run)

        # Individual outcomes if no treatment given
        patient_array_good_outcome_no_treatment_bool = np.random.binomial(
            1, patient_array_good_outcome_base_prob, patients_per_run)

        # Baseline good outcomes per 1000 patients
        self.baseline_good_outcomes_per_1000_patients = (
            (patient_array_good_outcome_no_treatment_bool.sum() /
             patients_per_run) * 1000)

        # Calculate overall expected extra good outcomes
        self.additional_good_outcomes_per_1000_patients = (
            ((patient_array_good_outcome_improved_prob.sum() /
              patients_per_run) * 1000))
