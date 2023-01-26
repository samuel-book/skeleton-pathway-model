import numpy as np
import pandas as pd
import simpy
import time

from classes.globvars import Globvars
from classes.patient import Patient
from classes.pathway import Pathway

class Model(object):
    """
    The main model object.

    class Model():

    Attributes
    ----------
    env:
        SimPy environment
    
    globvars:
        Imported global variables (object)
    
    Methods
    -------
    __init__:
        Constructor class for model

    """

    def __init__(self, scenario):
        """Constructor class for model"""

        # Scenario overwrites default Globvars values
        self.globvars = Globvars(scenario)

        # Set up SimPy environment
        self.env = simpy.Environment()
        
        # Set up pathway
        self.pathway = Pathway(self.env, self.globvars)


    def generate_patient_arrival(self):
        """
        SimPy process. Generate patients. 

        Returns:
        --------

        patient:
            A patient object

        """

        # Continuous loop of patient arrivals
        arrival_count = 0
        while True:
            arrival_count += 1
            # Get patient object
            patient = Patient(self.globvars, arrival_count)
            # Pass patient to pathway
            self.env.process(self.pathway.process_patient(patient))
            # Sample time to next admission from exponential distribution
            time_to_next = np.random.exponential(self.globvars.inter_arrival_time)
            # SimPy delay to next arrival (using environment timeout)
            yield self.env.timeout(time_to_next)

    def run(self):
        """Model run: Initialise processes needed at model start, start model 
        running, and call end_run_routine.
        Note: All SimPy processes must be called with `env.process` in addition
        to the process function/method name"""

        time_start = time.time()

        # Initialise processes that will run on model run. 
        self.env.process(self.generate_patient_arrival())


        # Run
        self.env.run(until=self.globvars.run_duration)

        time_end = time.time()
        time_taken = time_end - time_start
        print (f'Sim time taken: {time_taken:0.0f}')