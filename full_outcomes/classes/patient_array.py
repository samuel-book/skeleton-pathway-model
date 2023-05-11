import numpy as np

class Patient_array:
    def __init__(self, number_of_patients, valid_min, valid_max, valid_dtypes):
        self._valid_min = valid_min
        self._valid_max = valid_max
        self._valid_dtypes = valid_dtypes
        self._number_of_patients = number_of_patients

    def __str__(self):
        """Prints info when print(Instance) is called."""
        return \
            f"""
            Clinical outcome:
                info: thibng
            """
    
    def __repr__(self):
        """Prints how to reproduce this instance of the class."""
        return \
            f"""
            Clinical_outcome(mrs_dists={self.mrs_dists})
            """
    
    def __set_name__(self, owner, name):
        self._name = name

    def __get__(self, instance, owner):
        return instance.__dict__[self._name]

    
    def __set__(self, instance, arr):

        run_sanity_checks(arr)
        # If sanity checks fail, an exception is raised.
        instance.__dict__[self._name] = arr

        
#     def __del__(self, instance, owner): # ???
#         # Return to default value.
#         instance.__dict__[self._name] = np.zeros(
#             self._number_of_patients,
#             dtype=np.dtype(self._valid_dtypes[0])
#         )
        

    def run_sanity_checks(arr):
        """
        Don't raise exceptions as this function goes along to ensure
        that all of the error messages are flagged up on the first
        run through.
        """
        # Sanity checks flag. Change this to False if any checks fail:
        sanity_checks_passed = True

        # Are all values the right dtype?
        if arr.dtype not in [
                dtype(valid_dtype) for valid_dtype in self._valid_dtypes
                ]:
            print(
                f'''
                All values in the array must be the same dtype.
                Available types are: {self._valid_dtypes}.
                '''
                 )
            sanity_checks_passed = False

        # Are all values within the allowed range?
        if np.all((arr >= self._valid_min) & (arr <= self._valid_max)) is False:
            print('Some values are outside the allowed range.')
            sanity_checks_passed = False

        # Is the array one-dimensional?
        if len(arr.shape) > 1:
            print(
                f'''
                Flattening the input array from shape {arr.shape} to
                shape {arr.ravel().shape}.
                '''
            )
            arr = arr.ravel()
            # This doesn't fail the sanity check.

        # Does the array contain the right number of patients?
        if len(arr) != number_of_patients:
            print(
                f'''
                This array contains {len(arr)} values
                but previous arrays have {self._number_of_patients} values.
                Please update the arrays to be the same length.
                '''
            )
            sanity_checks_passed = False
            
        # If any of the checks failed, raise exception now.
        if sanity_checks_passed is False:
            failed_str = 'Sanity checks failed. Values not updated.'
            raise ValueError(failed_str) from None
            # ^ this syntax prevents longer message printing.
            
        # return sanity_checks_passed
