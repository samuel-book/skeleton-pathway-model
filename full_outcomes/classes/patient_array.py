import numpy as np

class Patient_array:
    def __init__(
            self, 
            number_of_patients,
            valid_dtypes,
            valid_min=None,
            valid_max=None,
            name_str=None
        ):
        """Set up the patient array."""
        self._number_of_patients = number_of_patients
        self._valid_dtypes = valid_dtypes
        
        # Optional min and max allowed values of the array:
        self._valid_min = valid_min
        self._valid_max = valid_max
        
        # Optional name for an instance of this class:
        self._name = name_str
        
        # Initially create a data array with dummy data.
        if valid_min is not None:
            val = valid_min
        else:
            val = 0
        data = np.full(number_of_patients, val, dtype=valid_dtypes[0])
        self.data = data

    def __str__(self):
        """Prints info when print(Instance) is called."""
        print_str = '\n'.join([
            'Patient array:',
            f'  _number_of_patients = {self._number_of_patients}',
            f'  _valid_dtypes = {self._valid_dtypes}',
            ])
        att_labels = ['_valid_min', '_valid_max', '_name']
        atts = [self._valid_min, self._valid_max, self._name]
        for i, att in enumerate(atts):
            if att is not None:
                print_str += '\n  '
                print_str += att_labels[i] + ' = ' + f'{att}'
        return print_str
    
    def __repr__(self):
        """Prints how to reproduce this instance of the class."""
        return ''.join([
            f'Patient_array(',
            f'number_of_patients={self._number_of_patients}, ',
            f'valid_dtypes={self._valid_dtypes}, ',
            f'valid_min={self._valid_min}, ',
            f'valid_max={self._valid_max}, ',
            f'name_str={self._name}',
            f')'
            ])
    
    def __setattr__(self, key, value):
        """
        Set attribute to the given value.
        """
        if key[0] != '_':
            self.run_sanity_checks(value)
            # If sanity checks fail, an exception is raised.
        self.__dict__[key] = value
        
    def __delattr__(self, key):
        """
        Set attribute to None (setup attrs) or default array (result).
        """
        if key[0] != '_':
            # Return patient array to default values.
            # Select default value:
            if self._valid_min is not None:
                default_val = self._valid_min
            else:
                default_val = 0
            self.__dict__[key] = np.full(
                self._number_of_patients,
                default_val,
                dtype=np.dtype(self._valid_dtypes[0])
            )
        else:
            # Change setup value to None.
            self.__dict__[key] = None
        

    def run_sanity_checks(self, arr):
        """
        Check consistency of input data array with the setup values.
        
        Don't raise exceptions as this function goes along to ensure
        that all of the error messages are flagged up on the first
        run through.
        """
        # Sanity checks flag. Change this to False if any checks fail:
        sanity_checks_passed = True

        # Are all values the right dtype?
        if arr.dtype not in [
                np.dtype(valid_dtype) for valid_dtype in self._valid_dtypes
                ]:
            print(''.join([
                'All values in the array must be the same dtype. ',
                f'Available types are: {self._valid_dtypes}.'
                ]))
            sanity_checks_passed = False

        # Are all values within the allowed range?
        if self._valid_min is not None and self._valid_max is not None:
            if np.all(
                    (arr >= self._valid_min) & (arr <= self._valid_max)
                    ) == False:
                print('Some values are outside the allowed range.')
                sanity_checks_passed = False

        # Is the array one-dimensional?
        if len(arr.shape) > 1:
            print(''.join([
                f'Flattening the input array from shape {arr.shape} to ',
                f'shape {arr.ravel().shape}.'
                ]))
            arr = arr.ravel()
            # This doesn't fail the sanity check.

        # Does the array contain the right number of patients?
        if len(arr) != self._number_of_patients:
            print(''.join([
                f'This array contains {len(arr)} values ',
                f'but the expected number of patients is ',
                f'{self._number_of_patients}. ',
                f'Please update the arrays to be the same length.'
                ]))
            sanity_checks_passed = False
            
        # If any of the checks failed, raise exception now.
        if sanity_checks_passed is False:
            failed_str = 'Sanity checks failed. Values not updated.'
            raise ValueError(failed_str) from None
            # ^ this syntax prevents longer message printing.