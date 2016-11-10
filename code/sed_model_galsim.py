#!/usr/bin/env python
# encoding: utf-8
"""
sed_model_galsim.py

Model an SED as a superposition of templates with GalSim
"""
import numpy as np 
import galsim

### Define the possible SED templates in the model
k_SED_names = ['NGC_0695_spec', 'NGC_4125_spec', 'NGC_4552_spec', 'CGCG_049-057_spec']


class SEDModelGalSIm(object):
	"""
	An SED model as a superposition of templates

	Uses GalSim to render through bandpasses to get mags and fluxes
	"""
	def __init__(self, telescope_name="LSST", ref_filter_name='r'):
		self.telescope_name = telescope_name
		self.ref_filter_name = ref_filter_name

		## Initialize model parameters
		## A single redshift parameter 
		self.redshift = 0.0
		## A 'magnitude' parameter to set the amplitude of each SED template
		self.mags = [20.0 for i in xrange(len(k_SED_names))]
	
    def _load_sed_files(self):
        """
        Load SED templates from files.

        Copied from GalSim demo12.py
        """
        path, filename = os.path.split(__file__)
        datapath = os.path.abspath(os.path.join(path, "../dat/seds/"))
        self.SEDs = {}
        for SED_name in jifparams.k_SED_names:
            SED_filename = os.path.join(datapath, '{0}.sed'.format(SED_name))
            self.SEDs[SED_name] = galsim.SED(SED_filename, wave_type='Ang',
                                             flux_type='flambda')
        return None

    def _load_filter_files(self, wavelength_scale=1.0):
        """
        Load filters for drawing chromatic objects.

        @param wavelength_scale     Multiplicative scaling of the wavelengths
                                    input from the filter files to get
                                    nanometers from whatever the input units are
        """
        self.filters = load_filter_files(wavelength_scale, self.telescope_name)

    def get_SED(self):
        """
        Get the GalSim SED object given the SED parameters and redshift.

        This routine passes galsim_galaxy magnitude parameters to the GalSim
        SED.withMagnitude() method.

        The magnitude GalSimGalaxyModel magnitude parameters are defined for redshift zero. If a
        model is requested for a different redshift, then the SED amplitude is set before the
        redshift, resulting in output apparent magnitudes that may not match the input apparent
        magnitude parameter (unless z=0).
        """
        bp = self.filters[ref_filter_name]
        SEDs = [self.SEDs[SED_name].atRedshift(0.).withMagnitude(
            target_magnitude=self.mags[i],
            bandpass=bp).atRedshift(self.redshift)
                for i, SED_name in enumerate(k_SED_names)]
        return reduce(add, SEDs)

   def get_flux(self, filter_name='r'):
        """
        Get the flux of the galaxy model in the named bandpass

        @param filter_name  Name of the bandpass for the desired magnitude

        @returns the flux in the requested bandpass (in photon counts)
        """
        SED = self.get_SED()
        return SED.calculateFlux(self.filters[filter_name])

    def get_magnitude(self, filter_name='r'):
        """
        Get the magnitude of the galaxy model in the named bandpass

        @param filter_name  Name of the bandpass for the desired magnitude

        @returns the magnitude in the requested bandpass
        """
        SED = self.get_SED()
        return SED.calculateMagnitude(self.filters[filter_name])


def load_filter_file_to_bandpass(table, wavelength_scale=1.0,
                                 effective_diameter_meters=6.4,
                                 exptime_sec=30.):
    """
    Create a Galsim.Bandpass object from a lookup table

    @param table Either (1) the name of a file for reading the lookup table
                 values for a bandpass, or (2) an instance of a
                 galsim.LookupTable
    @param wavelength_scale The multiplicative scaling of the wavelengths in the
                            input bandpass file to get units of nm (not used if
                            table argument is a LookupTable instance)
    @param effective_diameter_meters The effective diameter of the telescope
                                     (including obscuration) for the zeropoint
                                     calculation
    @param exptime_sec The exposure time for the zeropoint calculation
    """
    if isinstance(table, str):
        dat = np.loadtxt(table)
        table = galsim.LookupTable(x=dat[:,0]*wavelength_scale, f=dat[:,1])
    elif not isinstance(table, galsim.LookupTable):
        raise ValueError("table must be a file name or galsim.LookupTable")
    bp = galsim.Bandpass(table, wave_type='nm')
    bp = bp.thin(rel_err=1e-4)
    return bp.withZeropoint(zeropoint='AB',
        effective_diameter=effective_diameter_meters,
        exptime=exptime_sec)


def load_filter_files(wavelength_scale=1.0, telescope_name="LSST",
					  filter_names='ugrizy', effective_diameter=6.4,
					  exptime_zeropoint=30.):
    """
    Load filters for drawing chromatic objects.

    Adapted from GalSim demo12.py

    @param wavelength_scale     Multiplicative scaling of the wavelengths
                                input from the filter files to get
                                nanometers from whatever the input units are
    @param telescope_name       Name of the telescope model ("LSST" or "WFIRST")
    """
    if telescope_name == "WFIRST":
        ### Use the Galsim WFIRST module
        filters = galsim.wfirst.getBandpasses(AB_zeropoint=True)
    else:
        ### Use filter information in this module
        path, filename = os.path.split(__file__)
        datapath = os.path.abspath(os.path.join(path, "../dat/seds/"))
        filters = {}
        for filter_name in filter_names:
            filter_filename = os.path.join(datapath, '{}_{}.dat'.format(
                telescope_name, filter_name))
            filters[filter_name] = load_filter_file_to_bandpass(
                filter_filename, wavelength_scale,
                effective_diameter,
                exptime_zeropoint
            )
    return filters

