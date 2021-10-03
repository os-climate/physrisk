import logging, os.path, rasterio, rasterio.sample, requests
import numpy as np
from rasterio.windows import from_bounds

# requires raterio and gdal; for binaries to install on Windows, at time of writing https://pypi.org/project/rasterio/ directs us to https://www.lfd.uci.edu/~gohlke/pythonlibs/#rasterio

class EventProviderWri():
    """Class to obtain World Resources Institute (WRI) hazard data from various sources."""
    
    __return_periods = [5, 10, 25, 50, 100, 250, 500, 1000]
    __wri_public_url = "http://wri-projects.s3.amazonaws.com/AqueductFloodTool/download/v2/"

    def __init__(self, src_key, event_type = "inundation", **kwargs):
        # different sources can be specified
        
        if event_type != "inundation":
            raise NotImplementedError("Sourcing of {0} data not supported".format(event_type))

        # for 'file', data must exist in folder specified
        if src_key == 'file':
            if 'folder' in kwargs:
                self.__get_events = lambda lons, lats, return_periods, scenario, sea_level, type, subsidence, year, cf = kwargs['folder'] : self.__get_inundation_file_based(
                    cf, lats, lons, return_periods, scenario, sea_level, type, subsidence, year)
            else:
                # enforced: otherwise very slow and hits WRI servers frequently
                raise KeyError("folder must be supplied")

        # for 'web', data will be downloaded from WRI public site as per http://wri-projects.s3.amazonaws.com/AqueductFloodTool/download/v2/index.html and cached locally
        if src_key == 'web':
            if 'cache_folder' in kwargs:
                # cache GeoTiffs in folder specified (important for local development)
                download = self.__download_inundation
                self.__get_events = lambda lons, lats, return_periods, scenario, sea_level, type, subsidence, year, cf = kwargs['cache_folder'], d = download : self.__get_inundation_file_based(
                    cf, lats, lons, return_periods, scenario, sea_level, type, subsidence, year, download_flood_data = d)
            else:
                # enforced: otherwise very slow and hits WRI servers frequently
                raise KeyError("cache_folder must be supplied")
        
        # for 'service': do we add ability to invoke a service to return event data?
        else:
            raise NotImplementedError("Source Key : {0} not handled.".format(src_key))

    def get_inundation_depth(self, lons, lats, return_periods = None, scenario = "rcp8p5", sea_level = 0, type = "coast", subsidence = True, year = 2080):
        """Return inundation depths for available return periods.

        Args:
            lats (List[float]): latitudes in degrees
            lons (List[float]): longitudes in degrees
            return_periods (List[int]): list of returns periods in years
            scenario (str): climate scenario ("historical", "rcp4p5" or "rcp8p5")
            sea_level (int): sea level rise as a percentile (0, 5 or 50)
            subsidence (bool): include subsidence ("nosub" or "wtsub")
            type (str): "coast" or "river"
            year (int): 2030, 2050 or 2080
        """
        self.__get_events(lats, lons, EventProviderWri.__return_periods if return_periods is None else return_periods, scenario, sea_level, type, subsidence, year)

    #region inundation

    def __get_inundation_file_based(self, folder, lons, lats, return_periods, scenario, sea_level, type, subsidence, year, download_flood_data = None):
        """Get inundation data by reading GeoTiff files."""
               
        intensities = []
        for period in [5, 10, 25, 50, 100, 250, 500, 1000]:
            filename_stub = self.__get_inundation_file_name_stub(period, scenario, sea_level, type, subsidence, year)
            filename = filename_stub + ".tif"
            path = os.path.join(folder, filename)

            if not os.path.isfile(path):
                if download_flood_data is None:
                    raise KeyError("file {0} not found".format(path))
                else:
                    with open(path, 'wb') as stream:
                        download_flood_data(stream, filename)
            
            with rasterio.open(path) as dataset:
                sg = rasterio.sample.sample_gen(dataset, [[lon, lat] for (lon, lat) in zip(lons, lats)])
                hw = 0.01 # half-width of window in degrees
                win = from_bounds(lons[0] - hw, lats[0] - hw, lons[0] + hw, lats[0] + hw, dataset.transform) # left, bottom, right, top
                intensities.append(list(sg))
                max_intensity = np.max(dataset.read(1, window = win))
        return intensities
 
    def __download_inundation(self, stream, filename):        
        url = EventProviderWri.__wri_public_url + filename       
        logging.info("Downloading file " + filename)      
        # small enough to download and write, but could chunk and stream
        r = requests.get(url, allow_redirects = True)
        stream.write(r.content)
        logging.info("Downloaded")  

    def __get_inundation_file_name_stub(self, return_period, scenario, sea_level, type, with_subsidence, year):
        if type not in ["coast", "river"]:
            raise ValueError("invalid type")

        return "{0}{1}_{2}_{3}_{4}_rp{5:04d}_{6}".format("inun", type, scenario, 
            "wtsub" if with_subsidence else "nosub", year, return_period, "0" if sea_level == 0 else "0_perc_{:02d}".format(sea_level))

    #for converstion to NetCDF
    #from osgeo import gdal
    #ds = gdal.Translate(dir + filename_stub + ".nc", dir + filename_stub + ".tif", format='NetCDF') 

    #endregion