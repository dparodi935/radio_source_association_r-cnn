from astropy.nddata import Cutout2D
from astropy.coordinates import SkyCoord
from astropy.table import Table
from astropy.wcs import WCS
from astropy.wcs.utils import proj_plane_pixel_scales
from astropy.io import fits
from astropy.visualization import ImageNormalize, SinhStretch, ZScaleInterval
import os
import numpy as np
from itertools import combinations
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import torch
from torchvision.ops import box_iou


class SamplesPreprocessor():
    def __init__(self):        
        #choose facet and cutout size
        self.fits_filename = "facet_5"
        self.size = 192
        self.max_neighbours = 10 #limit number of proposals generated
        
        self.create_file_paths()
        self.extract_data()
        self.open_catalogues()
        
        """ PyBDSF proposals: RA,DEC,Maj, and Min are all in degrees
            ELAIS-N1 Catalogue: Also all in degrees
        """
        
    
    def create_file_paths(self):
        #create filepaths
        catalogue_filename = f"{self.fits_filename}_sources" 
        gt_catalogue_filename = f"{self.fits_filename}_catalogue_official"
        script_dir = os.path.dirname(os.path.abspath(__file__))
        fits_folder_path = os.path.join(script_dir,"..","lofar_downloads")
        
        self.fits_filepath = os.path.join(fits_folder_path,f"{self.fits_filename}.fits")
        self.catalogue_filepath = os.path.join(fits_folder_path,f"{catalogue_filename}.fits")
        self.gt_catalogue_filepath = os.path.join(fits_folder_path,f"{gt_catalogue_filename}.fits")
        self.cutout_filepath = os.path.join(fits_folder_path, "cutouts") # where to find the cutouts used for training

        self.cutout_test_path = os.path.join(script_dir, "training_cutouts_output") # where to save training cutouts (with boxes draw)

        
    def extract_data(self):
        with fits.open(self.fits_filepath) as fits_file:
            self.data = fits_file[0].data
            
            hdr = fits_file[0].header
            self.beam_maj_deg = hdr["BMAJ"]   # deg
            self.beam_min_deg = hdr["BMIN"]

            print("BEAM")
            print(hdr["BMIN"])

            self.coord_system = WCS(fits_file[0].header).celestial
            
            self.coord_scale = proj_plane_pixel_scales(self.coord_system)[1] #1 is chosen because pixels are square, and to avoid RA related issues
            """Multiplying by the coord scale transforms pixels to degrees
            """

    def open_catalogues(self):
        #open PyBDSF catalogue and create Table
        self.catalogue = Table.read(self.catalogue_filepath, format="fits")
        self.gt_catalogue = Table.read(self.gt_catalogue_filepath, format="fits")
        
        # Normalize all Right Ascensions to strictly fall between 0 and 360 degrees
        self.catalogue["RA"] = self.catalogue["RA"] % 360
        self.gt_catalogue["RA"] = self.gt_catalogue["RA"] % 360
        

    def ellipse_extent(self, gauss):
        """Returns the greatest extent of the ellipse in the x and y direction

        Args:
            gauss (dict): Ellipse 

        Returns:
            tuple (x_extent, y_extent): 
        """
        coord_scale = self.coord_scale
        
        clean_dc_maj = np.nan_to_num(gauss["DC_Maj"], nan=0.0, neginf=0.0)      
        clean_dc_min = np.nan_to_num(gauss["DC_Min"], nan=0.0, neginf=0.0)        
  
        maj_deg = np.sqrt(clean_dc_maj**2 + self.beam_maj_deg**2)
        min_deg = np.sqrt(clean_dc_min**2 + self.beam_min_deg**2)
                
        theta = np.radians(90 - gauss["DC_PA"]) 
                
        x_extent = np.sqrt(((maj_deg/coord_scale) * np.cos(theta))**2  +
                        ((min_deg/coord_scale) * np.sin(theta))**2)
        y_extent = np.sqrt(((maj_deg/coord_scale) * np.sin(theta))**2  +
                        ((min_deg/coord_scale) * np.cos(theta))**2)
        
        return x_extent, y_extent

    def return_angular_distance(self, obj_1, obj_2):
        """Return distance between two points in terms of degrees

        Args:
            obj_1 (dict): _description_
            obj_2 (dict): _description_

        Returns:
            ang_distance (float): Distance in units of degrees
        """
        delta_ra = (obj_2["RA"] - obj_1["RA"] + 180) % 360 - 180
        delta_dec = obj_2["DEC"] - obj_1["DEC"]

        cos_dec = np.cos(np.radians(obj_1["DEC"]))
        
        delta_x = delta_ra * cos_dec
        delta_y = delta_dec
       
        ang_distance = np.sqrt(delta_x**2 + delta_y**2)
        
        return ang_distance

    def return_ellipse_displacement(self, gauss_1, gauss_2):
        """Returns the displacement in pixels between the centre of two ellipses

        Args:
            gauss_1 (dict): The origin ellipse
            gauss_2 (dict): The new ellipse to which the distance is calculated

        Returns:
            tuple (delta_x, delta_y): Displacement vector in pixels from ellipse 1 to ellipse 2
        """
        
        if "Xposn" in gauss_2:
            delta_x = gauss_2["Xposn"] - gauss_1["Xposn"]
            delta_y = gauss_2["Yposn"] - gauss_1["Yposn"]
        else:
            #this code is for when the catalogue distances needs to be translated into pixels
            delta_ra = gauss_2["RA"] - gauss_1["RA"]
            delta_dec = gauss_2["DEC"] - gauss_1["DEC"]
    
            cos_dec = np.cos(np.radians(gauss_1["DEC"]))
            
            # 3. Convert angular degrees to pixels
            # (RA increases to the left/East in standard FITS, so we invert delta_x)
            delta_x = -(delta_ra * cos_dec) / self.coord_scale
            delta_y = delta_dec / self.coord_scale

        return delta_x, delta_y
        
        
    def return_bounding_box(self, centre, centre_gauss, gauss_list=None, scale=1.0):
        """Returns the coordinates of the bottom left and top right corner of the tightly bounding box around an arbitrary number of 
        ellipses 

        Args:
            centre (tuple[float,float]): Coordinates (x,y) of the central ellipse in terms of the cutout's axes
            centre_gauss (dict): Central ellipse
            gauss_list (list[dict], optional): List of other ellipses being bounded
            scale (float): Manually scale the size of the bounding boxes

        Returns:
            list[float]: [x1,y1,x2,y2] The coordinates of the bottom left and top right of the bounding box in pixels
        """    

        #extent of central ellipse
        dx1, dy1 = self.ellipse_extent(centre_gauss)
        dx1, dy1 = dx1 * scale, dy1 * scale

        #bounding box around central ellipse
        x1 = centre[0] - dx1
        y1 = centre[1] - dy1
        x2 = centre[0] + dx1
        y2 = centre[1] + dy1
        
        if gauss_list is None:
            gauss_list = []
    
        if len(gauss_list) > 0:
            #cycles through other ellipses, figuring out where one extends beyond the bounding box and then adjusting the box
            
            for gauss in gauss_list:
                #(delta_x, delta_y) is the displacement of the centre of the new ellipse relative to the central one
                delta_x, delta_y = self.return_ellipse_displacement(centre_gauss, gauss)

                dx2, dy2 = self.ellipse_extent(gauss)
                dx2, dy2 = dx2 * scale, dy2 * scale

                x_pos_2 = centre[0] + delta_x
                y_pos_2 = centre[1] + delta_y
                
                #adjusts box if necessary
                x1 = min(x1, x_pos_2 - dx2)
                y1 = min(y1, y_pos_2 - dy2)
                x2 = max(x2, x_pos_2 + dx2)
                y2 = max(y2, y_pos_2 + dy2)
        
        #prevent clipping outside of box
        h, w = self.size, self.size   # or cutout.data.shape for safety
        x1 = max(x1, 0.0)
        y1 = max(y1, 0.0)
        x2 = min(x2, float(w))
        y2 = min(y2, float(h))
        
        return [x1,y1,x2,y2]


    def sort_by_proximity(self, centre_gauss, catalogue):
        """Returns the neighbour catalogue ordered by pixel distance from centre_gauss, nearest first."""
        if len(catalogue) == 0:
            return catalogue
        
        d2 = ((np.asarray(catalogue["Xposn"], float) - centre_gauss["Xposn"]) ** 2 +
            (np.asarray(catalogue["Yposn"], float) - centre_gauss["Yposn"]) ** 2)
        return catalogue[np.argsort(d2)]


    def generate_proposals(self, centre, centre_gauss, catalogue):
        """Generates the region proposals for the R-CNN by finding the bounding box of every combination of ellipse that includes the central one
           Number of proposals limited by a maximum number of neighbours considered
        Args:
            centre (tuple[float,float]): Coordinates of the centre of ellipse 1 in terms of the cutout's axes
            centre_gauss (dict): The ellipse that the cutout is centred on (and that is being classified by the R-CNN)
            catalogue (list[dict]): List of other ellipses within the cutout

        Returns:
            ndarray: List of region proposals, each given by the coordinates of the bottom left and top right corner in pixels [x1,y1,x2,y2]
        """
        proposals = [] 
        sorted_cat = self.sort_by_proximity(centre_gauss, catalogue)[:self.max_neighbours]
                
        for i in range(len(sorted_cat)+1):
            for combo in combinations(sorted_cat,i):
                combo_list = [combo[i] for i in range(len(combo))]
                proposals.append(self.return_bounding_box(centre, centre_gauss, gauss_list=combo_list))
        
        return np.array(proposals)
            
            
    def return_closest_gt_gauss(self, centre_gauss, gt_catalogue):
        """Finds, if possible, the closest gaussian from the ground truth catalogue. Accounts for astrometric errors and the size of sources

        Args:
            centre_gauss (dict): The central ellipse
            proximity_gt_catalogue (list[dict]): List of ellipses in the ground truth catalogue that are within the cutout

        Returns:
            dict: The closest ellipse in the ground truth catalogue
        """        
        #if there is no source from the ground truth catalogue within the cutout
        if len(gt_catalogue) == 0:
            return None

        normalised_distances = self.return_normalised_distances(centre_gauss, gt_catalogue)
        
        closest_gt_gauss = gt_catalogue[np.argmin(normalised_distances)]
        
        return closest_gt_gauss


    def number_of_sibling_sources(self, parent_name, catalogue):
        """Checks nearby sources in the ground truth catalogue and returns how many siblings the source has
           ie. is source part of multi-component source

        Args:
            parent_name (str): _description_
            catalogue (list[dict]): _description_

        Returns:
            int: Number of siblings
        """
        
        sibling_list = catalogue[catalogue["Parent_Source"] == parent_name]
        return len(sibling_list)
    

    def generate_gt(self, centre_gauss, proximity_gt_catalogue, cutout):
        """Returns the ground truth boxes and labels (0: background, 1: single, 2:MC) by finding the closest source in the ground truth catalogue

        Args:
            centre_gauss (dict): _description_
            proximity_gt_catalogue (list[dict]): _description_
            cutout (Cutout2D): _description_

        Returns:
            (ndarray, int): [x1,y1,x2,y2], label
        """
        
        closest_gt = self.return_closest_gt_gauss(centre_gauss, proximity_gt_catalogue) 
        
        if closest_gt is None: return np.empty((0, 4), dtype=np.float32), np.empty((0,), dtype=np.int64)

        #finds the position of the ground truth source in the cutout in terms of pixels
        astronomical_pos = SkyCoord(closest_gt["RA"], closest_gt["DEC"], unit="deg", frame="icrs")
        pixel_pos = self.coord_system.world_to_pixel(astronomical_pos)
        gt_source_centre = cutout.to_cutout_position(pixel_pos)
        
        #finds the other components of a multi-component source if it exists. Gives the list of siblings and the label
        parent_name = closest_gt["Parent_Source"]
        gt_siblings_list = proximity_gt_catalogue[proximity_gt_catalogue["Parent_Source"] == parent_name]
        num = self.number_of_sibling_sources(parent_name, proximity_gt_catalogue)
        if num == 1 : label=1
        elif num > 1: label=2
        
        #finds the bounding box around the ground truth source
        box = np.array([self.return_bounding_box(gt_source_centre, closest_gt, gauss_list=gt_siblings_list, scale=1/1.25)], dtype=np.float32)  # shape (1, 4)

        return box, label

    def return_normalised_distances(self, gauss, catalogue):
        distances = self.return_angular_distance(gauss, catalogue)
        clean_dc_maj = np.nan_to_num(catalogue["DC_Maj"], nan=0.0, neginf=0.0)
        dynamic_radii = np.maximum(self.beam_maj_deg, clean_dc_maj/2)
        normalised_distances = distances/dynamic_radii
        
        return normalised_distances

    
    def _in_image_bounds(self, pixel_pos):
        """Returns True if a full (size x size) cutout centred on pixel_pos
        would fit entirely within the image, i.e. doesn't need trimming/padding.
        """
        ny, nx = self.data.shape
        half = self.size / 2.0
        x, y = pixel_pos
        return (x - half >= 0) and (x + half <= nx) and (y - half >= 0) and (y + half <= ny)


    def source_filter(self, gauss, gt_catalogue):  
        """Filter out sources from training or test. Return true if source is to be filtered out

        Args:
            gauss (dict): _description_
            gt_catalogue (list [dict]): _description_

        Returns:
            bool: True if source is filtered out
        """
        # Require a peak SNR greater than 5
        p_n_ratio = gauss["Peak_flux"] / gauss["Isl_rms"]
        if p_n_ratio >= 5:
            pass
            #return False
        else:
            return True
        
        """lets a big neighbour steal matches from a genuine compact counterpart directly under the PyBDSF position. 
        Cheap guard: if any GT lies within ~1 beam in absolute distance, prefer the nearest of those; 
                    only fall back to normalised ranking otherwise."""
        
        #Filter out sources with no closeby sources from the ground truth catalogue
        limit = 1 #1 would be requiring every Pybdsf source to be within or aligned with a gt source. A value >1 adds some buffer
        normalised_distances = self.return_normalised_distances(gauss, gt_catalogue)
        
        min_norm_distance = np.min(normalised_distances)

        if min_norm_distance > limit:
            return True
        
        return False    
    
    
    def process_raw_data(self, cutout_data):
        """Clean and scale the data

        Args:
            cutout_data (ndarray): _description_

        Returns:
            ndarray: _description_
        """
        # Define cutout_data by extracting the raw array and cleaning initial NaNs
        #nan, posinf, neginf=0 means to set any Nans or infinities to zero
        #nan_to_num is the general function used to replace nans with number
        cutout_data = np.nan_to_num(cutout_data, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Scale and normalize the data. "clip=True" constrains the data to [0,1]
        norm = ImageNormalize(
            cutout_data, 
            interval=ZScaleInterval(), 
            stretch=SinhStretch(a=0.1), 
            clip=True
        )
        normalized = norm(cutout_data)
        
        #Converts np masked array into a regular numpy array. Any masked (hidden by the boolean astropy mask) values replaced with 0.0
        scaled_data = np.ma.filled(normalized, fill_value=0.0)
        
        # Cast to float32  (for performance), then clean up any NaNs/Infs from this downcast
        scaled_data = scaled_data.astype(np.float32)
        scaled_data = np.nan_to_num(scaled_data, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Clamp extreme outliers from the normalised data to a sane range
        scaled_data = np.clip(scaled_data, -10.0, 10.0)

        return scaled_data
    
    
    def generate_samples_list(self, verbose=True):

        """Generates the samples list, one row for each source

        Args:
            verbose (bool, optional): Debugging measure. Defaults to True.

        Returns:
            list[dict]: Each row corresponds to one source and includes the filepath to cutout, a list of proposed regions,
            the ground truth box and the label of that ground truth
        """
        catalogue = self.catalogue
        gt_catalogue = self.gt_catalogue
        
        samples = []
        skipped_ids = []
        
        half_size_deg = (self.size / 2.0) * self.coord_scale
        
        #cycles through every source 
        for idx, gauss in enumerate(catalogue):
            
            if self.source_filter(gauss, gt_catalogue): continue
            
            
            pixel_pos = (gauss["Xposn"], gauss["Yposn"])
            
            # Skip sources too close to the field edge for a full-size cutout,
            # rather than padding/trimming and risking distorted proposals or GT boxes.
            if not self._in_image_bounds(pixel_pos):
                skipped_ids.append(gauss["Gaus_id"])
                continue
            
            #creates the filepath to the cutout
            gaus_id = gauss["Gaus_id"]
            cutout_filename = f"{self.fits_filename}_cutout_{gaus_id}.npy"
            filepath = os.path.join(self.cutout_filepath, cutout_filename)
            
            # Extract the cutout using astropy
            cutout = Cutout2D(self.data, pixel_pos, self.size)
            
            scaled_data = self.process_raw_data(cutout.data)
            
            # Save the cleaned array to disk
            np.save(filepath, scaled_data)       
            
            source_centre = cutout.to_cutout_position(pixel_pos) 
            
            # Creates mask for catalogued sources that are within the cutout
            in_bounds = ((abs(catalogue["Xposn"] - gauss["Xposn"]) < self.size / 2) &
                        (abs(catalogue["Yposn"] - gauss["Yposn"]) < self.size / 2) &
                        (catalogue["Gaus_id"] != gauss["Gaus_id"]))
            
            #The modulus and sums of 180 are to account for the values wrapping around from 360 to 0
            delta_ra = (gt_catalogue["RA"] - gauss["RA"] + 180) % 360 - 180
                        
            cos_dec = np.cos(np.radians(gauss["DEC"]))
            
            # Creates mask for ground truth sources that are within the cutout
            in_bounds_gt = (
                (abs((delta_ra) * cos_dec) < half_size_deg) &
                (abs(gt_catalogue["DEC"] - gauss["DEC"]) < half_size_deg)
            )
                        
            proximity_catalogue = catalogue[in_bounds]
            proximity_gt_catalogue  = gt_catalogue[in_bounds_gt]
                            
            proposals = self.generate_proposals(source_centre, gauss, proximity_catalogue)
            gt_box, gt_label = self.generate_gt(gauss, proximity_gt_catalogue, cutout)
            
            sample = {"cutout_path":filepath, "proposals":proposals, "gt_box":gt_box, "gt_label":gt_label}
            samples.append(sample)      
            
            if verbose and (idx % 10 == 0):
                self.visualize_cutout(scaled_data, proposals, gt_box, gaus_id, gt_label)
 
        #Debugging
        if verbose:
            total = len(catalogue)
            final_number = len(samples)
            n_skipped = len(skipped_ids)
            print(
                f"[{self.fits_filename}] generated {len(samples)}/{total} samples "
                f"({n_skipped} skipped for being too close to the field edge, "
                f"{n_skipped / total:.1%})"
            )
            if n_skipped:
                print(f"[{self.fits_filename}] skipped Gaus_ids: {list(skipped_ids)}")
            
            filter_dropped = total-n_skipped-final_number    
            print(f"Number of sources filtered out: {filter_dropped}")
 
        return samples
    
    def _calculate_iou(self, boxA, boxB):
        boxA_tensor = torch.tensor(np.array([boxA]))
        boxB_tensor = torch.tensor(np.array([boxB]))
        return box_iou(boxA_tensor, boxB_tensor)[0][0].item()
    
    def visualize_cutout(self, image_data, proposals, gt_boxes, gaus_id, gt_label):
        """Helper method to plot the image, proposal boxes, and ground truth boxes."""
        fig, ax = plt.subplots(figsize=(8, 8))
        
        # Display the image (origin='lower' keeps standard astronomical orientation)
        ax.imshow(image_data, cmap="gray", origin="lower")
        
        # Determine the primary Ground Truth box for IoU calculation (if one exists)
        primary_gt_box = None
        if gt_boxes is not None and len(gt_boxes) > 0:
            primary_gt_box = gt_boxes[0] if gt_boxes.ndim == 2 else gt_boxes
        
        # 1. Plot PyBDSF Proposals (Cyan, Dashed, Thinner) and label with IoU
        for box in proposals:
            x1, y1, x2, y2 = box
            rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, 
                                     linewidth=1, edgecolor='cyan', 
                                     facecolor='none', linestyle='--', alpha=0.7)
            ax.add_patch(rect)
            
            # Calculate and plot IoU if a Ground Truth box exists
            iou_text = "IoU: N/A"
            if primary_gt_box is not None:
                iou = self._calculate_iou(box, primary_gt_box)
                iou_text = f"IoU: {iou:.2f}"
            
            # Place IoU text at the top-left of the proposal to avoid clashing with the GT label
            ax.text(x1, y2, iou_text, color='cyan', fontsize=8, va="bottom", ha="left")
            
        # 2. Plot Ground Truth Box (Lime Green, Solid, Thicker)
        if gt_boxes is not None and len(gt_boxes) > 0:
            # Note: gt_boxes might be shape (1, 4) or empty (0, 4)
            if gt_boxes.ndim == 1:
                gt_boxes = [gt_boxes]  # Ensure it's iterable if 1D
            for box in gt_boxes:
                x1, y1, x2, y2 = box
                rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, 
                                         linewidth=2.5, edgecolor='lime', 
                                         facecolor='none')
                ax.add_patch(rect)
                ax.text(x1, y1 - 2, f"GT Label: {gt_label}", 
                        color='lime', fontsize=10, va="top")

        ax.set_title(f"Source ID: {gaus_id} | {len(proposals)} Proposals")
        ax.axis("off")
        filepath = os.path.join(self.cutout_test_path,f"{gaus_id}.png")
        plt.savefig(f"{filepath}")
        plt.close(fig) # Adding this prevents Matplotlib from eating up all your RAM!
        