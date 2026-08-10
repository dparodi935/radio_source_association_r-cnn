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
import glob, re


# --- folder names, one per catalogue type. Edit to match your disk layout. ---
DIR_MOSAIC = "mosaics"
DIR_RAW = "pybdsf_raw"        # proposals come from here (all components)
DIR_LARGE = "pybdsf_raw_large"    # cutout centres come from here
DIR_COMP = "final_comp"    # Gaus_id -> Source_Name, the ground-truth grouping
DIR_CUTOUT = "cutouts"


def _safe(name):
    """IAU names contain + . - which are awkward in filenames."""
    return re.sub(r"[^A-Za-z0-9_]", "_", str(name))
 
 
def discover_mosaics(data_root):
    """Returns the sorted RA_DEC ids for every mosaic present on disk."""
    pattern = os.path.join(data_root, DIR_MOSAIC, "mosaic_*.fits")
    ids = []
    for path in glob.glob(pattern): #glob is a module that uses unix style characters to search for all files with the specififed patter
        #re.match checks if the mosaic_*.fits string is at the beginning of the variable path
        m = re.match(r"mosaic_(.+)\.fits$", os.path.basename(path)) 
        if m:
            ids.append(m.group(1))
    return sorted(ids)



class SamplesPreprocessor:
    """Builds training samples for ONE mosaic."""
 
    def __init__(self, mosaic_id, data_root, size=200, max_neighbours=11,
                 reuse_cutouts=True):
        self.mosaic_id = mosaic_id
        self.data_root = data_root
        self.size = size
        self.max_neighbours = max_neighbours
        self.reuse_cutouts = reuse_cutouts
 
        self.create_file_paths()
        self.extract_data()
        self.open_catalogues()
        
        """ WRITE UNITS AND MORE HERE. SHOULD ALL BE IN ARCSEC
        """
         

    def create_file_paths(self):
        r, m = self.data_root, self.mosaic_id
        self.fits_filepath = os.path.join(r, DIR_MOSAIC, f"mosaic_{m}.fits")
        self.raw_cat_path = os.path.join(r, DIR_RAW, f"{m}.fits")
        self.large_cat_path = os.path.join(r, DIR_LARGE, f"{m}.fits")
        self.comp_cat_path = os.path.join(r, DIR_COMP, f"{m}.fits")
        self.cutout_dir = os.path.join(r, DIR_CUTOUT, m)
        os.makedirs(self.cutout_dir, exist_ok=True)
 
        for p in (self.fits_filepath, self.raw_cat_path,
                  self.large_cat_path, self.comp_cat_path):
            if not os.path.exists(p):
                raise FileNotFoundError(f"[{m}] missing: {p}")

        script_dir = os.path.dirname(os.path.abspath(__file__))
        self.cutout_test_path = os.path.join(script_dir, "training_cutouts_output") # where to save training cutouts (with boxes draw)

    def extract_data(self):
        with fits.open(self.fits_filepath) as f:
            self.data = np.squeeze(f[0].data)
            if self.data.ndim != 2:
                raise ValueError(f"expected 2D image, got {self.data.shape}")
            self.coord_system = WCS(f[0].header).celestial
            self.coord_scale = proj_plane_pixel_scales(self.coord_system)[1]  # deg/px
        self.pixel_arcsec = self.coord_scale * 3600.0
    
    def _shape_scale(self, table):
        """Pixels per catalogue shape unit. LoTSS catalogues use arcsec; PyBDSF's
        own FITS writer uses degrees. Decide from TUNIT, fall back on magnitude."""
        unit = getattr(table["Maj"], "unit", None)
        if unit is not None:
            u = str(unit).lower()
            if u.startswith("deg"):
                return 1.0 / self.coord_scale
            if u.startswith("arcsec") or u == "s":
                return 1.0 / self.pixel_arcsec
        med = float(np.nanmedian(np.asarray(table["Maj"], float)))
        # a source is never 0.1 deg across, and never 1e-4 arcsec
        return 1.0 / self.coord_scale if med < 0.1 else 1.0 / self.pixel_arcsec    
        
    def open_catalogues(self):
        self.raw_catalogue = Table.read(self.raw_cat_path, format="fits")
        self.centre_catalogue = Table.read(self.large_cat_path, format="fits")
        
        #brightness cut
        sel = (np.asarray(self.centre_catalogue["Total_flux"], float) > 10.0)
        self.centre_catalogue = self.centre_catalogue[sel]

        comp = Table.read(self.comp_cat_path, format="fits")
 
        # NOTE the inversion: in the component catalogue,
        #   Component_Name == the raw PyBDSF catalogue's Source_Name
        #   Source_Name    == the merged value-added source (our grouping label)
        comp_names = np.asarray(comp["Component_Name"]).astype(str)
        va_names = np.asarray(comp["Source_Name"]).astype(str)
 
        # exclude deblended entries: one component split into several sources
        if "Deblended_from" in comp.colnames:
            deb = np.asarray(comp["Deblended_from"]).astype(str)
            keep = (deb == "") | (deb == "--") | (deb == "nan")
            self.n_deblended = int((~keep).sum())
            comp_names, va_names = comp_names[keep], va_names[keep]
        else:
            self.n_deblended = 0
 
        self.source_of = dict(zip(comp_names.tolist(), va_names.tolist()))
 
        self.members_of = {}
        for cn, va in zip(comp_names.tolist(), va_names.tolist()):
            self.members_of.setdefault(va, []).append(cn)
 
        # row lookup into the raw catalogue, keyed by its Source_Name
        raw_names = np.asarray(self.raw_catalogue["Source_Name"]).astype(str)
        self.row_of_key = {n: i for i, n in enumerate(raw_names)}
 
        self.n_unjoined = sum(1 for n in comp_names if n not in self.row_of_key)

        # DR1/DR2 catalogues carry RA/DEC but no pixel columns -> derive them
        for tbl in (self.raw_catalogue, self.centre_catalogue):
            if "Xposn" not in tbl.colnames:
                x, y = self.coord_system.all_world2pix(
                    np.asarray(tbl["RA"], float), np.asarray(tbl["DEC"], float), 0)
                tbl["Xposn"] = x
                tbl["Yposn"] = y

        # --- NEW: shape-column units -> pixels ---
        self.shape_scale = self._shape_scale(self.raw_catalogue)

        med_px = float(np.nanmedian(np.asarray(self.raw_catalogue["Maj"], float))) * self.shape_scale
        print(f"[{self.mosaic_id}] pixel={self.pixel_arcsec:.2f}\"  median Maj={med_px:.1f} px "
              f"(cutout {self.size} px = {self.size * self.pixel_arcsec:.0f}\")")
        if med_px > self.size / 2:
            raise ValueError(
                f"[{self.mosaic_id}] median source is {med_px:.0f} px across but the "
                f"cutout is only {self.size} px — check the Maj/Min units")

        
    def ellipse_extent(self, gauss):
        theta = np.radians(90.0 - gauss["PA"])
        a = gauss["Maj"] * self.shape_scale     # -> pixels
        b = gauss["Min"] * self.shape_scale
        x_extent = np.hypot(a * np.cos(theta), b * np.sin(theta))
        y_extent = np.hypot(a * np.sin(theta), b * np.cos(theta))
        return x_extent, y_extent
        

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
        

    def return_bounding_box(self, centre, gauss_1, gauss_list=None):
        """Tight box (in cutout pixel coords) around gauss_1 plus any others.
 
        centre: cutout-frame position of gauss_1.
        Others are placed by their pixel offset from gauss_1.
        """
        dx1, dy1 = self.ellipse_extent(gauss_1)
        x1, y1 = centre[0] - dx1, centre[1] - dy1
        x2, y2 = centre[0] + dx1, centre[1] + dy1
 
        if gauss_list is None:
            gauss_list = []
 
        for gauss in gauss_list:
            dx2, dy2 = self.ellipse_extent(gauss)
            gx = centre[0] + (gauss["Xposn"] - gauss_1["Xposn"])
            gy = centre[1] + (gauss["Yposn"] - gauss_1["Yposn"])
            x1 = min(x1, gx - dx2)
            y1 = min(y1, gy - dy2)
            x2 = max(x2, gx + dx2)
            y2 = max(y2, gy + dy2)
 
        # clip to the cutout, in ABSOLUTE cutout coords (not relative to centre)
        x1 = max(x1, 0.0)
        y1 = max(y1, 0.0)
        x2 = min(x2, float(self.size))
        y2 = min(y2, float(self.size))
        return [x1, y1, x2, y2]


    def sort_by_proximity(self, centre_gauss, catalogue):
        """Returns the neighbour catalogue ordered by pixel distance from centre_gauss, nearest first."""
        if len(catalogue) == 0:
            return catalogue
        
        d2 = ((np.asarray(catalogue["Xposn"], float) - centre_gauss["Xposn"]) ** 2 +
            (np.asarray(catalogue["Yposn"], float) - centre_gauss["Yposn"]) ** 2)
        return catalogue[np.argsort(d2)]


    def generate_proposals(self, centre, centre_gauss, neighbours):
        """Generates the region proposals for the R-CNN by finding the bounding box of every combination of ellipse that includes the central one
            Number of proposals limited by a maximum number of neighbours considered
        Args:
            centre (tuple[float,float]): Coordinates of the centre of ellipse 1 in terms of the cutout's axes
            centre_gauss (dict): The ellipse that the cutout is centred on (and that is being classified by the R-CNN)
            catalogue (list[dict]): List of other ellipses within the cutout

        Returns:
            ndarray: List of region proposals, each given by the coordinates of the bottom left and top right corner in pixels [x1,y1,x2,y2]
        """
        """Every subset of neighbours, unioned with the centre component."""
        # in generate_proposals, before sorting
        #beam_px = self.bmaj / self.coord_scale        # read BMAJ in extract_data
        #keep = (np.asarray(nb["Maj"], float) * self.shape_scale > 1.5 * beam_px)
        #nb = nb[keep]
        
        nb = self.sort_by_proximity(centre_gauss, neighbours)[:self.max_neighbours]
        n_dropped = max(0, len(neighbours) - self.max_neighbours)
 
        proposals = []
        for k in range(len(nb) + 1):
            for combo in combinations(range(len(nb)), k):
                members = [nb[i] for i in combo]
                proposals.append(
                    self.return_bounding_box(centre, centre_gauss, gauss_list=members))
 
        return np.asarray(proposals, dtype=np.float32), n_dropped           
           
    def generate_gt(self, centre_gauss, cutout_origin, centre_pix):
        """ONE box: the union over all components sharing the centre's Source_Name.
 
        Returns (1,4) float32 and (1,) int64, or empty arrays if unavailable.
        """
        empty = (np.empty((0, 4), np.float32), np.empty((0,), np.int64))
 
        key = str(centre_gauss["Source_Name"])
        va_name = self.source_of.get(key)
        if va_name is None:
            return empty                      # centre absent from component cat
 
        member_keys = self.members_of.get(va_name, [])
        rows = [self.row_of_key[k] for k in member_keys if k in self.row_of_key]
        if not rows:
            return empty
        members = self.raw_catalogue[rows]
 
        x1 = y1 = np.inf
        x2 = y2 = -np.inf
        for m in members:
            # member position in cutout coords, via the cutout's own origin
            cx = m["Xposn"] - cutout_origin[0]
            cy = m["Yposn"] - cutout_origin[1]
            bx = self.return_bounding_box((cx, cy), m)
            x1, y1 = min(x1, bx[0]), min(y1, bx[1])
            x2, y2 = max(x2, bx[2]), max(y2, bx[3])
 
        if not (x2 > x1 and y2 > y1):
            return empty                      # union clipped away entirely
 
        label = 2 if len(member_keys) > 1 else 1
        return (np.array([[x1, y1, x2, y2]], np.float32),
                np.array([label], np.int64))
 
    
    def _in_image_bounds(self, pixel_pos):
        """Returns True if a full (size x size) cutout centred on pixel_pos
        would fit entirely within the image, i.e. doesn't need trimming/padding.
        """
        ny, nx = self.data.shape
        half = self.size / 2.0
        x, y = pixel_pos
        return (x - half >= 0) and (x + half <= nx) and (y - half >= 0) and (y + half <= ny)

    def _write_cutout(self, cutout, filepath, rms_jy):
        if self.reuse_cutouts and os.path.exists(filepath):
            return True
        d = np.nan_to_num(cutout.data, nan=0.0, posinf=0.0, neginf=0.0)
        if not np.isfinite(d).any() or rms_jy <= 0:
            return False
        s = d / rms_jy                                   # image in sigma units
        ch0 = np.sqrt(np.clip((s - 1.0) / 29.0, 0.0, 1.0))   # 1-30 sigma
        ch1 = (s > 3.0).astype(np.float32)
        ch2 = (s > 5.0).astype(np.float32)
        np.save(filepath, np.stack([ch0, ch1, ch2]).astype(np.float32))
        return True
        
    def generate_samples_list(self, verbose=True):
        raw = self.raw_catalogue
        samples = []
        n_skipped = n_nogt = n_truncated = 0
 
        raw_x = np.asarray(raw["Xposn"], float)
        raw_y = np.asarray(raw["Yposn"], float)
        raw_key = np.asarray(raw["Source_Name"]).astype(str)
        half = self.size / 2.0
 
        for centre_gauss in self.centre_catalogue:
            pixel_pos = (float(centre_gauss["Xposn"]), float(centre_gauss["Yposn"]))
            if not self._in_image_bounds(pixel_pos):
                n_skipped += 1
                continue
 
            key = str(centre_gauss["Source_Name"])
            filepath = os.path.join(self.cutout_dir, f"{_safe(key)}.npy")
 
            cutout = Cutout2D(self.data, pixel_pos, self.size)
            self._write_cutout(cutout, filepath, rms_jy = float(centre_gauss["Isl_rms"]) * 1e-3)
 
            origin = (cutout.origin_original[0], cutout.origin_original[1])
            source_centre = (pixel_pos[0] - origin[0], pixel_pos[1] - origin[1])
 
            in_bounds = ((np.abs(raw_x - pixel_pos[0]) < half) &
                         (np.abs(raw_y - pixel_pos[1]) < half) &
                         (raw_key != key))
            neighbours = raw[in_bounds]
 
            proposals, n_dropped = self.generate_proposals(
                source_centre, centre_gauss, neighbours)
            if n_dropped:
                n_truncated += 1
 
            gt_box, gt_label = self.generate_gt(centre_gauss, origin, source_centre)
            if len(gt_box) == 0:
                n_nogt += 1
 
            samples.append({
                "mosaic_id": self.mosaic_id,
                "source_name": key,
                "cutout_path": filepath,
                "proposals": proposals,
                "gt_box": gt_box,
                "gt_label": gt_label,
                "n_dropped": n_dropped,
            })
 
        if verbose:
            total = len(self.centre_catalogue)
            print(f"[{self.mosaic_id}] {len(samples)}/{total} samples "
                  f"({n_skipped} edge-skipped, {n_nogt} without GT, "
                  f"{n_truncated} neighbour-truncated, "
                  f"{self.n_deblended} deblended dropped, "
                  f"{self.n_unjoined} comp rows unjoined)")
        return samples
    
        
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
        