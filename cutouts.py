from astropy.nddata import Cutout2D
from astropy.coordinates import SkyCoord
from astropy.table import Table
from astropy.wcs import WCS
from astropy.wcs.utils import proj_plane_pixel_scales
from astropy.io import fits
import os
import numpy as np
from itertools import combinations
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import torch
from torchvision.ops import box_iou
import glob, re
import torchvision.transforms.functional as TF

ENCODING_VERSION = "sigma3_rot"      # bump: cache now holds rotated cutouts

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
                 reuse_cutouts=True, rotations=(0,)):
        self.mosaic_id = mosaic_id
        self.data_root = data_root
        self.size = size
        self.max_neighbours = max_neighbours
        self.reuse_cutouts = reuse_cutouts
        self.rotations = tuple(rotations)      # (0,) for val/test

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
        self.cutout_dir = os.path.join(
            r, DIR_CUTOUT, f"{m}_{ENCODING_VERSION}_{self.size}")
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
        self.beam_arcsec = float(f[0].header["BMAJ"]) * 3600.0
        
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
                
        # shape-column units -> pixels ---
        self.shape_scale = self._shape_scale(self.raw_catalogue)

        med_px = float(np.nanmedian(np.asarray(self.raw_catalogue["Maj"], float))) * self.shape_scale
        print(f"[{self.mosaic_id}] pixel={self.pixel_arcsec:.2f}\"  median Maj={med_px:.1f} px "
              f"(cutout {self.size} px = {self.size * self.pixel_arcsec:.0f}\")")
        if med_px > self.size / 2:
            raise ValueError(
                f"[{self.mosaic_id}] median source is {med_px:.0f} px across but the "
                f"cutout is only {self.size} px — check the Maj/Min units")

        self.n_truth_lost = 0

    
    def _rotate_point(self, x, y, angle_deg):
        """Rotate about the cutout centre, matching scipy.ndimage.rotate."""
        c = (self.size - 1) / 2.0
        th = np.radians(-angle_deg)
        cos, sin = np.cos(th), np.sin(th)
        dx, dy = np.asarray(x, float) - c, np.asarray(y, float) - c
        return cos * dx - sin * dy + c, sin * dx + cos * dy + c


    def _rotate_image(self, img, angle_deg):
        if angle_deg % 360 == 0:
            return img
        t = torch.from_numpy(np.ascontiguousarray(img))          # (C,H,W)
        r = TF.rotate(t, float(angle_deg),
                      interpolation=TF.InterpolationMode.BILINEAR,
                      expand=False, fill=0.0)
        return r.numpy().astype(np.float32)


    def _box_from_members(self, rows, cutout_xy, angle_deg):
        """Tight box around a set of components after rotation.

        rows      : table rows (for Maj/Min/PA)
        cutout_xy : (n,2) their positions in UNROTATED cutout coords
        """
        xs, ys = [], []
        rx, ry = self._rotate_point(cutout_xy[:, 0], cutout_xy[:, 1], angle_deg)
        for row, x, y in zip(rows, np.atleast_1d(rx), np.atleast_1d(ry)):
            # the ellipse rotates with the image: add angle to its PA
            theta = np.radians(90.0 - (row["PA"] + angle_deg))
            a = row["Maj"] * self.shape_scale
            b = row["Min"] * self.shape_scale
            dx = np.hypot(a * np.cos(theta), b * np.sin(theta))
            dy = np.hypot(a * np.sin(theta), b * np.cos(theta))
            xs += [x - dx, x + dx]
            ys += [y - dy, y + dy]
        xs, ys = np.array(xs), np.array(ys)
        return [float(max(xs.min(), 0.0)), float(max(ys.min(), 0.0)),
                float(min(xs.max(), self.size)), float(min(ys.max(), self.size))]
                        

    def sort_by_proximity(self, centre_gauss, catalogue):
        """Returns the neighbour catalogue ordered by pixel distance from centre_gauss, nearest first."""
        if len(catalogue) == 0:
            return catalogue
        
        d2 = ((np.asarray(catalogue["Xposn"], float) - centre_gauss["Xposn"]) ** 2 +
            (np.asarray(catalogue["Yposn"], float) - centre_gauss["Yposn"]) ** 2)
        return catalogue[np.argsort(d2)]
 
    
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
    
    
    def _filter_unresolved(self, table):
        """Drop compact, likely-unrelated neighbours (Mostert's GBC proxy)."""
        if len(table) == 0:
            return table
        maj = np.asarray(table["Maj"], float)              # arcsec
        mnr = np.maximum(np.asarray(table["Min"], float), 1e-6)
        keep = (maj > 1.5 * self.beam_arcsec) | (maj / mnr > 1.5)
        return table[keep]


    def _gt_member_indices(self, centre_gauss, nb):
        """Indices into [centre] + list(nb) of the true association.
        Returns [] if the centre has no entry in the component catalogue."""
        key = str(centre_gauss["Source_Name"])
        va_name = self.source_of.get(key)
        if va_name is None:
            return []
        member_keys = set(self.members_of.get(va_name, []))
        if not member_keys:
            return []
        idx = [0]                                          # centre always
        for i, r in enumerate(nb):
            if str(r["Source_Name"]) in member_keys:
                idx.append(i + 1)
                
        n_true = len(member_keys)
        if n_true > len(idx):
            self.n_truth_lost += 1
            
        return idx


    def _make_cutout_array(self, cutout, centre_gauss):
        """(3,H,W) sigma-encoded cutout, or None if unusable."""
        rms_jy = float(centre_gauss["Isl_rms"]) * 1e-3     # mJy/beam -> Jy/beam
        d = np.nan_to_num(cutout.data, nan=0.0, posinf=0.0, neginf=0.0)
        if not np.isfinite(d).any() or rms_jy <= 0 or np.ptp(d) == 0:
            return None
        s = d / rms_jy
        ch0 = np.sqrt(np.clip((s - 1.0) / 29.0, 0.0, 1.0))
        ch1 = (s > 3.0).astype(np.float32)
        ch2 = (s > 5.0).astype(np.float32)
        return np.stack([ch0, ch1, ch2]).astype(np.float32)
       
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
 
            # neighbours, sorted + filtered exactly as generate_proposals does
            in_bounds = ((np.abs(raw_x - pixel_pos[0]) < half) &
                         (np.abs(raw_y - pixel_pos[1]) < half) &
                         (raw_key != key))
            neighbours = self._filter_unresolved(raw[in_bounds])
            nb = self.sort_by_proximity(centre_gauss, neighbours)[:self.max_neighbours]
            n_dropped = max(0, len(neighbours) - self.max_neighbours)

            # component list: index 0 = centre, then neighbours
            members = [centre_gauss] + [r for r in nb]
            mem_xy = np.array(
                [[pixel_pos[0] - origin[0], pixel_pos[1] - origin[1]]] +
                [[float(r["Xposn"]) - origin[0], float(r["Yposn"]) - origin[1]]
                 for r in nb], dtype=float)

            # true association, as indices into `members`
            gt_idx = self._gt_member_indices(centre_gauss, nb)

            base_img = self._make_cutout_array(cutout, centre_gauss)
            if base_img is None:

                n_skipped += 1
                continue

            for angle in self.rotations:
                img = self._rotate_image(base_img, angle)
                fp = os.path.join(self.cutout_dir, f"{_safe(key)}_r{angle}.npy")
                if not (self.reuse_cutouts and os.path.exists(fp)):
                    np.save(fp, img)

                proposals = []
                for k in range(len(nb) + 1):
                    for combo in combinations(range(1, len(nb) + 1), k):
                        sel = (0,) + combo
                        proposals.append(self._box_from_members(
                            [members[j] for j in sel], mem_xy[list(sel)], angle))
                proposals = np.asarray(proposals, dtype=np.float32)

                if gt_idx:
                    gt_box = np.array([self._box_from_members(
                        [members[j] for j in gt_idx],
                        mem_xy[list(gt_idx)], angle)], dtype=np.float32)
                    gt_label = np.array([2 if len(gt_idx) > 1 else 1], np.int64)
                else:
                    gt_box = np.empty((0, 4), np.float32)
                    gt_label = np.empty((0,), np.int64)
                    if angle == self.rotations[0]:
                        n_nogt += 1

                nb_xy_rot = np.stack(self._rotate_point(
                    mem_xy[1:, 0], mem_xy[1:, 1], angle), axis=1) \
                    if len(nb) else np.empty((0, 2))

                samples.append({
                    "mosaic_id": self.mosaic_id,
                    "source_name": key,
                    "angle": angle,
                    "cutout_path": fp,
                    "proposals": proposals,
                    "gt_box": gt_box,
                    "gt_label": gt_label,
                    "neighbour_xy": nb_xy_rot.astype(np.float32),
                    "n_dropped": n_dropped,
                })
 
        if verbose:
            total = len(self.centre_catalogue)
            print(f"[{self.mosaic_id}] {len(samples)}/{total} samples "
                  f"({n_skipped} edge-skipped, {n_nogt} without GT, "
                  f"{n_truncated} neighbour-truncated, "
                  f"{self.n_deblended} deblended dropped, "
                  f"{self.n_unjoined} comp rows unjoined)"
                  f"{self.n_truth_lost/len(samples)} = fraction of true siblings being filtered out")
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
        