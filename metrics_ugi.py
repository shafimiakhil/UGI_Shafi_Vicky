import os
from pathlib import Path
import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from rasterio.features import rasterize
from sklearn.metrics import roc_auc_score, average_precision_score
from scipy.ndimage import label, binary_dilation, generate_binary_structure

def calculate_geobia_metrics(pred_bin_array, gt_mask_array, boundary_tolerance_px=2):
    """
    Calcula las métricas utilizando un raster de predicción que ya es binario.
    """
    gt_flat = gt_mask_array.flatten()
    pred_flat = pred_bin_array.flatten()
    
    valid_pixels = gt_flat != 255
    gt_valid = gt_flat[valid_pixels]
    pred_valid = pred_flat[valid_pixels]

    # AUROC y AP sobre datos binarios
    auroc = roc_auc_score(gt_valid, pred_valid)
    ap = average_precision_score(gt_valid, pred_valid)

    pred_mask = pred_bin_array > 0
    gt_mask = gt_mask_array > 0

    # 1. IoU
    intersection = np.logical_and(pred_mask, gt_mask).sum()
    union = np.logical_or(pred_mask, gt_mask).sum()
    iou = intersection / union if union > 0 else 0.0

    # 2. Object Recall
    structure = generate_binary_structure(2, 2)
    gt_labeled, gt_num_objects = label(gt_mask, structure=structure)
    objects_found = 0
    if gt_num_objects > 0:
        for i in range(1, gt_num_objects + 1):
            gt_object = (gt_labeled == i)
            if np.logical_and(gt_object, pred_mask).any():
                objects_found += 1
        object_recall = objects_found / gt_num_objects
    else:
        object_recall = 0.0

    # 3. Boundary F-score
    def get_boundaries(mask):
        dilated = binary_dilation(mask)
        return np.logical_xor(dilated, mask)
        
    gt_boundaries = get_boundaries(gt_mask)
    pred_boundaries = get_boundaries(pred_mask)
    
    gt_boundaries_dilated = binary_dilation(gt_boundaries, iterations=boundary_tolerance_px)
    pred_boundaries_dilated = binary_dilation(pred_boundaries, iterations=boundary_tolerance_px)
    
    boundary_tp = np.logical_and(pred_boundaries, gt_boundaries_dilated).sum()
    pred_boundary_pixels = pred_boundaries.sum()
    gt_boundary_pixels = gt_boundaries.sum()
    
    bound_precision = boundary_tp / pred_boundary_pixels if pred_boundary_pixels > 0 else 0.0
    boundary_tp_recall = np.logical_and(gt_boundaries, pred_boundaries_dilated).sum()
    bound_recall = boundary_tp_recall / gt_boundary_pixels if gt_boundary_pixels > 0 else 0.0
    
    boundary_f_score = 2 * (bound_precision * bound_recall) / (bound_precision + bound_recall) if (bound_precision + bound_recall) > 0 else 0.0

    return {
        "AUROC": round(auroc, 4),
        "Average_Precision": round(ap, 4),
        "IoU": round(iou, 4),
        "Object_Recall": round(object_recall, 4),
        "Boundary_F1": round(boundary_f_score, 4)
    }

def save_to_csv(metrics_dict, test_name, csv_path="historial_metricas.csv"):
    """
    Guarda o concatena los resultados en un archivo CSV.
    """
    metrics_dict["Test_ID"] = test_name
    df = pd.DataFrame([metrics_dict])
    
    # Reordenar para que Test_ID sea la primera columna
    cols = ["Test_ID"] + [c for c in df.columns if c != "Test_ID"]
    df = df[cols]
    
    if os.path.exists(csv_path):
        df.to_csv(csv_path, mode='a', header=False, index=False)
    else:
        df.to_csv(csv_path, mode='w', header=True, index=False)
    print(f"Resultados guardados en {csv_path}")

def main():
    # Configuración del test actual
    test_id = "Test_1_UNet_Baseline" # Cambia este nombre en cada ejecución
    data_dir = Path(__file__).resolve().parent
    vector_path = data_dir / "22174_UrbanGreenInfras.gpkg"
    pred_raster_path = data_dir / "ugi_m2_r1_none_predicted.tif"
    
    with rasterio.open(pred_raster_path) as src:
        pred_bin_array = src.read(1)
        meta = src.meta
        transform = src.transform
        shape = (src.height, src.width)
    
    gdf = gpd.read_file(vector_path, layer="22174_urbangreeninfras")
    if gdf.crs != meta['crs']:
        gdf = gdf.to_crs(meta['crs'])
    
    shapes = ((geom, 1) for geom in gdf.geometry)
    gt_mask_array = rasterize(
        shapes=shapes,
        out_shape=shape,
        transform=transform,
        fill=0,
        dtype='uint8'
    )
    
    metrics = calculate_geobia_metrics(pred_bin_array, gt_mask_array, boundary_tolerance_px=2)
    
    for k, v in metrics.items():
        print(f"{k}: {v}")
        
    save_to_csv(metrics, test_id, csv_path=data_dir / "historial_metricas.csv")

if __name__ == "__main__":
    main()