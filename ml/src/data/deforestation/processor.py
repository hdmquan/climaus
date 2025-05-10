import numpy as np
from src.common import setup_logger, PATH
from src.data.deforestation.processor import process_ndvi_tiff

logger = setup_logger()

RAW_DIR = PATH.R_DATA / "deforestation"
PROCESSED_DIR = PATH.P_DATA / "deforestation"

PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


def process_all_ndvi_tiles(tile_size=64, stride=32):
    tifs = list(RAW_DIR.glob("ndvi_*.tif"))

    if not tifs:
        logger.warning(f"No .tif files found in {RAW_DIR}")
        return

    for tif_path in sorted(tifs):
        filename = tif_path.stem  # No .tif
        out_path = PROCESSED_DIR / f"{filename}.npz"

        if out_path.exists():
            logger.info(f"Skipped (already processed): {filename}")
            continue

        logger.info(f"Processing {filename}...")
        tiles = process_ndvi_tiff(tif_path, tile_size=tile_size, stride=stride)

        if not tiles:
            logger.warning(f"No valid tiles extracted from {filename}")
            continue

        tiles_np = np.stack(tiles)  # (N, tile_size, tile_size)

        np.savez_compressed(out_path, tiles=tiles_np)
        logger.success(f"Saved {tiles_np.shape[0]} tiles → {out_path}")


if __name__ == "__main__":
    logger.info("Starting NDVI tile processing pipeline")
    process_all_ndvi_tiles()
    logger.success(f"All NDVI tiles processed and stored in {PROCESSED_DIR}")
