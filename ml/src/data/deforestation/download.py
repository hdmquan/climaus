import os
import re
from datetime import datetime, timedelta
from pathlib import Path
from typing import List

import ee
from dotenv import load_dotenv
from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive

from src.common import setup_logger, PATH
from src.data.metadata.constants import AUSTRALIA_BBOX

logger = setup_logger()

load_dotenv()
ee.Initialize()

DYNAMIC_WORLD_COLLECTION = "GOOGLE/DYNAMICWORLD/V1"
EXPORT_SCALE = 10
EXPORT_CRS = "EPSG:4326"
MAX_PIXELS = 1e13


def create_export_task(image: ee.Image, year: int, folder: str) -> ee.batch.Task:
    filename = f"dw_label_{year}"

    task = ee.batch.Export.image.toDrive(
        image=image,
        description=filename,
        folder=folder,
        fileNamePrefix=filename,
        region=AUSTRALIA_BBOX,
        scale=EXPORT_SCALE,
        crs=EXPORT_CRS,
        maxPixels=MAX_PIXELS,
    )
    return task


def get_dynamic_world_label_image(year: int) -> ee.Image:
    start = f"{year}-01-01"
    end = f"{year}-12-31"

    collection = (
        ee.ImageCollection(DYNAMIC_WORLD_COLLECTION)
        .filterDate(start, end)
        .filterBounds(ee.Geometry.Rectangle(AUSTRALIA_BBOX))
        .select("label")
    )

    # Take the most frequent class per pixel across the year
    return collection.mode().clip(ee.Geometry.Rectangle(AUSTRALIA_BBOX))


def export_annual_dynamic_world(years: List[int], folder: str):
    for year in years:
        image = get_dynamic_world_label_image(year)
        task = create_export_task(image, year, folder)
        task.start()
        logger.info(f"Export started for year {year}")


def authenticate_drive() -> GoogleDrive:
    # This auths locally — run in interactive machine
    gauth = GoogleAuth()
    gauth.LocalWebserverAuth()
    return GoogleDrive(gauth)


def download_exports_from_drive(folder_name: str, local_path: Path):
    local_path.mkdir(parents=True, exist_ok=True)
    drive = authenticate_drive()

    logger.info(
        f"Searching for exported Dynamic World files in Drive folder '{folder_name}'..."
    )

    file_list = drive.ListFile(
        {"q": f"'{folder_name}' in parents and trashed=false"}
    ).GetList()

    dw_files = [f for f in file_list if re.match(r"dw_label_\d{4}\.tif", f["title"])]

    if not dw_files:
        logger.warning("No matching Dynamic World .tif files found.")
        return

    for file in dw_files:
        file_title = file["title"]
        dest_path = local_path / file_title

        if dest_path.exists():
            logger.info(f"Skipped (already exists): {file_title}")
            continue

        logger.info(f"⬇️ Downloading {file_title} → {dest_path}")
        file.GetContentFile(str(dest_path))

    logger.success(f"All Dynamic World GeoTIFFs downloaded to {local_path}")


if __name__ == "__main__":

    folder = os.getenv("GOOGLE_DRIVE_FOLDER_NAME")
    drive_url = os.getenv("GOOGLE_DRIVE_URL")

    logger.info("Starting Dynamic World export tasks to Google Drive...")

    export_annual_dynamic_world(list(range(2014, 2024)), folder)

    logger.success("All export tasks launched.")

    download_exports_from_drive(folder, PATH.R_DATA / "deforestation")

    logger.warning("These will pile up in Google Drive. Go to:")
    logger.warning(str(drive_url))
    logger.warning("and delete manually when done. Sorry. Google is Google.")
