# This model will be simplistic and data shouldn't shift too much
# Hence the very manual download of data, gomen :/

import os
import re
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List

import ee
from dotenv import load_dotenv
from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive

from src.common import setup_logger, PATH
from src.data.metadata.constants import AUSTRALIA_BBOX

logger = setup_logger()

load_dotenv()
ee.Initialize()

MODIS_COLLECTION = "MODIS/061/MOD13Q1"
EXPORT_SCALE = 250
EXPORT_CRS = "EPSG:4326"
MAX_PIXELS = 1e13


def create_export_task(
    image: ee.Image, year: int, month: int, folder: str
) -> ee.batch.Task:

    date_str = f"{year}_{month:02d}" if month else f"{year}"
    filename = f"ndvi_{date_str}"

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


def get_clean_ndvi_image(year: int, month: int = 1) -> ee.Image:
    # Get NDVI image filtered by month/year
    start = f"{year}-{month:02d}-01"
    end_date = datetime(year, month, 1) + timedelta(days=31)
    end = end_date.strftime("%Y-%m-%d")

    collection = (
        ee.ImageCollection(MODIS_COLLECTION)
        .filterDate(start, end)
        .filterBounds(ee.Geometry.Rectangle(AUSTRALIA_BBOX))
        .select(["NDVI", "SummaryQA"])
        .sort("system:time_start")
    )

    image = collection.first()
    return image.updateMask(image.select("SummaryQA").eq(0)).select("NDVI")


def export_annual_data(years: List[int], folder: str):
    for year in years:
        image = get_clean_ndvi_image(year)
        task = create_export_task(image, year, month=0, folder=folder)
        task.start()
        logger.info(f"Export started for year {year}")


def export_monthly_data(year: int, months: List[int], folder: str):
    for month in months:
        image = get_clean_ndvi_image(year, month)
        task = create_export_task(image, year=year, month=month, folder=folder)
        task.start()
        logger.info(f"Export started for 2024-{month:02d}")


def authenticate_drive() -> GoogleDrive:
    # NOTE: Because of this, it should be run locally :(
    gauth = GoogleAuth()
    gauth.LocalWebserverAuth()
    return GoogleDrive(gauth)


def download_exports_from_drive(folder_name: str, local_path: Path):
    local_path.mkdir(parents=True, exist_ok=True)
    drive = authenticate_drive()

    logger.info(f"Searching for exported NDVI files in Drive folder '{folder_name}'...")

    file_list = drive.ListFile(
        {"q": f"'{folder_name}' in parents and trashed=false"}
    ).GetList()

    ndvi_files = [
        f for f in file_list if re.match(r"ndvi_\d{4}(_\d{2})?\.tif", f["title"])
    ]

    if not ndvi_files:
        logger.warning("No matching NDVI .tif files found.")
        return

    for file in ndvi_files:
        file_title = file["title"]
        dest_path = local_path / file_title

        if dest_path.exists():
            logger.info(f"Skipped (already exists): {file_title}")
            continue

        logger.info(f"⬇️ Downloading {file_title} → {dest_path}")
        file.GetContentFile(str(dest_path))

    logger.success(f"All NDVI GeoTIFFs downloaded to {local_path}")


if __name__ == "__main__":

    folder = os.getenv("GOOGLE_DRIVE_FOLDER_NAME")
    drive_url = os.getenv("GOOGLE_DRIVE_URL")

    logger.info("Starting NDVI export tasks to Google Drive")

    # 1 image every years for 10 years
    export_annual_data(list(range(2014, 2024)), folder)
    # 1 image every month for 2024
    export_monthly_data(2024, list(range(1, 13)), folder)

    logger.success("All export tasks launched.")

    download_exports_from_drive(folder, PATH.R_DATA / "deforestation")

    logger.warning("WARNING: These will pile up in Google Drive. Go to:")
    logger.warning(str(drive_url))
    logger.warning(
        "and delete manually when done. I hate Google but MODIS is a bit complex :("
    )
