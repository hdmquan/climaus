import os
import re
import time
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
ee.Initialize(project=os.getenv("GOOGLE_PROJECT_ID"))

DYNAMIC_WORLD_COLLECTION = "GOOGLE/DYNAMICWORLD/V1"
EXPORT_SCALE = 10
EXPORT_CRS = "EPSG:4326"
MAX_PIXELS = 1e13


def create_export_task(image: ee.Image, year: int) -> ee.batch.Task:
    filename = f"dw_label_{year}"

    task = ee.batch.Export.image.toDrive(
        image=image,
        description=filename,
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

    return collection.mode().clip(ee.Geometry.Rectangle(AUSTRALIA_BBOX))


def export_annual_dynamic_world(years: List[int]):
    for year in years:
        logger.info(f"Processing year {year}...")
        image = get_dynamic_world_label_image(year)
        task = create_export_task(image, year)
        task.start()
        logger.info(f"Export started for year {year}, waiting for it to finish...")

        while True:
            status = task.status()["state"]
            if status in ("COMPLETED", "FAILED", "CANCELLED"):
                logger.info(f"Year {year}: export task finished with state: {status}")
                break
            logger.info(f"Year {year}: current task status = {status}")
            time.sleep(30)


def wait_for_tasks(tasks: List[ee.batch.Task], poll_interval: int = 30):
    logger.info("Waiting for all Earth Engine export tasks to complete...")
    while True:
        statuses = [task.status()["state"] for task in tasks]
        if all(state in ("COMPLETED", "FAILED", "CANCELLED") for state in statuses):
            break
        logger.info(f"Task statuses: {statuses}")
        time.sleep(poll_interval)

    for task, state in zip(tasks, statuses):
        logger.info(f"Task {task.config['description']} finished with state: {state}")


def authenticate_drive() -> GoogleDrive:
    gauth = GoogleAuth()
    gauth.LoadClientConfigFile(str(PATH.CRED / "google_oauth_client_secret.json"))
    gauth.LocalWebserverAuth()
    return GoogleDrive(gauth)


def download_exports_from_drive(local_path: Path):
    local_path.mkdir(parents=True, exist_ok=True)
    drive = authenticate_drive()

    logger.info(f"Searching for exported Dynamic World files in Drive root...")

    file_list = drive.ListFile({"q": "trashed=false"}).GetList()

    dw_files = [f for f in file_list if re.match(r"dw_label_\d{4}.*\.tif", f["title"])]

    if not dw_files:
        logger.warning("No matching Dynamic World .tif files found.")
        return

    for file in dw_files:
        file_title = file["title"]
        dest_path = local_path / file_title

        if dest_path.exists():
            logger.info(f"Skipped (already exists): {file_title}")
            continue

        logger.info(f"Downloading {file_title} → {dest_path}")
        file.GetContentFile(str(dest_path))

    logger.success(f"All Dynamic World GeoTIFFs downloaded to {local_path}")

    # Optional: delete files after download
    for file in dw_files:
        logger.info(f"Deleting file from Drive: {file['title']}")
        file.Delete()

    logger.success("All downloaded files deleted from Drive.")


if __name__ == "__main__":

    logger.info("Starting Dynamic World export tasks to Google Drive root...")

    export_annual_dynamic_world(list(range(2014, 2024)))

    logger.success("All export tasks completed.")

    download_exports_from_drive(PATH.R_DATA / "deforestation")

    logger.warning("These will pile up in Google Drive. Go to:")
    logger.warning("https://drive.google.com/drive/my-drive")
    logger.warning("and delete manually if necessary.")
