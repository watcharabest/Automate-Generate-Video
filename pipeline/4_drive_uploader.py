import csv
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional

from dotenv import load_dotenv
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload

load_dotenv()

ROOT_DIR = Path(__file__).resolve().parent.parent
STORY_PLAN_PATH = ROOT_DIR / "output" / "story_plan.json"
FINAL_DIR = ROOT_DIR / "output" / "final"
DAILY_LOG_CSV = ROOT_DIR / "output" / "daily_log.csv"

SCOPES = [
    "https://www.googleapis.com/auth/drive.file",
    "https://www.googleapis.com/auth/spreadsheets",
]


def get_credentials() -> service_account.Credentials:
    """Load Google Service Account credentials from GOOGLE_CREDENTIALS_JSON env var."""
    raw = os.getenv("GOOGLE_CREDENTIALS_JSON", "").strip()
    if not raw:
        raise RuntimeError("GOOGLE_CREDENTIALS_JSON environment variable is not set.")

    if raw.startswith("{"):
        info = json.loads(raw)
        return service_account.Credentials.from_service_account_info(info, scopes=SCOPES)

    cred_path = Path(raw)
    if cred_path.exists():
        return service_account.Credentials.from_service_account_file(
            str(cred_path), scopes=SCOPES
        )
    raise RuntimeError("GOOGLE_CREDENTIALS_JSON is neither valid JSON nor an existing file path.")


def create_drive_folder(drive_service: Any, name: str, parent_id: Optional[str]) -> str:
    metadata: Dict[str, Any] = {
        "name": name,
        "mimeType": "application/vnd.google-apps.folder",
    }
    if parent_id:
        metadata["parents"] = [parent_id]
    folder = drive_service.files().create(body=metadata, fields="id, webViewLink").execute()
    return folder["id"]


def upload_mp4(drive_service: Any, local_path: Path, drive_name: str, parent_id: Optional[str]) -> str:
    metadata: Dict[str, Any] = {"name": drive_name}
    if parent_id:
        metadata["parents"] = [parent_id]
    media = MediaFileUpload(str(local_path), mimetype="video/mp4", resumable=True)
    uploaded = (
        drive_service.files()
        .create(body=metadata, media_body=media, fields="id, webViewLink")
        .execute()
    )
    return uploaded.get("webViewLink", f"https://drive.google.com/file/d/{uploaded['id']}/view")


def append_daily_log(
    creds: Optional[service_account.Credentials],
    date_str: str,
    topic: str,
    part_count: int,
    drive_link: str,
    status: str,
) -> None:
    """Append production log row to Google Sheet, or fallback to output/daily_log.csv."""
    sheet_id = os.getenv("GOOGLE_SHEET_ID", "").strip()
    row = [date_str, topic, part_count, drive_link, status]

    if sheet_id and creds is not None:
        try:
            sheets_service = build("sheets", "v4", credentials=creds)
            sheets_service.spreadsheets().values().append(
                spreadsheetId=sheet_id,
                range="A1",
                valueInputOption="USER_ENTERED",
                body={"values": [row]},
            ).execute()
            print(f"[Uploader] Appended log row to Google Sheet {sheet_id}")
            return
        except Exception as exc:
            print(f"[Uploader] Google Sheet append failed ({exc}); falling back to CSV.")

    DAILY_LOG_CSV.parent.mkdir(parents=True, exist_ok=True)
    write_header = not DAILY_LOG_CSV.exists()
    with DAILY_LOG_CSV.open("a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(["Date", "Topic", "Part Count", "Drive Link", "Status"])
        writer.writerow(row)
    print(f"[Uploader] Appended log row to {DAILY_LOG_CSV}")


def upload_story_outputs() -> str:
    """Upload rendered MP4s to Google Drive following [dd-mm-yy] naming rules."""
    if not STORY_PLAN_PATH.exists():
        raise FileNotFoundError(f"{STORY_PLAN_PATH} not found.")

    plan = json.loads(STORY_PLAN_PATH.read_text(encoding="utf-8"))
    topic_slug = plan["topic_slug"]
    is_multi_part = bool(plan.get("is_multi_part", False))
    episodes = plan["episodes"]

    date_tag = datetime.now().strftime("[%d-%m-%y]")
    date_log = datetime.now().strftime("%Y-%m-%d")
    parent_folder_id = os.getenv("GOOGLE_DRIVE_FOLDER_ID", "").strip() or None

    creds = get_credentials()
    drive_service = build("drive", "v3", credentials=creds)

    if not is_multi_part:
        ep = episodes[0]
        local_mp4 = FINAL_DIR / f"{topic_slug}_part{ep['part_number']}.mp4"
        drive_name = f"{date_tag} {ep['title']}.mp4"
        link = upload_mp4(drive_service, local_mp4, drive_name, parent_folder_id)
        print(f"[Uploader] Uploaded single-part -> {drive_name} ({link})")
        append_daily_log(creds, date_log, ep["title"], 1, link, "SUCCESS")
        return link

    folder_title = f"{date_tag} {episodes[0]['title']}"
    subfolder_id = create_drive_folder(drive_service, folder_title, parent_folder_id)
    folder_link = f"https://drive.google.com/drive/folders/{subfolder_id}"

    for ep in episodes:
        p_num = ep["part_number"]
        local_mp4 = FINAL_DIR / f"{topic_slug}_part{p_num}.mp4"
        drive_name = f"Part {p_num} - {ep['title']}.mp4"
        ep_link = upload_mp4(drive_service, local_mp4, drive_name, subfolder_id)
        print(f"[Uploader] Uploaded {drive_name} ({ep_link})")

    append_daily_log(creds, date_log, episodes[0]["title"], len(episodes), folder_link, "SUCCESS")
    return folder_link


if __name__ == "__main__":
    upload_story_outputs()
