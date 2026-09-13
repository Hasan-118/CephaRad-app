"""
Google Drive Storage for Aariz
ذخیره و بازیابی مش‌ها در Google Drive
"""

import streamlit as st
import pickle
import io
import hashlib
import json


def get_drive_service():
    """اتصال به Google Drive با Service Account"""
    try:
        from google.oauth2 import service_account
        from googleapiclient.discovery import build

        # خواندن JSON از secrets
        json_str = st.secrets["gdrive"]["service_account_json"]

        # اگر JSON به صورت string است، parse کن
        if isinstance(json_str, str):
            creds_info = json.loads(json_str)
        else:
            creds_info = json_str

        creds = service_account.Credentials.from_service_account_info(
            creds_info,
            scopes=['https://www.googleapis.com/auth/drive.file']
        )

        service = build('drive', 'v3', credentials=creds, cache_discovery=False)
        return service
    except Exception as e:
        st.error(f"❌ خطا در اتصال به Google Drive: {type(e).__name__}: {e}")
        return None


def get_folder_id():
    """دریافت Folder ID از secrets"""
    try:
        return st.secrets["gdrive"]["folder_id"]
    except Exception:
        return None


def list_files_in_folder(service, folder_id, name_prefix=""):
    """لیست فایل‌های یک پوشه"""
    try:
        query = f"'{folder_id}' in parents and trashed=false"
        if name_prefix:
            query += f" and name contains '{name_prefix}'"
        results = service.files().list(
            q=query, pageSize=100, fields="files(id, name, size, modifiedTime)"
        ).execute()
        return results.get('files', [])
    except Exception as e:
        st.warning(f"⚠️ خطا در لیست فایل‌ها: {e}")
        return []


def upload_mesh_to_drive(mesh, filename, folder_id=None):
    """ذخیره مش در Google Drive"""
    try:
        from googleapiclient.http import MediaIoBaseUpload
    except ImportError:
        st.error("❌ پکیج `google-api-python-client` نصب نیست")
        return None

    service = get_drive_service()
    if service is None:
        return None

    if folder_id is None:
        folder_id = get_folder_id()

    if folder_id is None:
        st.error("❌ Folder ID در secrets تنظیم نشده")
        return None

    try:
        # Serialize mesh
        mesh_bytes = pickle.dumps(mesh)
        mesh_stream = io.BytesIO(mesh_bytes)

        # جستجوی فایل موجود
        existing = list_files_in_folder(service, folder_id, filename)
        existing_files = [f for f in existing if f['name'] == filename]

        media = MediaIoBaseUpload(mesh_stream, mimetype='application/octet-stream')

        if existing_files:
            # به‌روزرسانی فایل موجود
            file_id = existing_files[0]['id']
            service.files().update(
                fileId=file_id,
                media_body=media
            ).execute()
            return file_id
        else:
            # ساخت فایل جدید
            file_metadata = {
                'name': filename,
                'parents': [folder_id]
            }
            file = service.files().create(
                body=file_metadata,
                media_body=media,
                fields='id'
            ).execute()
            return file.get('id')
    except Exception as e:
        st.error(f"❌ خطا در ذخیره در Google Drive: {type(e).__name__}: {e}")
        return None


def download_mesh_from_drive(filename, folder_id=None):
    """بارگذاری مش از Google Drive"""
    try:
        from googleapiclient.http import MediaIoBaseDownload
    except ImportError:
        return None

    service = get_drive_service()
    if service is None:
        return None

    if folder_id is None:
        folder_id = get_folder_id()

    if folder_id is None:
        return None

    try:
        files = list_files_in_folder(service, folder_id, filename)
        matching = [f for f in files if f['name'] == filename]

        if not matching:
            return None

        file_id = matching[0]['id']
        request = service.files().get_media(fileId=file_id)
        file_stream = io.BytesIO()
        downloader = MediaIoBaseDownload(file_stream, request)

        done = False
        while not done:
            status, done = downloader.next_chunk()

        file_stream.seek(0)
        mesh = pickle.load(file_stream)
        return mesh
    except Exception as e:
        st.warning(f"⚠️ خطا در بارگذاری از Google Drive: {type(e).__name__}: {e}")
        return None


def delete_mesh_from_drive(filename, folder_id=None):
    """حذف مش از Google Drive"""
    service = get_drive_service()
    if service is None:
        return False

    if folder_id is None:
        folder_id = get_folder_id()

    try:
        files = list_files_in_folder(service, folder_id, filename)
        matching = [f for f in files if f['name'] == filename]
        for f in matching:
            service.files().delete(fileId=f['id']).execute()
        return True
    except Exception as e:
        st.warning(f"⚠️ خطا در حذف: {e}")
        return False


def get_file_hash(uploaded_file):
    """محاسبه hash فایل"""
    if uploaded_file is None:
        return None
    try:
        uploaded_file.seek(0)
        content = uploaded_file.read()
        uploaded_file.seek(0)
        return hashlib.md5(content).hexdigest()[:12]
    except Exception:
        return None
