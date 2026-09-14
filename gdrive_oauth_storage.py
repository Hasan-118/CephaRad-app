"""
Google Drive Storage with OAuth 2.0
ذخیره و بازیابی مش‌ها با احراز هویت OAuth
"""

import streamlit as st
import pickle
import io
import hashlib


SCOPES = ['https://www.googleapis.com/auth/drive.file']
TOKEN_URI = 'https://oauth2.googleapis.com/token'
REDIRECT_URI = 'http://localhost'


def _get_config():
    """دریافت تنظیمات از secrets"""
    try:
        return {
            'client_id': st.secrets["gdrive_oauth"]["client_id"],
            'client_secret': st.secrets["gdrive_oauth"]["client_secret"],
            'folder_id': st.secrets["gdrive_oauth"]["folder_id"],
            'refresh_token': st.secrets["gdrive_oauth"].get("refresh_token", ""),
        }
    except Exception as e:
        st.error(f"❌ خطا در خواندن secrets: {e}")
        return None


def get_oauth_credentials():
    """ساخت credentials از refresh token"""
    from google.oauth2.credentials import Credentials
    from google.auth.transport.requests import Request

    config = _get_config()
    if config is None:
        return None

    refresh_token = config.get('refresh_token', '').strip()
    if not refresh_token:
        return None

    try:
        creds = Credentials(
            token=None,
            refresh_token=refresh_token,
            token_uri=TOKEN_URI,
            client_id=config['client_id'],
            client_secret=config['client_secret'],
            scopes=SCOPES,
        )
        creds.refresh(Request())
        return creds
    except Exception as e:
        st.error(f"❌ خطا در refresh token: {type(e).__name__}: {e}")
        return None


def get_drive_service():
    """اتصال به Google Drive با OAuth"""
    try:
        from googleapiclient.discovery import build
        creds = get_oauth_credentials()
        if creds is None:
            return None
        return build('drive', 'v3', credentials=creds, cache_discovery=False)
    except Exception as e:
        st.error(f"❌ خطا در اتصال به Google Drive: {e}")
        return None


def get_auth_url():
    """ساخت URL برای autorize اولیه"""
    config = _get_config()
    if config is None:
        return None

    from google_auth_oauthlib.flow import Flow

    flow = Flow.from_client_config(
        {
            "web": {
                "client_id": config['client_id'],
                "client_secret": config['client_secret'],
                "auth_uri": "https://accounts.google.com/o/oauth2/auth",
                "token_uri": TOKEN_URI,
                "redirect_uris": [REDIRECT_URI],
            }
        },
        scopes=SCOPES,
    )
    flow.redirect_uri = REDIRECT_URI

    auth_url, _ = flow.authorization_url(
        access_type='offline',
        prompt='consent',
        include_granted_scopes='true'
    )
    return auth_url


def exchange_code_for_token(code):
    """تبدیل code به refresh token"""
    config = _get_config()
    if config is None:
        return None

    from google_auth_oauthlib.flow import Flow

    flow = Flow.from_client_config(
        {
            "web": {
                "client_id": config['client_id'],
                "client_secret": config['client_secret'],
                "auth_uri": "https://accounts.google.com/o/oauth2/auth",
                "token_uri": TOKEN_URI,
                "redirect_uris": [REDIRECT_URI],
            }
        },
        scopes=SCOPES,
    )
    flow.redirect_uri = REDIRECT_URI

    try:
        flow.fetch_token(code=code)
        return flow.credentials.refresh_token
    except Exception as e:
        st.error(f"❌ خطا در تبدیل code: {type(e).__name__}: {e}")
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
        return None

    service = get_drive_service()
    if service is None:
        return None

    if folder_id is None:
        config = _get_config()
        folder_id = config['folder_id'] if config else None

    if folder_id is None:
        return None

    try:
        mesh_bytes = pickle.dumps(mesh)
        mesh_stream = io.BytesIO(mesh_bytes)

        existing = list_files_in_folder(service, folder_id, filename)
        existing_files = [f for f in existing if f['name'] == filename]

        media = MediaIoBaseUpload(mesh_stream, mimetype='application/octet-stream')

        if existing_files:
            file_id = existing_files[0]['id']
            service.files().update(fileId=file_id, media_body=media).execute()
            return file_id
        else:
            file_metadata = {'name': filename, 'parents': [folder_id]}
            file = service.files().create(
                body=file_metadata, media_body=media, fields='id'
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
        config = _get_config()
        folder_id = config['folder_id'] if config else None

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
        return pickle.load(file_stream)
    except Exception as e:
        st.warning(f"⚠️ خطا در بارگذاری از Google Drive: {type(e).__name__}: {e}")
        return None
