"""
Google Drive Storage with OAuth 2.0
ذخیره و بازیابی مش‌ها با احراز هویت OAuth - نسخه نهایی با PKCE + فایل موقت
"""

import streamlit as st
import pickle
import io
import hashlib
import secrets
import base64
import os


SCOPES = ['https://www.googleapis.com/auth/drive.file']
TOKEN_URI = 'https://oauth2.googleapis.com/token'
REDIRECT_URI = 'http://localhost'

# مسیر فایل موقت برای ذخیره code_verifier
PKCE_FILE = '/tmp/aariz_pkce_verifier.txt'


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


def _generate_pkce_pair():
    """ساخت code_verifier و code_challenge"""
    code_verifier = base64.urlsafe_b64encode(secrets.token_bytes(32)).rstrip(b'=').decode('utf-8')
    code_challenge = base64.urlsafe_b64encode(
        hashlib.sha256(code_verifier.encode('utf-8')).digest()
    ).rstrip(b'=').decode('utf-8')
    return code_verifier, code_challenge


def _save_verifier(code_verifier):
    """ذخیره code_verifier در فایل موقت"""
    try:
        with open(PKCE_FILE, 'w') as f:
            f.write(code_verifier)
    except Exception as e:
        st.warning(f"⚠️ خطا در ذخیره code_verifier: {e}")


def _load_verifier():
    """بارگذاری code_verifier از فایل موقت"""
    try:
        if os.path.exists(PKCE_FILE):
            with open(PKCE_FILE, 'r') as f:
                return f.read().strip()
    except Exception:
        pass
    return None


def _delete_verifier():
    """پاک کردن فایل code_verifier"""
    try:
        if os.path.exists(PKCE_FILE):
            os.remove(PKCE_FILE)
    except Exception:
        pass


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
    """اتصال به Google Drive"""
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
    """ساخت URL برای autorize اولیه با PKCE"""
    config = _get_config()
    if config is None:
        return None

    from urllib.parse import urlencode

    # پاک کردن code_verifier قدیمی
    _delete_verifier()

    code_verifier, code_challenge = _generate_pkce_pair()

    # ذخیره در فایل موقت + session_state
    _save_verifier(code_verifier)
    st.session_state["oauth_code_verifier"] = code_verifier

    # نمایش دیباگ
    st.write(f"🔍 **Debug:** code_verifier ذخیره شد، طول: `{len(code_verifier)}`")

    params = {
        'client_id': config['client_id'],
        'redirect_uri': REDIRECT_URI,
        'response_type': 'code',
        'scope': ' '.join(SCOPES),
        'access_type': 'offline',
        'prompt': 'consent',
        'code_challenge': code_challenge,
        'code_challenge_method': 'S256',
    }

    auth_url = f"https://accounts.google.com/o/oauth2/v2/auth?{urlencode(params)}"
    return auth_url


def exchange_code_for_token(code):
    """تبدیل code به refresh token"""
    import requests

    config = _get_config()
    if config is None:
        return None

    # تلاش از فایل موقت، اگر نبود از session_state
    code_verifier = _load_verifier()
    if not code_verifier:
        code_verifier = st.session_state.get("oauth_code_verifier", None)

    if not code_verifier:
        st.error("❌ code_verifier پیدا نشد.")
        return None

    token_data = {
        'client_id': config['client_id'],
        'client_secret': config['client_secret'],
        'code': code,
        'code_verifier': code_verifier,
        'grant_type': 'authorization_code',
        'redirect_uri': REDIRECT_URI,
    }

    try:
        response = requests.post(TOKEN_URI, data=token_data, timeout=30)

        # دیباگ
        st.write(f"🔍 **Debug:** status=`{response.status_code}`, verifier_length=`{len(code_verifier)}`")

        response.raise_for_status()
        tokens = response.json()
        refresh_token = tokens.get('refresh_token')

        if refresh_token:
            _delete_verifier()
            st.session_state.pop("oauth_code_verifier", None)
            return refresh_token
        else:
            st.error("❌ refresh_token در پاسخ نبود")
            return None
    except Exception as e:
        st.error(f"❌ خطا: {type(e).__name__}: {e}")
        st.code(response.text if 'response' in locals() else "no response")
        return None


def list_files_in_folder(service, folder_id, name_prefix=""):
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
        st.error(f"❌ خطا در ذخیره: {type(e).__name__}: {e}")
        return None


def download_mesh_from_drive(filename, folder_id=None):
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
        st.warning(f"⚠️ خطا در بارگذاری: {type(e).__name__}: {e}")
        return None
