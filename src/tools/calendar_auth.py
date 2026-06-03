"""Google OAuth2 authentication with automatic token refresh."""

import json
import os
from pathlib import Path
from datetime import datetime
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google.auth.exceptions import RefreshError
from google_auth_oauthlib.flow import InstalledAppFlow
import webbrowser


class GoogleCalendarAuth:
    """Handle Google Calendar OAuth2 authentication with automatic token refresh."""
    
    TOKEN_FILE = Path(__file__).parent.parent.parent / "token.json"
    CREDENTIALS_FILE = Path(__file__).parent.parent.parent / "credentials.json"
    SCOPES = ["https://www.googleapis.com/auth/calendar"]
    
    @classmethod
    def get_credentials(cls) -> Credentials:
        """
        Load and refresh Google Calendar credentials.
        Handles authentication flow if token.json is missing.
        
        Returns:
            google.oauth2.credentials.Credentials: Valid credentials with refreshed token
            
        Raises:
            FileNotFoundError: If credentials.json doesn't exist
            RefreshError: If token refresh fails
        """
        if not cls.CREDENTIALS_FILE.exists():
            raise FileNotFoundError(
                f"credentials.json not found at {cls.CREDENTIALS_FILE}\n"
                "Get it from: https://developers.google.com/calendar/api/quickstart/python"
            )
        
        credentials = None
        
        # Load existing token if available
        if cls.TOKEN_FILE.exists():
            print(f"📂 Loading existing token from {cls.TOKEN_FILE}")
            with open(cls.TOKEN_FILE, "r") as f:
                token_data = json.load(f)
            
            # Create Credentials object from saved token
            credentials = Credentials(
                token=token_data.get("token"),
                refresh_token=token_data.get("refresh_token"),
                token_uri=token_data.get("token_uri"),
                client_id=token_data.get("client_id"),
                client_secret=token_data.get("client_secret"),
                scopes=token_data.get("scopes", cls.SCOPES)
            )
        else:
            print("❌ token.json not found. Starting OAuth2 authentication flow...")
            credentials = cls._authenticate()
        
        # Check if token is expired and refresh if needed
        if credentials.expired and credentials.refresh_token:
            try:
                print("🔄 Token expired. Refreshing...")
                credentials.refresh(Request())
                cls._save_credentials(credentials)
                print("✅ Token refreshed successfully")
            except RefreshError as e:
                print(f"⚠️  Token refresh failed: {e}")
                print("Starting re-authentication flow...")
                credentials = cls._authenticate()
        
        return credentials
    
    @classmethod
    def _authenticate(cls) -> Credentials:
        """
        Execute Google OAuth2 authentication flow.
        Opens browser for user to authorize and saves the token.
        
        Returns:
            Credentials: Authenticated credentials
        """
        try:
            # Create OAuth2 flow from credentials.json
            flow = InstalledAppFlow.from_client_secrets_file(
                cls.CREDENTIALS_FILE,
                cls.SCOPES
            )
            
            print("\n" + "="*60)
            print("🔐 Google Calendar Authentication Required")
            print("="*60)
            print("\nOpening browser for Google authentication...")
            print("If browser doesn't open, visit the URL shown in the next step.\n")
            
            # Run the flow - opens browser automatically
            credentials = flow.run_local_server(port=0, open_browser=True)
            
            # Save the credentials
            cls._save_credentials(credentials)
            print("\n✅ Authentication successful!")
            print(f"📝 Token saved to {cls.TOKEN_FILE}\n")
            
            return credentials
            
        except Exception as e:
            raise RuntimeError(
                f"Authentication failed: {str(e)}\n"
                "Please ensure credentials.json is valid and try again."
            )
    
    @classmethod
    def _save_credentials(cls, credentials: Credentials) -> None:
        """Save refreshed credentials back to token.json."""
        token_data = {
            "token": credentials.token,
            "refresh_token": credentials.refresh_token,
            "token_uri": credentials.token_uri,
            "client_id": credentials.client_id,
            "client_secret": credentials.client_secret,
            "scopes": credentials.scopes or cls.SCOPES,
            "universe_domain": getattr(credentials, "universe_domain", "googleapis.com"),
            "account": "",
            "expiry": credentials.expiry.isoformat() if credentials.expiry else None
        }
        
        with open(cls.TOKEN_FILE, "w") as f:
            json.dump(token_data, f, indent=2)
        
        print(f"💾 Credentials saved to {cls.TOKEN_FILE}")
    
    @classmethod
    def get_credentials_dict(cls) -> dict:
        """
        Get credentials as a dictionary for services that need it.
        
        Returns:
            dict: Credentials data with refreshed token
        """
        credentials = cls.get_credentials()
        return {
            "token": credentials.token,
            "refresh_token": credentials.refresh_token,
            "token_uri": credentials.token_uri,
            "client_id": credentials.client_id,
            "client_secret": credentials.client_secret,
            "scopes": credentials.scopes,
            "expiry": credentials.expiry.isoformat() if credentials.expiry else None
        }

