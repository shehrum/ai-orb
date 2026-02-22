"""Load .env before any test module imports trigger Agent() creation."""

from dotenv import load_dotenv

load_dotenv()
