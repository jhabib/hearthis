"""Create local deployment credentials without printing their values."""
from pathlib import Path
import secrets

path = Path(__file__).resolve().parents[1] / ".env"
if path.exists():
    raise SystemExit(".env already exists; preserving its configuration")
with path.open("x") as stream:
    stream.write(f"HEARTHIS_DATABASE_PASSWORD={secrets.token_hex(24)}\n")
    stream.write(f"HEARTHIS_API_TOKEN={secrets.token_hex(32)}\n")
    stream.write("HEARTHIS_PORT=18080\n")
path.chmod(0o600)
print("Created .env with private local credentials and port 18080")
