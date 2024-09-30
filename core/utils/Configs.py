
from pydantic import BaseSettings

class Settings(BaseSettings):
    """Use this class for checking environments vars
    """
    pguser: str
    pgpass: str
    pgdbname: str
    pghost: str
    #auth, secret generated using openssl rand -hex 32
    secret_key: str
    algorithm: str
    access_token_expiration_minutes: int

    class Config:
        env_file = ".env"