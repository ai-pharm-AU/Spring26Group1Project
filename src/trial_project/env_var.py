import os

def get_env_var(name):
    # Try Google Colab secret store
    """
    try:
        from google.colab import userdata
        key = userdata.get(name)
        if key:
            return key
    except Exception:
        pass  # Not running in Colab
        """

    # Try environment variable
    key = os.getenv(name)
    if key:
        return key

    #  Fallback (prompt or error)
    raise RuntimeError(f"{name} not found in Colab secrets or environment variables")
