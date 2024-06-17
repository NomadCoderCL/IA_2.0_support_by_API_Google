from cryptography.fernet import Fernet

def load_key():
    return open("api_key.key", "rb").read()

def load_api_key():
    key = load_key()
    cipher_suite = Fernet(key)
    with open("encrypted_api_key.txt", "rb") as encrypted_file:
        encrypted_api_key = encrypted_file.read()
    api_key = cipher_suite.decrypt(encrypted_api_key)
    return api_key.decode()
