from cryptography.fernet import Fernet

def encrypt_api_key(api_key):
    key = Fernet.generate_key()
    cipher_suite = Fernet(key)
    encrypted_api_key = cipher_suite.encrypt(api_key.encode())

    with open('api_key.key', 'wb') as key_file:
        key_file.write(key)

    with open('encrypted_api_key.txt', 'wb') as encrypted_file:
        encrypted_file.write(encrypted_api_key)

    print("Clave de API encriptada y guardada exitosamente.")
    print(f"Guarda esta clave en un lugar seguro: {key.decode()}")

def decrypt_key(encrypted_key):
    # Usa la clave Fernet para desencriptar la clave API
    with open('api_key.key', 'rb') as key_file:
        key = key_file.read()
    cipher_suite = Fernet(key)
    decrypted_key = cipher_suite.decrypt(encrypted_key)
    return decrypted_key.decode('utf-8')

if __name__ == "__main__":
    api_key = " clave de API " # Reemplaza con tu clave de API real
    encrypt_api_key(api_key)
