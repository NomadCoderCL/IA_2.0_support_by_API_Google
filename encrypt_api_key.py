from cryptography.fernet import Fernet

key = Fernet.generate_key()
with open('api_key.key', 'wb') as key_file:
    key_file.write(key)
api_key = " clave de API ".encode()  # Reemplaza con tu clave de API real
cipher_suite = Fernet(key)
ciphered_api_key = cipher_suite.encrypt(api_key)
with open('encrypted_api_key.txt', 'wb') as encrypted_file:
    encrypted_file.write(ciphered_api_key)
print("Clave de API encriptada y guardada exitosamente.")
