#!/usr/bin/env python3
import os
import sys
import json
import base64
import tempfile
import subprocess
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.backends import default_backend
from cryptography.fernet import Fernet

CONFIG_FILE = "soundhive_config.json"

def load_encrypted_config():
    try:
        with open(CONFIG_FILE, "r") as f:
            encrypted_config = json.load(f)
        salt_b64 = encrypted_config["salt"]
        encrypted_data = encrypted_config["data"]
        salt = base64.b64decode(salt_b64)
        return salt, encrypted_data
    except Exception as e:
        print("Error loading configuration file:", e)
        sys.exit(1)

def decrypt_config(master_password, salt, encrypted_data):
    password_bytes = master_password.encode()
    kdf = PBKDF2HMAC(
        algorithm=hashes.SHA256(),
        length=32,
        salt=salt,
        iterations=100000,
        backend=default_backend()
    )
    key = base64.urlsafe_b64encode(kdf.derive(password_bytes))
    fernet = Fernet(key)
    try:
        decrypted_json = fernet.decrypt(encrypted_data.encode())
        config_data = json.loads(decrypted_json.decode())
        return config_data, key  # Return key so we can re-encrypt later.
    except Exception as e:
        print("Failed to decrypt configuration:", e)
        sys.exit(1)

def encrypt_config(config_data, key, salt):
    fernet = Fernet(key)
    config_json = json.dumps(config_data, indent=4).encode()
    encrypted_data = fernet.encrypt(config_json)
    out_data = {
        "salt": base64.b64encode(salt).decode('utf-8'),
        "data": encrypted_data.decode('utf-8')
    }
    return out_data

def edit_config(config_data):
    # Write decrypted config to a temporary file.
    with tempfile.NamedTemporaryFile("w+", delete=False, suffix=".json") as tf:
        temp_filename = tf.name
        json.dump(config_data, tf, indent=4)
    # Open the temporary file in the user's editor.
    editor = os.getenv("EDITOR", "nano")
    subprocess.call([editor, temp_filename])
    # After editing, read the file back.
    try:
        with open(temp_filename, "r") as tf:
            new_config = json.load(tf)
    except Exception as e:
        print("Error reading edited configuration:", e)
        os.unlink(temp_filename)
        sys.exit(1)
    os.unlink(temp_filename)
    return new_config

def main():
    salt, encrypted_data = load_encrypted_config()
    master_password = os.getenv("MASTER_PASS")
    if not master_password:
        master_password = input("Enter master password to decrypt configuration: ").strip()
    config_data, key = decrypt_config(master_password, salt, encrypted_data)
    print("Current configuration:")
    print(json.dumps(config_data, indent=4))
    
    choice = input("Do you want to edit the configuration? (y/n): ").strip().lower()
    if choice != "y":
        print("No changes made.")
        sys.exit(0)
    
    new_config = edit_config(config_data)
    new_encrypted = encrypt_config(new_config, key, salt)
    try:
        with open(CONFIG_FILE, "w") as f:
            json.dump(new_encrypted, f, indent=4)
        print("Configuration updated and re-encrypted successfully.")
    except Exception as e:
        print("Error writing updated configuration:", e)
        sys.exit(1)

if __name__ == "__main__":
    main()
