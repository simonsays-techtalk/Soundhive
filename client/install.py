#!/usr/bin/env python3
import sys, subprocess, os

# Define home directory and dedicated virtual environment path.
HOME_DIR = os.environ["HOME"]
VENV_DIR = os.path.join(HOME_DIR, "venv", "soundhive")

# Ensure the installer is running inside the dedicated virtual environment.
if sys.prefix != VENV_DIR:
    print(f"Not in virtual environment. Using dedicated venv at {VENV_DIR}")
    if not os.path.exists(VENV_DIR):
        print(f"Creating virtual environment in {VENV_DIR} ...")
        subprocess.check_call([sys.executable, "-m", "venv", VENV_DIR])
        print("Virtual environment created.")
    new_python = os.path.join(VENV_DIR, "bin", "python3")
    print(f"Re-launching installer with {new_python} ...")
    os.execv(new_python, [new_python] + sys.argv)

# Now inside the dedicated virtual environment.
# Ensure pip is available.
try:
    import pip
except ModuleNotFoundError:
    print("pip is not available. Bootstrapping pip using ensurepip...")
    import ensurepip
    ensurepip.bootstrap()

# Pre-import dependency check.
dependencies = {
    "aiohttp": "aiohttp",
    "python-vlc": "vlc",
    "sounddevice": "sounddevice",
    "soundfile": "soundfile",
    "cryptography": "cryptography"
}

missing = []
for pkg, mod in dependencies.items():
    try:
        __import__(mod)
    except ModuleNotFoundError:
        missing.append(pkg)

if missing:
    print("Missing packages detected:", missing)
    for pkg in missing:
        print(f"Installing {pkg}...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", pkg])
    print("Restarting installer after installing missing dependencies...")
    os.execv(sys.executable, [sys.executable] + sys.argv)

# Now import the rest of the modules.
import os
import sys
import subprocess
import json
import venv
import tempfile
import stat
import platform
import base64

# Added imports for encryption.
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.backends import default_backend
from cryptography.fernet import Fernet

# VERSION = 2.5.40
# Define repository and configuration details.
REPO_DIR = os.path.join(HOME_DIR, "Soundhive")
REPO_URL = "https://github.com/simonsays-techtalk/Soundhive.git"
SERVICE_FILE_PATH = "/etc/systemd/system/soundhive.service"
CONFIG_FILE = "soundhive_config.json"
MIC2HAT_FLAG_FILE = ".mic2hat_installed"

def auto_confirm(prompt):
    if os.path.exists(MIC2HAT_FLAG_FILE):
        print(prompt + " y (auto-confirmed)")
        return "y"
    else:
        return input(prompt)

def manual_input(prompt):
    return input(prompt)

def detect_device_type():
    machine = platform.machine().strip()
    if machine == "armv6l":
        print("Detected device type: Pi Zero / Pi 1 (minimal install recommended).")
        return "minimal"
    else:
        print("Detected device type: " + machine + " (full install recommended).")
        return "full"

def prompt_for_configuration(install_type):
    if os.path.exists(CONFIG_FILE):
        choice = manual_input(f"Detected previous configuration in '{CONFIG_FILE}'.\nWould you like to use the existing configuration? [y/n]: ").strip().lower()
        if choice == "y":
            try:
                with open(CONFIG_FILE, "r") as f:
                    config = json.load(f)
                print("Using existing configuration.")
                return config
            except Exception as e:
                print(f"Error reading existing configuration: {e}")
                sys.exit(1)
        else:
            print("Proceeding with new configuration. The existing configuration will be overwritten.")
    print("Soundhive Client Setup")
    ha_url = manual_input("Enter Home Assistant URL (e.g., http://192.168.1.100:8123): ").strip()
    auth_token = manual_input("Enter Home Assistant Auth Token: ").strip()
    tts_engine = manual_input("Enter TTS Engine (default: tts.google_translate_en_com): ").strip()
    if not tts_engine:
        tts_engine = "tts.google_translate_en_com"
    alsa_device = manual_input("Enter ALSA device (default: dmix:CARD=seeed2micvoicec,DEV=0): ").strip()
    if not alsa_device:
        alsa_device = "dmix:CARD=seeed2micvoicec,DEV=0"
    
    if install_type == "full":
        stt_uri = manual_input("Enter STT server URI (default: http://192.168.1.100:10900/inference): ").strip()
        if not stt_uri:
            stt_uri = "http://192.168.1.100/inference"
        llm_uri = manual_input("Enter LLM server URI (default: http://192.168.1.100:11434/api/generate): ").strip()
        if not llm_uri:
            llm_uri = "http://192.168.1.100:11434/api/generate"
        active_timeout = manual_input("Enter active timeout in seconds (default: 15): ").strip()
        if not active_timeout:
            active_timeout = 15
        else:
            try:
                active_timeout = int(active_timeout)
            except ValueError:
                active_timeout = 15
    else:
        stt_uri = ""
        llm_uri = ""
        active_timeout = 0

    volume = manual_input("Enter default volume (0.0 to 1.0, default: 0.5): ").strip()
    if not volume:
        volume = 0.5
    else:
        try:
            volume = float(volume)
        except ValueError:
            volume = 0.5

    rms_threshold = "0.008"
    config = {
        "ha_url": ha_url,
        "auth_token": auth_token,
        "tts_engine": tts_engine,
        "alsa_device": alsa_device,
        "stt_uri": stt_uri,
        "stt_format": "wav",
        "active_timeout": active_timeout,
        "volume": volume,
        "wake_keyword": "hey assistant",
        "sleep_keyword": "goodbye",
        "llm_uri": llm_uri,
        "alarm_keyword": "alarm now",
        "clear_alarm_keyword": "clear alarm",
        "rms_threshold": rms_threshold,
        "install_type": install_type
    }
    return config

def write_encrypted_config_file(config, filename=CONFIG_FILE):
    master_password = manual_input("Enter a master password to secure your configuration: ").strip()
    confirm_password = manual_input("Confirm master password: ").strip()
    if master_password != confirm_password:
        print("Passwords do not match. Exiting.")
        sys.exit(1)
    password_bytes = master_password.encode()
    salt = os.urandom(16)
    kdf = PBKDF2HMAC(
         algorithm=hashes.SHA256(),
         length=32,
         salt=salt,
         iterations=100000,
         backend=default_backend()
    )
    key = base64.urlsafe_b64encode(kdf.derive(password_bytes))
    fernet = Fernet(key)
    config_json = json.dumps(config, indent=4).encode()
    encrypted_data = fernet.encrypt(config_json)
    out_data = {
         "salt": base64.b64encode(salt).decode('utf-8'),
         "data": encrypted_data.decode('utf-8')
    }
    try:
        with open(filename, "w") as f:
            json.dump(out_data, f, indent=4)
        print(f"Encrypted configuration saved to {filename}")
        return master_password
    except Exception as e:
        print(f"Error writing encrypted config file: {e}")
        sys.exit(1)

def print_install_summary(config):
    summary = f"""
    
Installer version: 2.5.20
---------------------------------------------------------    
The following components will be installed/configured:
---------------------------------------------------------
Installation Type: {config.get("install_type")}
---------------------------------------------------------
Python Packages:
  - aiohttp
  - python-vlc
  - sounddevice
  - soundfile
  - cryptography

System Packages:
  - git
  - ffmpeg
  - vlc
  - libvlc-dev
  - portaudio19-dev

Other Actions:
  - Creation of a dedicated virtual environment in:
      {VENV_DIR}
  - Cloning the Soundhive repository from GitHub into:
      {REPO_DIR}
      (Repository now contains "client" and "custom_component" folders)
  - (Optional) Installation of Respeaker mic2hat drivers via bash script (for supported devices)
  - Creation and activation of a systemd service for Soundhive Client
  - If applicable, installation of a shutdown script and service.
  - Configuration will be set to a {config.get("install_type")} install.
---------------------------------------------------------
"""
    print(summary)
    choice = manual_input("Do you want to continue with the installation? [y/n]: ").strip().lower()
    if choice != "y":
        print("Installation cancelled.")
        sys.exit(0)

def install_dependencies():
    print("Installing required Python packages...")
    try:
        subprocess.run([sys.executable, "-m", "pip", "install", "--upgrade", 
                        "aiohttp", "python-vlc", "sounddevice", "soundfile", "cryptography"], check=True)
        print("Python dependencies installed.")
    except subprocess.CalledProcessError as e:
        print(f"Error installing Python dependencies: {e}")
        sys.exit(1)
    
    print("Installing system packages...")
    try:
        subprocess.run(["sudo", "apt-get", "update"], check=True)
        subprocess.run(["sudo", "apt-get", "install", "-y", 
                        "git", "ffmpeg", "vlc", "libvlc-dev", "portaudio19-dev"], check=True)
        print("System packages installed.")
    except subprocess.CalledProcessError as e:
        print(f"Error installing system packages: {e}")
        sys.exit(1)

def install_respeaker_drivers_bash():
    if os.path.exists(MIC2HAT_FLAG_FILE):
        print("Respeaker drivers already installed, skipping installation.")
        return
    choice = manual_input("Do you want to install Respeaker mic2hat drivers? [y/n]: ").strip().lower()
    if choice != "y":
        print("Skipping Respeaker driver installation.")
        return

    bash_script = r"""#!/usr/bin/env bash
set -eo pipefail

kernel_formatted="$(uname -r | cut -f1,2 -d.)"
driver_url_status="$(curl -ILs https://github.com/HinTak/seeed-voicecard/archive/refs/heads/v$kernel_formatted.tar.gz | tac | grep -o "^HTTP.*" | cut -f2 -d' ' | head -1)"

if [ ! "$driver_url_status" = 200 ]; then
  echo "Could not find driver for kernel $kernel_formatted"
  exit 1
fi

apt-get update
apt-get install --no-install-recommends --yes \
    curl raspberrypi-kernel-headers dkms i2c-tools libasound2-plugins alsa-utils

temp_dir="$(mktemp -d)"

function finish {
   rm -rf "${temp_dir}"
}

trap finish EXIT

pushd "${temp_dir}"

echo 'Downloading source code'
curl -L -o - "https://github.com/HinTak/seeed-voicecard/archive/refs/heads/v$kernel_formatted.tar.gz" | tar -xzf -
cd seeed-voicecard-"$kernel_formatted"/

echo 'Building kernel module'
ver='0.3'
mod='seeed-voicecard'
src='./'
kernel="$(uname -r)"
marker='0.0.0'
threads="$(getconf _NPROCESSORS_ONLN)"
memory="$(LANG=C free -m|awk '/^Mem:/{print $2}')"
if [ "${memory}" -le 512 ] && [ "${threads}" -gt 2 ]; then
  threads=2
fi

mkdir -p "/usr/src/${mod}-${ver}"
cp -a "${src}"/* "/usr/src/${mod}-${ver}/"
dkms add -m "${mod}" -v "${ver}"
dkms build -k "${kernel}" -m "${mod}" -v "${ver}" -j "${threads}" && {
    dkms install --force -k "${kernel}" -m "${mod}" -v "${ver}"
}

mkdir -p "/var/lib/dkms/${mod}/${ver}/${marker}"

echo 'Updating boot configuration'
config='/boot/config.txt'
cp seeed-*-voicecard.dtbo /boot/overlays
grep -q "^snd-soc-ac108$" /etc/modules || echo "snd-soc-ac108" >> /etc/modules
sed -i -e 's:#dtparam=i2c_arm=on:dtparam=i2c_arm=on:g' "${config}"
echo "dtoverlay=i2s-mmap" >> "${config}"
echo "dtparam=i2s=on" >> "${config}"
mkdir -p /etc/voicecard
cp *.conf *.state /etc/voicecard
cp seeed-voicecard /usr/bin/
cp seeed-voicecard.service /lib/systemd/system/
systemctl enable --now seeed-voicecard.service

echo 'Done. Please reboot the system.'
popd
"""
    try:
        with tempfile.NamedTemporaryFile("w", delete=False, suffix=".sh") as temp_script:
            temp_script.write(bash_script)
            temp_script_name = temp_script.name
        os.chmod(temp_script_name, os.stat(temp_script_name).st_mode | stat.S_IEXEC)
        print("Running Respeaker driver installation script...")
        try:
            subprocess.run(["sudo", temp_script_name], capture_output=True, text=True, check=True)
        except subprocess.CalledProcessError as e:
            combined_output = (e.stdout + e.stderr).lower()
            if "dkms tree already contains: seeed-voicecard-0.3" in combined_output:
                print("Mic2hat drivers already installed (DKMS module present), skipping installation.")
            else:
                print("Error installing Respeaker drivers via bash script:")
                print(e.stderr)
                sys.exit(1)
        print("Respeaker drivers installed successfully (or already present).")
        with open(MIC2HAT_FLAG_FILE, "w") as f:
            f.write("installed")
    except Exception as exc:
        print(f"Unexpected error: {exc}")
        sys.exit(1)
    finally:
        if os.path.exists(temp_script_name):
            os.remove(temp_script_name)

def clone_repository():
    if not os.path.exists(REPO_DIR):
        print(f"Cloning Soundhive repository from {REPO_URL} into {REPO_DIR} ...")
        try:
            parent_dir = os.path.dirname(REPO_DIR)
            if not os.path.exists(parent_dir):
                os.makedirs(parent_dir)
            subprocess.run(["git", "clone", REPO_URL, REPO_DIR], check=True)
            print("Repository cloned successfully.")
        except subprocess.CalledProcessError as e:
            print(f"Error cloning repository: {e}")
            sys.exit(1)
    else:
        print("Repository already exists, pulling latest changes...")
        try:
            subprocess.run(["git", "-C", REPO_DIR, "pull"], check=True)
        except subprocess.CalledProcessError as e:
            print(f"Error updating repository: {e}")

def create_systemd_service(master_pass):
    user = os.getlogin()
    working_dir = os.path.join(REPO_DIR, "client")
    client_script = os.path.join(working_dir, "soundhive_client.py")
    try:
        subprocess.run(["chmod", "+x", client_script], check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error setting executable permission on {client_script}: {e}")
        sys.exit(1)
    venv_python = os.path.join(VENV_DIR, "bin", "python3")
    exec_command = f"{venv_python} {client_script}"
    service_content = f"""[Unit]
Description=Soundhive Client Service
After=network.target

[Service]
Type=simple
User={user}
WorkingDirectory={working_dir}
Environment="MASTER_PASS={master_pass}"
ExecStart={exec_command}
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
"""
    try:
        print(f"Creating systemd service file at {SERVICE_FILE_PATH}...")
        with open("soundhive.service", "w") as f:
            f.write(service_content)
        subprocess.run(["sudo", "cp", "soundhive.service", SERVICE_FILE_PATH], check=True)
        subprocess.run(["sudo", "systemctl", "daemon-reload"], check=True)
        subprocess.run(["sudo", "systemctl", "enable", "soundhive.service"], check=True)
        subprocess.run(["sudo", "systemctl", "start", "soundhive.service"], check=True)
        print("Soundhive Client service created and started successfully.")
    except Exception as e:
        print(f"Error creating systemd service: {e}")
        sys.exit(1)

def create_poweroff_service():
    poweroff_src = os.path.join(REPO_DIR, "client", "poweroff.py")
    poweroff_dest = "/usr/local/bin/poweroff_pi.py"
    print(f"Copying shutdown script from {poweroff_src} to {poweroff_dest} ...")
    try:
        subprocess.run(["sudo", "cp", poweroff_src, poweroff_dest], check=True)
        subprocess.run(["sudo", "chmod", "+x", poweroff_dest], check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error installing shutdown script: {e}")
        sys.exit(1)
    
    poweroff_service_file = "/etc/systemd/system/poweroff_pi.service"
    poweroff_service_content = f"""[Unit]
Description=Shutdown Pi Button Service
After=multi-user.target

[Service]
Type=simple
ExecStart={poweroff_dest}
Restart=always
User=root

[Install]
WantedBy=multi-user.target
"""
    try:
        print(f"Creating shutdown service file at {poweroff_service_file}...")
        with open("poweroff_pi.service", "w") as f:
            f.write(poweroff_service_content)
        subprocess.run(["sudo", "cp", "poweroff_pi.service", poweroff_service_file], check=True)
        subprocess.run(["sudo", "systemctl", "daemon-reload"], check=True)
        subprocess.run(["sudo", "systemctl", "enable", "poweroff_pi.service"], check=True)
        subprocess.run(["sudo", "systemctl", "start", "poweroff_pi.service"], check=True)
        print("Shutdown Pi Button service created and started successfully.")
    except subprocess.CalledProcessError as e:
        print(f"Error creating shutdown service: {e}")
        sys.exit(1)

def main():
    device_type = detect_device_type()  # Determine if minimal or full install.
    print_install_summary({"install_type": device_type})
    config = prompt_for_configuration(device_type)
    # Write the encrypted configuration and capture the master password.
    master_pass = write_encrypted_config_file(config)
    install_dependencies()
    if device_type == "full":
        install_respeaker_drivers_bash()
    clone_repository()
    client_config_dest = os.path.join(REPO_DIR, "client", CONFIG_FILE)
    print(f"Copying configuration file to {client_config_dest} ...")
    subprocess.run(["cp", CONFIG_FILE, client_config_dest], check=True)
    create_systemd_service(master_pass)
    
    if os.path.exists(MIC2HAT_FLAG_FILE):
        print("Mic2hat drivers detected, installing shutdown service...")
        create_poweroff_service()
    
    print("Soundhive Client setup is complete.")

if __name__ == "__main__":
    main()
