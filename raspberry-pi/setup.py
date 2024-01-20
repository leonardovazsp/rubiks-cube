import os
import subprocess
import sys
import argparse

def create_virtualenv(path):
    """Create a virtual environment."""
    if not os.path.exists(path):
        subprocess.check_call([sys.executable, '-m', 'venv', path])
    print("Virtual environment created at {}".format(path))

def install_requirements(venv_path, requirements_path):
    """Install requirements from a requirements.txt file."""
    pip_executable = os.path.join(venv_path, 'bin', 'pip')
    subprocess.check_call([pip_executable, 'install', '-r', requirements_path])
    print("Requirements installed.")

def setup_systemd_service(service_name, service_template, service_path):
    """Create and enable a systemd service."""
    with open(service_name, 'w') as service_file:
        service_file.write(service_template)
    
    subprocess.check_call(['sudo', 'mv', service_name, service_path])
    subprocess.check_call(['sudo', 'systemctl', 'daemon-reload'])
    subprocess.check_call(['sudo', 'systemctl', 'enable', service_name])
    subprocess.check_call(['sudo', 'systemctl', 'start', service_name])
    print("Systemd service set up and started.")

def set_environment_variables(variables):
    output = ""
    for var in variables:
        output += f'Environment="{var[0]}"="{var[1]}"\n'
    return output

def main(environment_variables):
    app_dir = os.path.dirname(os.path.realpath(__file__))
    venv_path = os.path.join(app_dir, 'venv')
    requirements_path = os.path.join(app_dir, 'requirements.txt')
    service_name = 'rubiks.service'
    service_path = '/etc/systemd/system/'
    environment_variables = set_environment_variables(environment_variables)

    service_template = f"""[Unit]
Description=Rubiks Cube Controller API
After=network.target

[Service]
{environment_variables}
User=pi
WorkingDirectory={app_dir}
ExecStartPre=ngrok http --domain=servopi.ngrok.io 8000
ExecStart={venv_path}/bin/python {app_dir}/main.py
Restart=on-failure

[Install]
WantedBy=multi-user.target
"""

    create_virtualenv(venv_path)
    install_requirements(venv_path, requirements_path)
    setup_systemd_service(service_name, service_template, service_path)
    


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--server-type', type=str, help='Define the type of server (master or camera)')
    args = parser.parse_args()
    assert args.server_type is not None, "Please define the type of server. For example, python3 setup.py --server-type camera"
    environment_variables = [("SERVER_TYPE", args.server_type)]
    main(environment_variables)
