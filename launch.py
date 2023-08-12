from VenvManager import VenvManager
from utilities.utility import create_requirements_file
from os.path import exists
from os import mkdir

if __name__ == "__main__":
    venv_manager = VenvManager()

    if exists("cache") is False:
        mkdir("cache")

    create_requirements_file()

    venv_manager.InstallWRequirements()

    venv_manager.RunScript("main")
