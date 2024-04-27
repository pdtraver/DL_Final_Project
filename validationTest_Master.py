import subprocess

def run_script(environment, script_name):
    """
    Executes a Python script within a specified Conda environment using 'conda run'.
    Args:
    - environment: The name of the Conda environment.
    - script_name: The name of the Python script to run.
    """
    command = f"conda run -n {environment} python {script_name}"
    result = subprocess.run(command, shell=True)
    if result.stdout:
        print(result.stdout)
    if result.stderr:
        print("Error:", result.stderr)

# Running scripts in different Conda environments
run_script('OpenSTL', 'validationTest_SimVP.py')
run_script('Slot', 'validationTest_Slot.py')