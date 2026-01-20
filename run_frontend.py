import sys
import os
from streamlit.web import cli as stcli


def main():
    # 1. Get the absolute path of the current directory (Project Root)
    root_dir = os.path.abspath(os.path.dirname(__file__))

    # 2. Add it to sys.path
    # This makes 'src' importable from anywhere in the execution
    if root_dir not in sys.path:
        sys.path.append(root_dir)

    # 3. Construct the command to run streamlit
    # We point it to the internal app file
    script_path = os.path.join(root_dir, "src", "frontend", "Home.py")

    # 4. Mimic the command line arguments for Streamlit
    sys.argv = ["streamlit", "run", script_path]

    print(f"🚀 Launching Dashboard from root: {root_dir}")

    # 5. Execute Streamlit
    sys.exit(stcli.main())


if __name__ == "__main__":
    main()