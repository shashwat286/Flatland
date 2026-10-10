import json
import os

def create_notebook():
    cells = []

    def add_markdown(text):
        cells.append({
            "cell_type": "markdown",
            "metadata": {},
            "source": text
        })

    def add_code(code):
        """
        Jupyter expects source as a single string (or list of strings with \\n).
        Storing as a single string is the safest and most compatible approach.
        """
        cells.append({
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": code
        })

    add_markdown("# Flatland APPO Training on Google Colab\n")

    add_markdown("### 1. Install Dependencies\n")
    add_code(
        "!pip install flatland-rl==4.0.3\n"
        "!pip install torch numpy\n"
        "# After this cell finishes, go to Runtime → Restart runtime, then Run All again"
    )

    add_markdown("### 2. Create Project Directory Structure\n")
    add_code(
        "import os\n"
        "os.makedirs('agents', exist_ok=True)\n"
        "open('agents/__init__.py', 'a').close()\n"
        "print('Directory structure ready.')"
    )

    files_to_write = [
        "config.py",
        "shortestpath.py",
        "NewAgentInitObs.py",
        "agents/models.py",
        "Training.py"
    ]

    add_markdown("### 3. Write Project Files\n")

    for filepath in files_to_write:
        if os.path.exists(filepath):
            with open(filepath, 'r') as f:
                content = f.read()
            # %%writefile MUST be the absolute first characters of the source string
            add_code(f"%%writefile {filepath}\n{content}")
        else:
            print(f"Warning: {filepath} not found, skipping.")

    add_markdown("### 4. Run Training\n")
    add_code("!python Training.py")

    notebook = {
        "cells": cells,
        "metadata": {
            "accelerator": "GPU",
            "colab": {
                "provenance": [],
                "gpuType": "T4"
            },
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "codemirror_mode": {"name": "ipython", "version": 3},
                "file_extension": ".py",
                "mimetype": "text/x-python",
                "name": "python",
                "nbconvert_exporter": "python",
                "pygments_lexer": "ipython3",
                "version": "3.10.0"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 5
    }

    output_path = "Flatland_Colab_Ready.ipynb"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(notebook, f, indent=2, ensure_ascii=False)
    print(f"Successfully generated {output_path}")

if __name__ == "__main__":
    create_notebook()
